import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_iou(pred_mask, gt_mask):
    """Calculates IoU between prediction and ground truth."""
    if gt_mask is None: return 0.0
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    p = (pred_mask > 127).astype(np.uint8)
    g = (gt_mask > 127).astype(np.uint8)
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return 0.0 if union == 0 else intersection / union

# ==========================================
# 2. THE GRAPHCUT CLASS (Backend Logic)
# ==========================================
class GraphCut:
    def __init__(self, image, scribbles, n_components=5, bins=32, beta_sigma=10.0, lam=50.0):
        self.image = image
        self.scribbles = scribbles
        self.h, self.w = image.shape[:2]
        self.n_components = n_components
        self.bins = bins 
        self.beta_sigma = beta_sigma
        self.lam = lam
        
        self.bg_gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        self.fg_gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
    
    def fit_gmm(self):
        fg_mask_scribble = np.all(self.scribbles > 200, axis=-1)
        b, g, r = cv2.split(self.scribbles)
        bg_mask_scribble = (b < 50) & (g < 50) & (r > 200)

        pixels = self.image.reshape(-1, 3)
        fg_pixels = self.image[fg_mask_scribble]
        bg_pixels = self.image[bg_mask_scribble]

        if len(fg_pixels) > 0: self.fg_gmm.fit(fg_pixels)
        if len(bg_pixels) > 0: self.bg_gmm.fit(bg_pixels)

        return fg_mask_scribble, bg_mask_scribble

    def compute_unary_potentials(self, fg_mask_scribble, bg_mask_scribble):
        reshaped_img = self.image.reshape(-1, 3)
        h, w = self.h, self.w
        INFINITY = 1e9

        # 1. GMM Energy
        fg_energy_gmm = -self.fg_gmm.score_samples(reshaped_img).reshape(h, w)
        bg_energy_gmm = -self.bg_gmm.score_samples(reshaped_img).reshape(h, w)

        # 2. Histogram Energy
        fg_pixels = self.image[fg_mask_scribble]
        bg_pixels = self.image[bg_mask_scribble]
        
        if len(fg_pixels) == 0: fg_pixels = np.array([[255, 255, 255]])
        if len(bg_pixels) == 0: bg_pixels = np.array([[0, 0, 255]])

        hist_fg, _ = np.histogramdd(fg_pixels, bins=self.bins, range=[(0,256), (0,256), (0,256)])
        hist_bg, _ = np.histogramdd(bg_pixels, bins=self.bins, range=[(0,256), (0,256), (0,256)])

        hist_fg += 1
        hist_bg += 1
        hist_fg /= np.sum(hist_fg)
        hist_bg /= np.sum(hist_bg)

        bin_width = 256 / self.bins
        img_indices = (self.image // bin_width).astype(int)
        img_indices = np.clip(img_indices, 0, self.bins - 1)

        prob_fg = hist_fg[img_indices[:,:,0], img_indices[:,:,1], img_indices[:,:,2]]
        prob_bg = hist_bg[img_indices[:,:,0], img_indices[:,:,1], img_indices[:,:,2]]

        fg_energy_hist = -np.log(prob_fg)
        bg_energy_hist = -np.log(prob_bg)

        # 3. Assign Caps
        source_gmm = bg_energy_gmm.copy()
        sink_gmm = fg_energy_gmm.copy()
        source_hist = bg_energy_hist.copy()
        sink_hist = fg_energy_hist.copy()

        # 4. Hard Constraints
        for s_caps, t_caps in [(source_gmm, sink_gmm), (source_hist, sink_hist)]:
            s_caps[fg_mask_scribble] = INFINITY
            t_caps[fg_mask_scribble] = 0
            s_caps[bg_mask_scribble] = 0
            t_caps[bg_mask_scribble] = INFINITY
        
        return source_gmm, sink_gmm, source_hist, sink_hist

    def compute_pairwise_potentials(self):
        img_float = self.image.astype(np.float32)
        diff_v = img_float[:-1, :, :] - img_float[1:, :, :]
        dist_v = np.sum(diff_v**2, axis=2)
        weights_v = self.lam * np.exp(-dist_v / (2 * self.beta_sigma**2))
        
        diff_h = img_float[:, :-1, :] - img_float[:, 1:, :]
        dist_h = np.sum(diff_h**2, axis=2)
        weights_h = self.lam * np.exp(-dist_h / (2 * self.beta_sigma**2))
        
        final_weights_v = np.zeros((self.h, self.w), dtype=np.float64)
        final_weights_v[:-1, :] = weights_v
        final_weights_h = np.zeros((self.h, self.w), dtype=np.float64)
        final_weights_h[:, :-1] = weights_h
        return final_weights_v, final_weights_h

    def solve(self):
        fg_mask_scribble, bg_mask_scribble = self.fit_gmm()
        
        if np.sum(fg_mask_scribble) == 0 or np.sum(bg_mask_scribble) == 0:
            return None, None

        s_gmm, t_gmm, s_hist, t_hist = self.compute_unary_potentials(fg_mask_scribble, bg_mask_scribble)
        v_weights, h_weights = self.compute_pairwise_potentials()
        
        structure_right = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]) 
        structure_down = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) 

        # GMM
        g1 = maxflow.Graph[float]()
        nodeids1 = g1.add_grid_nodes((self.h, self.w))
        g1.add_grid_tedges(nodeids1, s_gmm, t_gmm)
        g1.add_grid_edges(nodeids1, weights=h_weights, structure=structure_right, symmetric=True)
        g1.add_grid_edges(nodeids1, weights=v_weights, structure=structure_down, symmetric=True)
        g1.maxflow()
        mask_gmm = np.logical_not(g1.get_grid_segments(nodeids1)).astype(np.uint8) * 255

        # Hist
        g2 = maxflow.Graph[float]()
        nodeids2 = g2.add_grid_nodes((self.h, self.w))
        g2.add_grid_tedges(nodeids2, s_hist, t_hist)
        g2.add_grid_edges(nodeids2, weights=h_weights, structure=structure_right, symmetric=True)
        g2.add_grid_edges(nodeids2, weights=v_weights, structure=structure_down, symmetric=True)
        g2.maxflow()
        mask_hist = np.logical_not(g2.get_grid_segments(nodeids2)).astype(np.uint8) * 255
        
        return mask_gmm, mask_hist

# ==========================================
# 3. THE TKINTER GUI CLASS
# ==========================================
class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Graph Cut Tool")
        self.root.geometry("1200x800")
        
        # --- FIX: Initialize params BEFORE setup_ui ---
        self.params = {"lam": 2000, "sigma": 10.0, "k": 5, "bins": 32}
        
        self.img = None
        self.gt_img = None
        self.scribble_mask = None
        self.drawing = False
        self.brush_color = 'fg' # 'fg' or 'bg'
        self.brush_size = 5
        self.tk_image = None
        self.masks_result = None
        
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Control Panel (Left) ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="ns")
        
        lbl_file = ttk.Label(control_frame, text="File Operations", font=("Arial", 10, "bold"))
        lbl_file.pack(pady=(0, 5), anchor="w")
        
        btn_load = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        btn_load.pack(fill="x", pady=2)
        
        btn_load_gt = ttk.Button(control_frame, text="Load Ground Truth (Opt)", command=self.load_gt)
        btn_load_gt.pack(fill="x", pady=2)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        lbl_draw = ttk.Label(control_frame, text="Scribble Mode", font=("Arial", 10, "bold"))
        lbl_draw.pack(pady=(0, 5), anchor="w")
        
        self.var_mode = tk.StringVar(value="fg")
        
        style = ttk.Style()
        style.configure("FG.TButton", foreground="green")
        btn_fg = ttk.Button(control_frame, text="FOREGROUND (White)", command=lambda: self.set_mode('fg'), style="FG.TButton")
        btn_fg.pack(fill="x", pady=2)
        
        style.configure("BG.TButton", foreground="red")
        btn_bg = ttk.Button(control_frame, text="BACKGROUND (Red)", command=lambda: self.set_mode('bg'), style="BG.TButton")
        btn_bg.pack(fill="x", pady=2)

        lbl_size = ttk.Label(control_frame, text="Brush Size:")
        lbl_size.pack(pady=(10, 0), anchor="w")
        self.scale_brush = ttk.Scale(control_frame, from_=1, to=30, orient="horizontal", command=self.update_brush_size)
        self.scale_brush.set(5)
        self.scale_brush.pack(fill="x", pady=5)
        
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        lbl_params = ttk.Label(control_frame, text="Parameters", font=("Arial", 10, "bold"))
        lbl_params.pack(pady=(0, 5), anchor="w")
        
        self.create_slider(control_frame, "Lambda", 0, 5000, 2000, "lam")
        # Sigma slider with 2 decimal precision
        self.create_slider(control_frame, "Sigma", 0.01, 100.0, 10.0, "sigma", resolution=0.01)
        self.create_slider(control_frame, "K (GMM)", 1, 10, 5, "k")
        self.create_slider(control_frame, "Bins (Hist)", 4, 64, 32, "bins")

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        btn_run = ttk.Button(control_frame, text="RUN SEGMENTATION", command=self.run_segmentation)
        btn_run.pack(fill="x", pady=10, ipady=5)
        
        btn_reset = ttk.Button(control_frame, text="Reset Scribbles", command=self.reset_scribbles)
        btn_reset.pack(fill="x", pady=2)
        
        btn_save = ttk.Button(control_frame, text="Save Results", command=self.save_results)
        btn_save.pack(fill="x", pady=2)

        self.lbl_status = ttk.Label(control_frame, text="Status: Ready", foreground="blue")
        self.lbl_status.pack(side="bottom", pady=10)

        # --- Canvas Area (Right) ---
        canvas_container = ttk.Frame(self.root)
        canvas_container.grid(row=0, column=1, sticky="nsew")
        
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, bg="gray")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        h_scroll = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
    def create_slider(self, parent, label, min_v, max_v, default, param_key, resolution=1.0):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        # Initial label formatting
        val_str = f"{default:.2f}" if resolution < 1 else f"{int(default)}"
        lbl = ttk.Label(frame, text=f"{label}: {val_str}")
        lbl.pack(side="top", anchor="w")
        
        def update_val(val):
            val = float(val)
            self.params[param_key] = val
            if resolution < 1:
                lbl.config(text=f"{label}: {val:.2f}")
            else:
                lbl.config(text=f"{label}: {int(val)}")

        scale = ttk.Scale(frame, from_=min_v, to=max_v, orient="horizontal", command=update_val)
        scale.set(default)
        scale.pack(fill="x")
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.img = cv2.imread(file_path)
            if self.img is None:
                messagebox.showerror("Error", "Could not read image")
                return
            
            self.scribble_mask = np.zeros_like(self.img)
            self.display_image_on_canvas()
            self.lbl_status.config(text=f"Loaded: {os.path.basename(file_path)}")

    def load_gt(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg")])
        if file_path:
            self.gt_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            messagebox.showinfo("Info", "Ground Truth Loaded (will show IoU on run)")

    def display_image_on_canvas(self):
        if self.img is None: return
        
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(im_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def set_mode(self, mode):
        self.brush_color = mode
        color = "Green" if mode == 'fg' else "Red"
        self.lbl_status.config(text=f"Mode: {color} Brush")

    def update_brush_size(self, val):
        self.brush_size = int(float(val))

    def start_draw(self, event):
        if self.img is None: return
        self.drawing = True
        self.last_x = self.canvas.canvasx(event.x)
        self.last_y = self.canvas.canvasy(event.y)

    def draw(self, event):
        if not self.drawing or self.img is None: return
        
        curr_x = self.canvas.canvasx(event.x)
        curr_y = self.canvas.canvasy(event.y)
        
        tk_color = "white" if self.brush_color == 'fg' else "red"
        self.canvas.create_line(self.last_x, self.last_y, curr_x, curr_y, 
                                fill=tk_color, width=self.brush_size, capstyle=tk.ROUND, smooth=True)
        
        cv_color = (255, 255, 255) if self.brush_color == 'fg' else (0, 0, 255)
        
        pt1 = (int(self.last_x), int(self.last_y))
        pt2 = (int(curr_x), int(curr_y))
        
        cv2.line(self.scribble_mask, pt1, pt2, cv_color, self.brush_size)
        
        self.last_x = curr_x
        self.last_y = curr_y

    def stop_draw(self, event):
        self.drawing = False

    def reset_scribbles(self):
        if self.img is not None:
            self.scribble_mask = np.zeros_like(self.img)
            self.display_image_on_canvas()
            self.lbl_status.config(text="Scribbles Cleared")

    def run_segmentation(self):
        if self.img is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return

        self.lbl_status.config(text="Processing... Please wait.")
        self.root.update()
        
        try:
            gc = GraphCut(self.img, self.scribble_mask, 
                          n_components=int(self.params['k']),
                          bins=int(self.params['bins']),
                          beta_sigma=self.params['sigma'],
                          lam=self.params['lam'])
            
            mask_gmm, mask_hist = gc.solve()
            
            if mask_gmm is None:
                messagebox.showerror("Error", "Please draw both Foreground (White) and Background (Red) scribbles.")
                self.lbl_status.config(text="Failed: Missing annotations")
                return

            self.masks_result = (mask_gmm, mask_hist)
            self.show_results_window(mask_gmm, mask_hist)
            self.lbl_status.config(text="Processing Complete.")

        except Exception as e:
            print(e)
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            self.lbl_status.config(text="Error")

    def show_results_window(self, mask_gmm, mask_hist):
        # Calculate IoU if GT exists
        iou_gmm_txt = ""
        iou_hist_txt = ""
        if self.gt_img is not None:
            iou_gmm = calculate_iou(mask_gmm, self.gt_img)
            iou_hist = calculate_iou(mask_hist, self.gt_img)
            iou_gmm_txt = f" (IoU: {iou_gmm:.3f})"
            iou_hist_txt = f" (IoU: {iou_hist:.3f})"

        # Create Overlays
        # GMM Overlay
        overlay_gmm = self.img.copy()
        overlay_gmm[mask_gmm == 255] = cv2.addWeighted(self.img[mask_gmm == 255], 0.6, 
                                                       np.full_like(self.img[mask_gmm == 255], (0,255,0)), 0.4, 0)
        
        # Hist Overlay
        overlay_hist = self.img.copy()
        overlay_hist[mask_hist == 255] = cv2.addWeighted(self.img[mask_hist == 255], 0.6, 
                                                         np.full_like(self.img[mask_hist == 255], (0,255,0)), 0.4, 0)
        
        # Convert Scribble Mask to RGB for display (BGR -> RGB)
        scribbles_rgb = cv2.cvtColor(self.scribble_mask, cv2.COLOR_BGR2RGB)

        # Matplotlib Grid Layout (2 Rows x 3 Columns)
        plt.figure(figsize=(15, 8))
        
        # --- Row 1 ---
        plt.subplot(2, 3, 1)
        plt.title("User Annotation\n(Red=BG, White=FG)")
        plt.imshow(scribbles_rgb)
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title(f"GMM Result{iou_gmm_txt}")
        plt.imshow(mask_gmm, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title("GMM Overlay")
        plt.imshow(cv2.cvtColor(overlay_gmm, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # --- Row 2 ---
        plt.subplot(2, 3, 4)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title(f"Histogram Result{iou_hist_txt}")
        plt.imshow(mask_hist, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title("Histogram Overlay")
        plt.imshow(cv2.cvtColor(overlay_hist, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def save_results(self):
        if self.masks_result is None:
            messagebox.showwarning("Warning", "Run segmentation first!")
            return
            
        cv2.imwrite("output_gmm_mask.png", self.masks_result[0])
        cv2.imwrite("output_hist_mask.png", self.masks_result[1])
        messagebox.showinfo("Saved", "Saved masks as 'output_gmm_mask.png' and 'output_hist_mask.png'")

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()