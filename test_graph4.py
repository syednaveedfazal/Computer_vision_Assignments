import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def calculate_iou(pred_mask, gt_mask):
    """
    Calculates Intersection over Union (IoU) between prediction and ground truth.
    Assumes inputs are 0 (Background) and 255 (Foreground).
    """
    # Convert to binary (0 and 1)
    # We use > 127 threshold to be safe against JPEG artifacts in GT
    p = (pred_mask > 127).astype(np.uint8)
    g = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

class GraphCut:
    def __init__(self, image, scribbles, n_components=5, bins=32, beta_sigma=10.0, lam=50.0):
        """
        Initialize the GraphCut model.
        Args:
            image: Input RGB image (H, W, 3).
            scribbles: User annotation mask (H, W, 3). Red=BG, White=FG.
            n_components: Number of GMM components.
            bins: Number of bins per channel for Color Histogram.
            beta_sigma: Sigma parameter for pairwise potential.
            lam: Lambda weight for pairwise potential.
        """
        self.image = image
        self.scribbles = scribbles
        self.h, self.w = image.shape[:2]
        
        self.n_components = n_components
        self.bins = bins 
        self.beta_sigma = beta_sigma
        self.lam = lam
        
        # Initialize GMMs
        self.bg_gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        self.fg_gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
    
    def fit_gmm(self):
        """Extracts pixels based on scribbles and fits the GMMs."""
        # FG is White
        fg_mask_scribble = np.all(self.scribbles > 200, axis=-1)
        # BG is Red (B low, G low, R high)
        b, g, r = cv2.split(self.scribbles)
        bg_mask_scribble = (b < 50) & (g < 50) & (r > 200)

        pixels = self.image.reshape(-1, 3)
        fg_pixels = self.image[fg_mask_scribble]
        bg_pixels = self.image[bg_mask_scribble]

        if len(fg_pixels) > 0:
            self.fg_gmm.fit(fg_pixels)
        else:
            print("Warning: No foreground scribbles found.")
            
        if len(bg_pixels) > 0:
            self.bg_gmm.fit(bg_pixels)
        else:
            print("Warning: No background scribbles found.")

        return fg_mask_scribble, bg_mask_scribble

    def compute_unary_potentials(self, fg_mask_scribble, bg_mask_scribble):
        """Computes unary costs using BOTH GMM and Color Histogram methods."""
        reshaped_img = self.image.reshape(-1, 3)
        h, w = self.h, self.w
        INFINITY = 1e9

        # --- 1. GMM Energy ---
        fg_energy_gmm = -self.fg_gmm.score_samples(reshaped_img).reshape(h, w)
        bg_energy_gmm = -self.bg_gmm.score_samples(reshaped_img).reshape(h, w)

        # --- 2. Histogram Energy ---
        fg_pixels = self.image[fg_mask_scribble]
        bg_pixels = self.image[bg_mask_scribble]
        
        bins = self.bins
        hist_fg, _ = np.histogramdd(fg_pixels, bins=bins, range=[(0,256), (0,256), (0,256)])
        hist_bg, _ = np.histogramdd(bg_pixels, bins=bins, range=[(0,256), (0,256), (0,256)])

        # Smoothing & Normalization
        hist_fg += 1
        hist_bg += 1
        hist_fg /= np.sum(hist_fg)
        hist_bg /= np.sum(hist_bg)

        # Lookup
        bin_width = 256 / bins
        img_indices = (self.image // bin_width).astype(int)
        img_indices = np.clip(img_indices, 0, bins - 1)

        prob_fg = hist_fg[img_indices[:,:,0], img_indices[:,:,1], img_indices[:,:,2]]
        prob_bg = hist_bg[img_indices[:,:,0], img_indices[:,:,1], img_indices[:,:,2]]

        fg_energy_hist = -np.log(prob_fg)
        bg_energy_hist = -np.log(prob_bg)

        # --- 3. Assign Capacities ---
        source_gmm = bg_energy_gmm.copy()
        sink_gmm = fg_energy_gmm.copy()
        source_hist = bg_energy_hist.copy()
        sink_hist = fg_energy_hist.copy()

        # --- 4. Apply Hard Constraints ---
        # FG Scribbles -> Connect to Source (Inf), Cut from Sink (0)
        source_gmm[fg_mask_scribble] = INFINITY
        sink_gmm[fg_mask_scribble] = 0
        source_hist[fg_mask_scribble] = INFINITY
        sink_hist[fg_mask_scribble] = 0

        # BG Scribbles -> Connect to Sink (Inf), Cut from Source (0)
        source_gmm[bg_mask_scribble] = 0
        sink_gmm[bg_mask_scribble] = INFINITY
        source_hist[bg_mask_scribble] = 0
        sink_hist[bg_mask_scribble] = INFINITY
        
        return source_gmm, sink_gmm, source_hist, sink_hist

    def compute_pairwise_potentials(self):
        """Computes pairwise weights."""
        img_float = self.image.astype(np.float32)
        
        # Vertical
        diff_v = img_float[:-1, :, :] - img_float[1:, :, :]
        dist_v = np.sum(diff_v**2, axis=2)
        weights_v = self.lam * np.exp(-dist_v / (2 * self.beta_sigma**2))
        
        # Horizontal
        diff_h = img_float[:, :-1, :] - img_float[:, 1:, :]
        dist_h = np.sum(diff_h**2, axis=2)
        weights_h = self.lam * np.exp(-dist_h / (2 * self.beta_sigma**2))
        
        # Padding
        final_weights_v = np.zeros((self.h, self.w), dtype=np.float64)
        final_weights_v[:-1, :] = weights_v
        
        final_weights_h = np.zeros((self.h, self.w), dtype=np.float64)
        final_weights_h[:, :-1] = weights_h
        
        return final_weights_v, final_weights_h

    def solve(self):
        """Returns mask_gmm, mask_hist (values 0 or 255)."""
        print("   -> Fitting GMMs and calculating Histogram stats...")
        fg_mask_scribble, bg_mask_scribble = self.fit_gmm()

        print("   -> Computing Unary Potentials (Both Methods)...")
        s_gmm, t_gmm, s_hist, t_hist = self.compute_unary_potentials(fg_mask_scribble, bg_mask_scribble)
        
        print("   -> Computing Pairwise Potentials...")
        v_weights, h_weights = self.compute_pairwise_potentials()
        
        structure_right = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]) 
        structure_down = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) 

        # Solve GMM Graph
        print("   -> Solving GMM Graph...")
        g1 = maxflow.Graph[float]()
        nodeids1 = g1.add_grid_nodes((self.h, self.w))
        g1.add_grid_tedges(nodeids1, s_gmm, t_gmm)
        g1.add_grid_edges(nodeids1, weights=h_weights, structure=structure_right, symmetric=True)
        g1.add_grid_edges(nodeids1, weights=v_weights, structure=structure_down, symmetric=True)
        g1.maxflow()
        mask_gmm = np.logical_not(g1.get_grid_segments(nodeids1)).astype(np.uint8) * 255

        # Solve Histogram Graph
        print("   -> Solving Histogram Graph...")
        g2 = maxflow.Graph[float]()
        nodeids2 = g2.add_grid_nodes((self.h, self.w))
        g2.add_grid_tedges(nodeids2, s_hist, t_hist)
        g2.add_grid_edges(nodeids2, weights=h_weights, structure=structure_right, symmetric=True)
        g2.add_grid_edges(nodeids2, weights=v_weights, structure=structure_down, symmetric=True)
        g2.maxflow()
        mask_hist = np.logical_not(g2.get_grid_segments(nodeids2)).astype(np.uint8) * 255
        
        return mask_gmm, mask_hist

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    
    # Parameters
    gmm_params = {
        "scissors.jpg":             {"lam": 4809, "sigma": 2.07,  "k": 1},
        "106024.jpg":               {"lam": 1717, "sigma": 17.58, "k": 9},
        "aero_2008_002358.jpg":     {"lam": 1614, "sigma": 17.24, "k": 2},
        "bike_2007_005878.jpg":     {"lam": 477,  "sigma": 24.82, "k": 3},
        "person7.jpg":              {"lam": 4363, "sigma": 30.36, "k": 2},
        "208001.jpg":               {"lam": 5000, "sigma": 45.3,  "k": 6}
    }

    hist_params = {
        "scissors.jpg":             {"lam": 1459, "sigma": 3.7,   "bins": 8},
        "106024.jpg":               {"lam": 4703, "sigma": 11.62, "bins": 52},
        "aero_2008_002358.jpg":     {"lam": 2333, "sigma": 12.04, "bins": 12},
        "bike_2007_005878.jpg":     {"lam": 509,  "sigma": 18.69, "bins": 28},
        "person7.jpg":              {"lam": 5000, "sigma": 37.0,  "bins": 8},
        "208001.jpg":               {"lam": 195,  "sigma": 41.6,  "bins": 4}
    }

    # Paths
    img_dir = "dataset/images"
    anno_dir = "dataset/images-labels"
    gt_dir = "dataset/images-gt"

    image_files = list(gmm_params.keys())
    num_images = len(image_files)
    
    # Lists to store IoU results
    gmm_iou_scores = []
    hist_iou_scores = []
    
    print(f"Starting Processing for {num_images} images...")

    for idx, img_name in enumerate(image_files):
        print(f"\n[{idx+1}/{num_images}] Processing: {img_name}")
        
        # Build file paths
        base_name = os.path.splitext(img_name)[0]
        anno_name = f"{base_name}-anno.png"
        gt_name = f"{base_name}.png" 
        
        img_path = os.path.join(img_dir, img_name)
        anno_path = os.path.join(anno_dir, anno_name)
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(img_path):
            print(f"  Error: Image not found {img_path}")
            continue
        if not os.path.exists(anno_path):
            print(f"  Error: Scribble not found {anno_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  Error: GT not found {gt_path}")
            continue

        # Load Data
        img = cv2.imread(img_path)
        scribbles = cv2.imread(anno_path)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

        # Get Params
        p_gmm = gmm_params[img_name]
        p_hist = hist_params[img_name]

        # --- RUN 1: GMM ---
        print(f"  > Running GMM Config: Lam={p_gmm['lam']}, Sig={p_gmm['sigma']}, K={p_gmm['k']}")
        gc_gmm_run = GraphCut(img, scribbles, n_components=p_gmm['k'], beta_sigma=p_gmm['sigma'], lam=p_gmm['lam'])
        mask_gmm, _ = gc_gmm_run.solve()
        
        # Calculate and Store GMM IoU
        iou_gmm = calculate_iou(mask_gmm, gt_img)
        gmm_iou_scores.append(iou_gmm)
        print(f"    GMM IoU: {iou_gmm:.4f}")

        # --- RUN 2: Histogram ---
        print(f"  > Running Hist Config: Lam={p_hist['lam']}, Sig={p_hist['sigma']}, Bins={p_hist['bins']}")
        gc_hist_run = GraphCut(img, scribbles, bins=p_hist['bins'], beta_sigma=p_hist['sigma'], lam=p_hist['lam'])
        _, mask_hist = gc_hist_run.solve()
        
        # Calculate and Store Hist IoU
        iou_hist = calculate_iou(mask_hist, gt_img)
        hist_iou_scores.append(iou_hist)
        print(f"    Hist IoU: {iou_hist:.4f}")

        # Display
        plt.figure(figsize=(16, 5))
        
        # Input
        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_name}\nInput")
        plt.axis('off')

        # Scribbles
        plt.subplot(1, 5, 2)
        plt.imshow(cv2.cvtColor(scribbles, cv2.COLOR_BGR2RGB))
        plt.title("Scribbles")
        plt.axis('off')

        # GT
        plt.subplot(1, 5, 3)
        plt.imshow(gt_img, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        # GMM Result
        plt.subplot(1, 5, 4)
        plt.imshow(mask_gmm, cmap='gray', vmin=0, vmax=255)
        plt.title(f"GMM Result\nIoU: {iou_gmm:.4f}")
        plt.xlabel(f"$\lambda$={p_gmm['lam']}, $\sigma$={p_gmm['sigma']}, K={p_gmm['k']}")
        plt.xticks([]) 
        plt.yticks([])

        # Hist Result
        plt.subplot(1, 5, 5)
        plt.imshow(mask_hist, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Hist Result\nIoU: {iou_hist:.4f}")
        plt.xlabel(f"$\lambda$={p_hist['lam']}, $\sigma$={p_hist['sigma']}, Bins={p_hist['bins']}")
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        print(f"  > Displaying results. Close window for next image.")
        plt.show()

    # --- FINAL SUMMARY ---
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    
    if gmm_iou_scores and hist_iou_scores:
        avg_gmm = np.mean(gmm_iou_scores)
        avg_hist = np.mean(hist_iou_scores)
        
        print(f"Processed {len(gmm_iou_scores)} images.")
        print(f"Average GMM IoU:       {avg_gmm:.4f}")
        print(f"Average Histogram IoU: {avg_hist:.4f}")
    else:
        print("No images were processed successfully.")
    print("="*40)