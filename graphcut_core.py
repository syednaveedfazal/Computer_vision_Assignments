import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def calculate_iou(pred_mask, gt_mask):

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


        self.image = image
        self.scribbles = scribbles
        self.h, self.w = image.shape[:2]
        # image.shape[:2] slices the first two dimensions of the image array
        


        self.n_components = n_components
        # n_components defines how many distinct colors make up foreground and background
        self.bins = bins
        # bins defines the resolution of the color histogram
        self.beta_sigma = beta_sigma
        # beta_sigma controls how sensitive the algorithm is to color changes between neighboring pixels
        # Low Sigma: The algorithm is hypersensitive.
        # High Sigma: The algorithm is relaxed.
        self.lam = lam
        # High Lambda, It forces pixels to group into large blobs. Result,
        # Smooth blobs, but might miss fine details 

        
        
        
        # Initialize GMMs
        self.bg_gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        self.fg_gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        # they return instance of the sklearn.mixture.GaussianMixture class
        # covariance_type='full' allows the clouds to be football shaped AND rotated in any direction
    
    
    def fit_gmm(self):

        # Define masks based on scribble colors
        # FG is White
        fg_mask_scribble = np.all(self.scribbles > 200, axis=-1)
        # checks every color channel, White color is made of high values in all channels
        # returns 2D grid of True/False values

        # BG is Red (B low, G low, R high)
        b, g, r = cv2.split(self.scribbles)
        bg_mask_scribble = (b < 50) & (g < 50) & (r > 200)
        # returns 2D grid of True/False values

        pixels = self.image.reshape(-1, 3)
        # Flatten image to (N, 3) for training
        # sklearn needs list of pixels not a 2D grid
        fg_pixels = self.image[fg_mask_scribble]
        # Pick out only the colors that is marked as Foreground
        bg_pixels = self.image[bg_mask_scribble]
        # Pick out only the colors that is marked as Background



        # Fit GMMs if scribbles exist
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
        
        
        reshaped_img = self.image.reshape(-1, 3)
        # sklearn needs list of pixels not a 2D grid
        # (Height, Width, 3) to (N_Pixels, 3)
        
        h, w = self.h, self.w
        INFINITY = 1e9

        # GMM Energy
        fg_energy_gmm = -self.fg_gmm.score_samples(reshaped_img).reshape(h, w)
        # self.fg_gmm.score_samples(reshaped_img) returns 1D array (a flat list)
        # Lower is Better
        bg_energy_gmm = -self.bg_gmm.score_samples(reshaped_img).reshape(h, w)
        # The scores came out as a long flat list
        # This creates a 2D map (a heatmap) of costs matching the original image dimensions



        # Histogram Energy
        fg_pixels = self.image[fg_mask_scribble]
        bg_pixels = self.image[bg_mask_scribble]
        
        bins = self.bins
        hist_fg, _ = np.histogramdd(fg_pixels, bins=bins, range=[(0,256), (0,256), (0,256)])
        hist_bg, _ = np.histogramdd(bg_pixels, bins=bins, range=[(0,256), (0,256), (0,256)])
        # returns 3D array



        # Smoothing & Normalization
        hist_fg += 1
        hist_bg += 1
        # adds 1 to all counts to avoid division by zero


        hist_fg /= np.sum(hist_fg)
        hist_bg /= np.sum(hist_bg)
        # converts Counts into Probabilities

      

        bin_width = 256 / bins
        img_indices = (self.image // bin_width).astype(int)
        # img_indices is a new version of original image where each pixel is within the range of 0 to bins-1 
        img_indices = np.clip(img_indices, 0, bins - 1)
        # safety guardrail


        # Probability Mapping
        prob_fg = hist_fg[img_indices[:,:,0], img_indices[:,:,1], img_indices[:,:,2]]
        prob_bg = hist_bg[img_indices[:,:,0], img_indices[:,:,1], img_indices[:,:,2]]

        fg_energy_hist = -np.log(prob_fg)
        bg_energy_hist = -np.log(prob_bg)
        # converts Probabilities into Energies
    

        # Assign Capacities
        source_gmm = bg_energy_gmm.copy()
        sink_gmm = fg_energy_gmm.copy()
        # source_caps and sink_caps are now 2D arrays which contain the costs for each pixel
        # we compare values of same pixels in fg_energy and bg_energy
        # lower links are cut

        source_hist = bg_energy_hist.copy()
        sink_hist = fg_energy_hist.copy()

        # Apply Constraints
        source_gmm[fg_mask_scribble] = INFINITY
        # If pixel is marked FG:
        # We want it connected to Source. Link to Source cannot be cut (Inf).
        # Link to Sink must be cut.
        sink_gmm[fg_mask_scribble] = 0
        source_hist[fg_mask_scribble] = INFINITY
        sink_hist[fg_mask_scribble] = 0


        source_gmm[bg_mask_scribble] = 0
        sink_gmm[bg_mask_scribble] = INFINITY
        # If pixel is marked BG:
        # We want it connected to Sink. Link to Sink cannot be cut (Inf).
        source_hist[bg_mask_scribble] = 0
        sink_hist[bg_mask_scribble] = INFINITY
        
        return source_gmm, sink_gmm, source_hist, sink_hist

    def compute_pairwise_potentials(self):
        
        img_float = self.image.astype(np.float32)
        
        # Vertical Edges 
        # Diff between pixel (y,x) and (y+1, x)
        diff_v = img_float[:-1, :, :] - img_float[1:, :, :]
        # img_float[:-1, :, :] Takes the image but chops off the last row. (ie Rows 0 to 99)
        # img_float[1:, :, :] Takes the image but chops off the first row. (ie Rows 1 to 100)
        # L2 Distance squared
        dist_v = np.sum(diff_v**2, axis=2)
        # Squares the differences to remove negative signs
        # (R_1-R_2)^2 + (G_1-G_2)^2 + (B_1-B_2)^2
        # RGB difference

        # Weight formula: lambda * exp(-||diff||^2 / (2*sigma^2))
        weights_v = self.lam * np.exp(-dist_v / (2 * self.beta_sigma**2))
        # Similar Colors have high weights, difficult to cut
        # Different Colors have low weights, easy to cut
        
        # Horizontal Edges
        # Diff between pixel (y,x) and (y, x+1)
        diff_h = img_float[:, :-1, :] - img_float[:, 1:, :]
        dist_h = np.sum(diff_h**2, axis=2)
        weights_h = self.lam * np.exp(-dist_h / (2 * self.beta_sigma**2))
        
        # Padding
        # Pad weights to match image shape (H, W) for add_grid_edges
        # PyMaxflow expects weight array size == node grid size
        # We pad the last row/col with zeros (edges that don't exist)
        final_weights_v = np.zeros((self.h, self.w), dtype=np.float64)
        final_weights_v[:-1, :] = weights_v
        
        final_weights_h = np.zeros((self.h, self.w), dtype=np.float64)
        final_weights_h[:, :-1] = weights_h
        
        return final_weights_v, final_weights_h

    def solve(self):
        
        
        print("Fitting GMMs and calculating Histogram stats...")
        fg_mask_scribble, bg_mask_scribble = self.fit_gmm()
        # trains the two Gaussian Mixture Models

        print("Computing Unary Potentials (Both Methods)...")
        s_gmm, t_gmm, s_hist, t_hist = self.compute_unary_potentials(fg_mask_scribble, bg_mask_scribble)
        # Returns four 2D arrays with scores for each pixel
        
        print("Computing Pairwise Potentials...")
        v_weights, h_weights = self.compute_pairwise_potentials()
        # Add Pairwise Edges


        # Structure arrays define the offset for the neighbor connection
        structure_right = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]) 
        structure_down = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) 

        # Solve GMM Graph
        print("Solving GMM Graph...")
        g1 = maxflow.Graph[float]()
        # Create graph with float capacities
        # our calculated costs are decimals
        
        nodeids1 = g1.add_grid_nodes((self.h, self.w))
        # Add nodes corresponding to pixels
        # This variable stores the ID numbers for all these new nodes
        # We need these IDs so we can attach wires (edges)


        g1.add_grid_tedges(nodeids1, s_gmm, t_gmm)
        # For every pixel it creates a wire from Source to Pixel with strength s_gmm[y,x]
        # and from Pixel to Sink with strength t_gmm[y,x]

   
        g1.add_grid_edges(nodeids1, weights=h_weights, structure=structure_right, symmetric=True)
        g1.add_grid_edges(nodeids1, weights=v_weights, structure=structure_down, symmetric=True)
        # Add edges. symmetric=True creates the reverse edge with same weight.

        
        g1.maxflow()
        # Solves the Max-Flow/Min-Cut optimization problem
        mask_gmm = np.logical_not(g1.get_grid_segments(nodeids1)).astype(np.uint8) * 255
        # Retrieve the segmentation result
        # get_grid_segments returns True for nodes connected to Source (FG), False for Sink (BG)

        # Solve Histogram Graph
        print("Solving Histogram Graph...")
        g2 = maxflow.Graph[float]()
        nodeids2 = g2.add_grid_nodes((self.h, self.w))
        g2.add_grid_tedges(nodeids2, s_hist, t_hist)
        g2.add_grid_edges(nodeids2, weights=h_weights, structure=structure_right, symmetric=True)
        g2.add_grid_edges(nodeids2, weights=v_weights, structure=structure_down, symmetric=True)
        g2.maxflow()
        mask_hist = np.logical_not(g2.get_grid_segments(nodeids2)).astype(np.uint8) * 255
        
        return mask_gmm, mask_hist


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

        # Run GMM
        print(f"  > Running GMM Config: Lam={p_gmm['lam']}, Sig={p_gmm['sigma']}, K={p_gmm['k']}")
        gc_gmm_run = GraphCut(img, scribbles, n_components=p_gmm['k'], beta_sigma=p_gmm['sigma'], lam=p_gmm['lam'])
        mask_gmm, _ = gc_gmm_run.solve()
        
        # Calculate and Store GMM IoU
        iou_gmm = calculate_iou(mask_gmm, gt_img)
        gmm_iou_scores.append(iou_gmm)
        print(f"    GMM IoU: {iou_gmm:.4f}")

        # Run Histogram 
        print(f"  > Running Hist Config: Lam={p_hist['lam']}, Sig={p_hist['sigma']}, Bins={p_hist['bins']}")
        gc_hist_run = GraphCut(img, scribbles, bins=p_hist['bins'], beta_sigma=p_hist['sigma'], lam=p_hist['lam'])
        _, mask_hist = gc_hist_run.solve()
        
        # Calculate and Store Hist IoU
        iou_hist = calculate_iou(mask_hist, gt_img)
        hist_iou_scores.append(iou_hist)
        print(f"Hist IoU: {iou_hist:.4f}")

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

        # Ground Truth
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
        print(f"Displaying results. Close window for next image.")
        plt.show()

    # Results
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