import numpy as np
import cv2
import skimage
from skimage import filters
import matplotlib
import math
import matplotlib.pyplot as plt
import os

np.set_printoptions(suppress=True)



# 1. Load Data
# Load the original image (for external energy)
img_path = 'data/final_mask.PNG'
img = cv2.imread(img_path)
if img is None:
    # Fallback if running locally without the exact file, though it's expected to be there
    raise FileNotFoundError(f"Could not load image {img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the mask from the previous step (initialization)
mask_path = 'data/final_mask.PNG'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Could not load mask {mask_path}. Please run the previous step first.")

# 2. Define Energy Functions

def get_external_energy(image):
    """
    Computes the External Energy based on Image Gradients.
    E_ext = -|grad(I)|^2
    The snake is attracted to high gradients (edges).
    """
    # Apply Gaussian blur to smooth noise and widen the capture range of edges
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    
    # Compute gradients using Sobel
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    
    # Magnitude squared
    gradient_magnitude_sq = grad_x**2 + grad_y**2
    
    # Normalize to 0-1 range for easier weighting
    if np.max(gradient_magnitude_sq) > 0:
        gradient_magnitude_sq /= np.max(gradient_magnitude_sq)
    
    # Invert because we want to MINIMIZE energy (high gradient = low energy)
    # E_ext = -Magnitude
    external_energy = -gradient_magnitude_sq
    
    return external_energy

# Pre-compute the external energy map (static)
external_energy_map = get_external_energy(img_gray)

def get_internal_energy(point, prev_point, next_point, avg_dist, alpha, beta):
    """
    Computes Internal Energy: E_int = alpha*E_cont + beta*E_curv
    """
    # 1. Continuity Energy (Elasticity)
    # Penalize deviation from the average distance between points (d_bar)
    dist = np.linalg.norm(point - prev_point)
    e_cont = (avg_dist - dist) ** 2
    
    # 2. Curvature Energy (Stiffness)
    # Penalize sharp angles (Second derivative)
    curvature_vec = prev_point - 2*point + next_point
    e_curv = np.linalg.norm(curvature_vec) ** 2
    
    return alpha * e_cont + beta * e_curv

# 3. Optimization Loop (Greedy Algorithm)

def optimize_snake(snake, energy_map, alpha=0.01, beta=0.1, gamma=1.5, iterations=50, window_size=3):
    """
    Iteratively minimizes the total energy using a greedy approach.
    """
    new_snake = np.copy(snake)
    n_points = len(snake)
    offset = window_size // 2
    
    for it in range(iterations):
        # Calculate average distance for continuity (Elasticity)
        dists = np.linalg.norm(new_snake - np.roll(new_snake, 1, axis=0), axis=1)
        avg_dist = np.mean(dists)
        
        moved_count = 0
        
        for i in range(n_points):
            curr_p = new_snake[i]
            prev_p = new_snake[(i - 1) % n_points]
            next_p = new_snake[(i + 1) % n_points]
            
            min_energy = float('inf')
            best_pos = curr_p
            
            # Search local neighborhood
            for dy in range(-offset, offset + 1):
                for dx in range(-offset, offset + 1):
                    candidate_p = curr_p + np.array([dx, dy])
                    
                    # Boundary check
                    cx, cy = int(candidate_p[0]), int(candidate_p[1])
                    if 0 <= cy < energy_map.shape[0] and 0 <= cx < energy_map.shape[1]:
                        
                        e_int = get_internal_energy(candidate_p, prev_p, next_p, avg_dist, alpha, beta)
                        e_ext = gamma * energy_map[cy, cx]
                        
                        total_energy = e_int + e_ext
                        
                        if total_energy < min_energy:
                            min_energy = total_energy
                            best_pos = candidate_p
            
            if not np.array_equal(best_pos, curr_p):
                new_snake[i] = best_pos
                moved_count += 1
        
        if moved_count == 0:
            break
            
    return new_snake

# 4. Process All Buildings
print("Detecting contours...")
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

refined_contours = []
print(f"Found {len(contours)} candidate regions. Optimizing...")

for i, contour in enumerate(contours):
    # Filter tiny noise that might have persisted (safety check)
    if cv2.contourArea(contour) < 500: 
        continue
        
    # Reshape to [N, 2] float array
    snake = contour.reshape(-1, 2).astype(float)
    
    # Run optimization for this specific building
    # We use relatively high beta (stiffness) to keep building walls straight
    optimized_snake = optimize_snake(
        snake, 
        external_energy_map, 
        alpha=2,   # Continuity
        beta=0.4,    # Stiffness (Increased for smoothing)
        gamma=1.0,   # Edge Attraction
        iterations=60,
        window_size=5
    )
    
    # Convert back to integer format for drawing
    refined_contours.append(optimized_snake.astype(np.int32))
    
    # Update progress
    print(f"Finished optimizing segment {i+1}/{len(contours)}")

print(f"Optimization complete for {len(refined_contours)} buildings.")

# 5. Create Outputs
# Output 1: Binary Mask (Black background, White buildings) - For Saving
h, w = img_gray.shape
binary_output_mask = np.zeros((h, w), dtype=np.uint8)
cv2.drawContours(binary_output_mask, refined_contours, -1, color=255, thickness=-1)

# Output 2: Visualization Image (Gray background, White buildings, Black outlines) - For Plotting
vis_img = np.full((h, w), 220, dtype=np.uint8) # Light Gray background
cv2.drawContours(vis_img, refined_contours, -1, color=255, thickness=-1) # White Fill
cv2.drawContours(vis_img, refined_contours, -1, color=0, thickness=2)    # Black Outline

# 6. Save Result
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# UPDATED: Save as .tif instead of .png
output_path = os.path.join(output_dir, 'final_refine_mask.tif')
cv2.imwrite(output_path, binary_output_mask)
print(f"Final refined binary mask saved to: {output_path}")

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# Show Original Mask
ax[0].imshow(mask, cmap='gray')
ax[0].set_title("Original Mask (Input)")
ax[0].axis('off')

# Show Refined Result (Visualization style)
ax[1].imshow(vis_img, cmap='gray')
ax[1].set_title("Refined Segmentation (Snake Optimization)")
ax[1].axis('off')

plt.tight_layout()
plt.show()