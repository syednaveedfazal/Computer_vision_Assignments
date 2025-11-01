import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
# --- Helper Constants ---
WEAK_PIXEL = 75
STRONG_PIXEL = 255


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.
    We let cv2.GaussianBlur determine the kernel size from sigma
    by setting ksize=(0, 0).
    """
    # sigmaY is set to sigmaX if 0
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def compute_gradients(img):
    """
    Compute gradient magnitude and direction (Sobel-based).
    We use ksize=3 and CV_64F for precision, matching the
    cv2.Canny(L2gradient=True) default.
    Return: gradient_magnitude, gradient_angle (in radians)
    """
    # Compute gradients in 64-bit float to avoid overflow
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude (L2 norm)
    # G = sqrt(Gx^2 + Gy^2)
    # THIS IS THE CORRECTED LINE:
    gradient_magnitude = np.hypot(sobel_x, sobel_y) 

    # Compute direction in radians [-pi, pi]
    # theta = atan2(Gy, Gx)
    gradient_angle = np.arctan2(sobel_y, sobel_x)

    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    mag: Gradient magnitude
    ang: Gradient angle in radians
    """
    h, w = mag.shape
    # Create a zero-initialized array for the thinned edges
    nms_result = np.zeros_like(mag, dtype=np.float32)

    # Convert angles from radians to degrees and make positive [0, 180]
    ang_deg = np.degrees(ang)
    ang_deg[ang_deg < 0] += 180

    # Iterate over all pixels (skip borders)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            angle = ang_deg[y, x]
            m = mag[y, x]

            # Find neighbors based on quantized angle
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                # Angle 0 deg: Horizontal edge
                n1, n2 = mag[y, x - 1], mag[y, x + 1]
            elif (22.5 <= angle < 67.5):
                # Angle 45 deg: Positive diagonal
                n1, n2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            elif (67.5 <= angle < 112.5):
                # Angle 90 deg: Vertical edge
                n1, n2 = mag[y - 1, x], mag[y + 1, x]
            elif (112.5 <= angle < 157.5):
                # Angle 135 deg: Negative diagonal
                n1, n2 = mag[y - 1, x - 1], mag[y + 1, x + 1]

            # Suppress if not a local maximum
            if (m >= n1) and (m >= n2):
                nms_result[y, x] = m
            # else: nms_result[y, x] is already 0

    return nms_result


def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.
    Returns an edge map with WEAK_PIXEL, STRONG_PIXEL, or 0.
    """
    result = np.zeros_like(nms, dtype=np.uint8)

    # Find strong and weak pixels
    strong_y, strong_x = np.where(nms >= high)
    weak_y, weak_x = np.where((nms < high) & (nms >= low))

    result[strong_y, strong_x] = STRONG_PIXEL
    result[weak_y, weak_x] = WEAK_PIXEL

    return result


def hysteresis(edge_map, weak, strong):
    """
    Perform edge tracking by hysteresis using Breadth-First Search (BFS).
    Promotes weak pixels to strong if they are connected to a strong pixel.
    """
    h, w = edge_map.shape
    final_edges = np.zeros((h, w), dtype=np.uint8)

    # Get initial seed points (all strong pixels)
    strong_y, strong_x = np.where(edge_map == strong)
    
    # Use deque as an efficient stack/queue for BFS/DFS
    stack = deque(zip(strong_y, strong_x))
    
    # Mark all strong pixels as final edges
    final_edges[strong_y, strong_x] = 255

    while stack:
        y, x = stack.pop()

        # Check all 8 neighbors
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if (ny == y and nx == x):
                    continue  # Skip self

                # Check if neighbor is a weak pixel and not yet visited
                if (edge_map[ny, nx] == weak) and (final_edges[ny, nx] == 0):
                    final_edges[ny, nx] = 255  # Promote to strong
                    stack.append((ny, nx))  # Add to stack to check its neighbors

    return final_edges


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.
    """
    # Normalize to 0-1 float images
    manual_norm = manual_edges.astype(np.float32) / 255.0
    cv_norm = cv_edges.astype(np.float32) / 255.0
    
    # --- MAD ---
    mad = np.mean(np.abs(manual_norm - cv_norm))

    # --- F1 Score ---
    manual_flat = (manual_norm.ravel() > 0.5)
    cv_flat = (cv_norm.ravel() > 0.5)

    eps = 1e-6  # Epsilon for numerical stability
    
    tp = np.sum(manual_flat & cv_flat) # True Positives
    fp = np.sum(manual_flat & ~cv_flat) # False Positives
    fn = np.sum(~manual_flat & cv_flat) # False Negatives

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    return mad, precision, recall, f1


IMAGE_FILE = os.path.join('data', 'bonn.jpg')
SIGMA = 1.0        # Sigma for Gaussian blur
LOW_THRESHOLD = 100  # Low threshold for Canny
HIGH_THRESHOLD = 200 # High threshold for Canny

# TODO: 1. Load the grayscale image 'bonn.jpg'
img_bgr = cv2.imread(IMAGE_FILE)
if img_bgr is None:
    print(f"Error: Could not load image '{IMAGE_FILE}'.")
    # As a fallback, create a dummy image
    img_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print("Using a random dummy image instead.")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# TODO: 2. Smooth the image using your Gaussian function
print(f"1. Applying Gaussian smoothing (sigma={SIGMA})...")
img_smooth = gaussian_smoothing(img_gray, SIGMA)

# TODO: 3. Compute gradients (magnitude and direction)
print("2. Computing gradients (Sobel)...")
grad_mag, grad_ang = compute_gradients(img_smooth)

# TODO: 4. Apply non-maximum suppression
print("3. Applying Non-Maximum Suppression...")
nms_img = nonmax_suppression(grad_mag, grad_ang)

# TODO: 5. Apply double threshold
print(f"4. Applying double threshold (Low={LOW_THRESHOLD}, High={HIGH_THRESHOLD})...")
threshold_img = double_threshold(nms_img, LOW_THRESHOLD, HIGH_THRESHOLD)

# TODO: 6. Perform hysteresis to obtain final edges
print("5. Performing hysteresis (edge tracking)...")
manual_edges = hysteresis(threshold_img, WEAK_PIXEL, STRONG_PIXEL)

# TODO: 7. Compare your result with cv2.Canny
print("6. Generating OpenCV benchmark...")
# We use apertureSize=3 and L2gradient=True to match our implementation
cv_edges = cv2.Canny(img_gray, 
                     LOW_THRESHOLD, 
                     HIGH_THRESHOLD, 
                     apertureSize=3, 
                     L2gradient=True)

print("7. Computing metrics...")
mad, precision, recall, f1 = compute_metrics(manual_edges, cv_edges)

print("\n--- METRICS (Manual vs. OpenCV) ---")
print(f"Mean Absolute Difference (MAD): {mad:.4f}")
print(f"Precision:                    {precision:.4f}")
print(f"Recall:                       {recall:.4f}")
print(f"F1-Score:                     {f1:.4f}")
print("---------------------------------")

# Check thresholds from previous exercise
if mad <= 0.07 and f1 >= 0.6:
    print("✅ SUCCESS: Metrics are within the target range (MAD <= 0.07, F1 >= 0.6).")
else:
    print("❌ NOTE: Metrics are outside the target range.")

# TODO: 8. Display original image, your edges, and OpenCV edges
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(manual_edges, cmap='gray')
plt.title('My Canny Implementation')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv_edges, cmap='gray')
plt.title('OpenCV cv2.Canny')
plt.axis('off')

plt.tight_layout()
plt.show()