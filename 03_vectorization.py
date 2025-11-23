import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Use the original image for visualization background
img_path = 'data/img_mosaic.tif' 
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load the refined mask from Question 2 (Now .tif)
mask_path = 'data/final_refine_mask.tif'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    # Fallback to .png if .tif not found (for safety)
    mask_path_png = 'data/final_refine_mask.png'
    mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("Could not load mask (checked .tif and .png). Please run Question 2 first.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Implementation of Question 3: Vectorization (with Orthogonal Regularization)

def regularize_polygon(poly):
    """
    Enforces 90-degree angles on the polygon contour.
    """
    # Poly is (N, 1, 2) -> (N, 2) float
    pts = poly.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return poly

    # 1. Find dominant orientation
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    
    # 2. Rotate points to align with axes
    m_align = cv2.getRotationMatrix2D(center, angle, 1.0)
    pts_rotated = cv2.transform(np.array([pts]), m_align)[0]
    
    # 3. Rectilinear Approximation (Manhattan geometry)
    lines = [] 
    
    for i in range(len(pts_rotated)):
        p1 = pts_rotated[i]
        p2 = pts_rotated[(i+1) % len(pts_rotated)]
        
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        
        if dx > dy: # Horizontal
            val = (p1[1] + p2[1]) / 2.0
            lines.append((0, val))
        else:       # Vertical
            val = (p1[0] + p2[0]) / 2.0
            lines.append((1, val))
            
    if not lines: return poly
    
    merged_lines = []
    current_type, current_val = lines[0]
    count = 1
    
    for i in range(1, len(lines)):
        t, v = lines[i]
        if t == current_type:
            current_val = (current_val * count + v) / (count + 1)
            count += 1
        else:
            merged_lines.append((current_type, current_val))
            current_type, current_val = t, v
            count = 1
    merged_lines.append((current_type, current_val))
    
    if len(merged_lines) > 1 and merged_lines[0][0] == merged_lines[-1][0]:
        t1, v1 = merged_lines[0]
        t_last, v_last = merged_lines.pop()
        merged_lines[0] = (t1, (v1 + v_last)/2)

    # 5. Reconstruct vertices
    new_pts = []
    for i in range(len(merged_lines)):
        t1, v1 = merged_lines[i]
        t2, v2 = merged_lines[(i+1) % len(merged_lines)]
        
        if t1 == 0 and t2 == 1: 
            new_pts.append([v2, v1])
        elif t1 == 1 and t2 == 0:
            new_pts.append([v1, v2])
            
    new_pts = np.array(new_pts, dtype=np.float32)
    if len(new_pts) < 3: return poly 
    
    # 6. Rotate back
    m_restore = cv2.getRotationMatrix2D(center, angle, 1.0)
    cv2.invertAffineTransform(m_align, m_restore)
    
    final_pts = cv2.transform(np.array([new_pts]), m_restore)[0]
    
    return final_pts.reshape(-1, 1, 2).astype(np.int32)

def vectorize_contours(binary_mask, epsilon_factor=0.02):
    """
    Converts binary mask blobs into regularized vector polygons.
    """
    _, binary_mask = cv2.threshold(binary_mask, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx_poly = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx_poly) >= 3:
            ortho_poly = regularize_polygon(approx_poly)
            polygons.append(ortho_poly)
            
    return polygons

# Run Vectorization
print("Vectorizing building segments...")
vector_polygons = vectorize_contours(mask, epsilon_factor=0.01) 
print(f"Generated {len(vector_polygons)} regularized polygons.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Create White-on-Black Vector Map (final_vectorized_map.PNG)
h, w, _ = img_rgb.shape
black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

cv2.polylines(black_canvas, vector_polygons, isClosed=True, color=(255, 255, 255), thickness=2)
for poly in vector_polygons:
    pts = poly.reshape(-1, 2)
    for pt in pts:
        cv2.circle(black_canvas, (pt[0], pt[1]), radius=3, color=(255, 255, 255), thickness=-1)

# 2. Create Annotated Original Image (final_result.PNG)
# We work on a copy to avoid modifying the original image for display
annotated_img = img_rgb.copy()
# OpenCV uses BGR for saving, but we loaded as RGB. 
# For plotting, we need RGB. For saving with cv2.imwrite, we need BGR.
# Since we are constructing 'annotated_img' from 'img_rgb', it is RGB.

# Draw on the RGB copy
cv2.polylines(annotated_img, vector_polygons, isClosed=True, color=(0, 255, 255), thickness=2) # Cyan
for poly in vector_polygons:
    pts = poly.reshape(-1, 2)
    for pt in pts:
        cv2.circle(annotated_img, (pt[0], pt[1]), radius=4, color=(255, 0, 0), thickness=-1) # Red

# 3. Save Files
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save 1: Vector Map
output_path_map = os.path.join(output_dir, 'final_vectorized_map.PNG')
cv2.imwrite(output_path_map, black_canvas)
print(f"Vectorized map saved to: {output_path_map}")

# Save 2: Final Result (Annotated)
# Convert RGB to BGR for OpenCV saving
annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
output_path_result = os.path.join(output_dir, 'final_result.PNG')
cv2.imwrite(output_path_result, annotated_img_bgr)
print(f"Final Result (Annotated Image) saved to: {output_path_result}")

# 4. Visualization: Side-by-Side Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

# Left: White-on-Black Vector Map
ax1.imshow(black_canvas)
ax1.set_title("Vectorized Map (White on Black)")
ax1.axis('off')

# Right: Original Image with Vector Overlay
ax2.imshow(annotated_img)
ax2.set_title("Vectorized Map Annotated on Original")
ax2.axis('off')

plt.tight_layout()
plt.show()