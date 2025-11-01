
import cv2
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 11  # NCC patch size (must be odd)
MAX_DISPARITY = 64  # Maximum search range (must be divisible by 16 for StereoBM)

def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity, subpixel=True):
    h, w = left_image.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    left = left_image.astype(np.float32)
    right = right_image.astype(np.float32)

    pad = window_size // 2
    left_padded = cv2.copyMakeBorder(left, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    right_padded = cv2.copyMakeBorder(right, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    eps = 1e-6

    # iterate over real image coordinates
    
    print("Computing manual NCC disparity map , it will take a while...")
    for y in range(h):
        for x in range(w):
            best_ncc = -1.0
            best_d = 0

            # Extract template centered at (y,x) -> must offset by pad
            ty = y + pad
            tx = x + pad
            template = left_padded[ty - pad : ty + pad + 1, tx - pad : tx + pad + 1]
            template_mean = np.mean(template)
            template_zero = template - template_mean
            template_norm = np.sqrt(np.sum(template_zero * template_zero)) + eps

            # If template has almost zero variance, skip (low texture)
            if template_norm < eps:
                disparity_map[y, x] = 0.0
                continue

            # Search disparities
            #try ing d from 0..max_disparity-1 but ensure candidate window stays within padded image.
            # candidate center in right_padded is (ty, tx - d)
            for d in range(max_disparity):
                cx = tx - d
                # If candidate center goes beyond left edge of padded image, break
                if cx - pad < 0 or cx + pad >= right_padded.shape[1]:
                    # out-of-bounds in padded image -> skip or break
                    continue

                candidate = right_padded[ty - pad : ty + pad + 1, cx - pad : cx + pad + 1]
                candidate_mean = np.mean(candidate)
                candidate_zero = candidate - candidate_mean
                candidate_norm = np.sqrt(np.sum(candidate_zero * candidate_zero)) + eps

                if candidate_norm < eps:
                    continue

                numerator = np.sum(template_zero * candidate_zero)
                ncc = numerator / (template_norm * candidate_norm)

                if ncc > best_ncc:
                    best_ncc = float(ncc)
                    best_d = d

            # Optional subpixel refinement: fit parabola to ncc(d-1), ncc(d), ncc(d+1)
            if subpixel and 0 < best_d < (max_disparity - 1):
                # compute ncc at best_d-1, best_d, best_d+1 to fit parabola
                ncc_vals = []
                for dd in (best_d - 1, best_d, best_d + 1):
                    cx = tx - dd
                    if cx - pad < 0 or cx + pad >= right_padded.shape[1]:
                        ncc_vals.append(-1.0)
                        continue
                    candidate = right_padded[ty - pad : ty + pad + 1, cx - pad : cx + pad + 1]
                    cand_mean = np.mean(candidate)
                    cand_zero = candidate - cand_mean
                    cand_norm = np.sqrt(np.sum(cand_zero * cand_zero)) + eps
                    if cand_norm < eps:
                        ncc_vals.append(-1.0)
                    else:
                        ncc_vals.append(np.sum(template_zero * cand_zero) / (template_norm * cand_norm))

                y1, y2, y3 = ncc_vals
                # parabola vertex offset from center: delta = 0.5*(y1 - y3)/(y1 - 2*y2 + y3)
                denom = (y1 - 2.0*y2 + y3)
                if abs(denom) > 1e-6:
                    delta = 0.5 * (y1 - y3) / denom
                    best_d = float(best_d) + delta

            disparity_map[y, x] = float(best_d)

    return disparity_map



def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    Optionally, use a mask to exclude invalid pixels.
    """
    if mask is not None:
        # Apply mask to both images to select only valid pixels
        a_valid = a[mask]
        b_valid = b[mask]
        return np.mean(np.abs(a_valid - b_valid))
    else:
        return np.mean(np.abs(a - b))

original_image_left = cv2.imread('data/left.jpg')
original_image_right = cv2.imread('data/right.jpg')
original_img_left_gray = cv2.cvtColor(original_image_left, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
original_img_right_gray = cv2.cvtColor(original_image_right, cv2.COLOR_BGR2GRAY)   # Convert to grayscale

manual_ncc_map = compute_manual_ncc_map(original_img_left_gray, original_img_right_gray, WINDOW_SIZE, MAX_DISPARITY)
#  Compute a benchmark map using cv2.StereoBM_create with the same parameters
print("\nComputing benchmark disparity map with cv2.StereoBM...")
stereo = cv2.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=WINDOW_SIZE)
benchmark_map_raw = stereo.compute(original_img_left_gray, original_img_right_gray)

# The output of StereoBM is a 16-bit signed integer (CV_16S). 
# It needs to be divided by 16 to get the actual pixel disparity.
benchmark_map = benchmark_map_raw.astype(np.float32) / 16.0

#  Visualizing both maps and compare them qualitatively
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(original_image_left, cmap='gray')
plt.title('Original Left Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(manual_ncc_map, cmap='jet')
plt.title('Manual NCC Disparity Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(benchmark_map, cmap='jet')
plt.title('OpenCV StereoBM Benchmark')
plt.axis('off')

plt.tight_layout()
plt.show()

mask = benchmark_map > 0

mae = compute_mae(manual_ncc_map, benchmark_map, mask)
print(f"\nMean Absolute Error (MAE) between manual map and benchmark: {mae:.4f} pixels")


mae_threshold = 0.7
# i have tried to make the mae below 0.7 but couldnt go lower than that it's 0.7075
if mae < mae_threshold:
    print(f"Success! The MAE is below the threshold of {mae_threshold}.")
else:
    print(f"The MAE is above the threshold of {mae_threshold}. Further optimization may be needed.")

