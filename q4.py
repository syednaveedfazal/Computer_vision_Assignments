import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Helper Function to Analyze Kernels
# ==============================================================================

def analyze_kernel(K):
    """
    Performs SVD on a kernel and returns its components and separability status.
    """
    K_float = K.astype(np.float32)
    w, u, vt = cv2.SVDecomp(K_float)
    tolerance = 1e-6
    is_separable = w[1][0] < tolerance if len(w) > 1 else True
    return is_separable, w, u, vt

# ==============================================================================
# Main Script
# ==============================================================================

# 1. Define all kernels to be tested
kernels = {
    "Kernel 1": np.array([
        [0.0113, 0.0838, 0.0113],
        [0.0838, 0.6193, 0.0838],
        [0.0113, 0.0838, 0.0113]
    ]),
    "Kernel 2": np.array([
        [-0.8984, 0.1472, 1.1410],
        [-1.9075, 0.1566, 2.1359],
        [-0.8659, 0.0573, 1.0337]
    ])
}

# 2. Load the image that will be used for filtering
try:
    img_color = cv2.imread('bonn.jpg')
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray_float = img_gray.astype(np.float32) / 255.0
except Exception as e:
    print(f"Fatal Error: Could not load 'bonn.jpg'. Please ensure it's in the correct directory.")
    exit()

# 3. Loop through each kernel, analyze it, and act accordingly
for kernel_name, K_original in kernels.items():
    print(f"--- Analyzing {kernel_name} ---")
    
    is_separable, w, u, vt = analyze_kernel(K_original)

    # --- Check if the kernel is separable ---
    if is_separable:
        print(f"Result: {kernel_name} IS separable. Skipping approximation steps.")
        print("-" * 40 + "\n")
        continue

    # --- If the code reaches here, the kernel is NOT separable ---
    print(f"Result: {kernel_name} IS NOT separable. Proceeding with approximation.")
    print("Singular Values:", w.flatten())

    # 4. Create the approximation using the highest singular value
    print("\nCreating the best rank-1 (separable) approximation...")
    first_singular_value = w[0][0]
    first_u_column = u[:, 0]
    first_vt_row = vt[0, :]
    K_approximated_2D = first_singular_value * (first_u_column.reshape(-1, 1) @ first_vt_row.reshape(1, -1))
    print("Approximated 2D Kernel:\n", K_approximated_2D)

    # 5. Filter the image with both the original and approximated kernels
    print("\nFiltering image...")
    filtered_original = cv2.filter2D(img_gray_float, -1, K_original)
    
    s_sqrt = np.sqrt(first_singular_value)
    col_vector_sep = s_sqrt * first_u_column
    row_vector_sep = s_sqrt * first_vt_row
    filtered_approximated = cv2.sepFilter2D(img_gray_float, -1, row_vector_sep, col_vector_sep)
    
    # ==============================================================================
    # 6. (NEW) Compute and Print the Pixel-wise Error
    # ==============================================================================
    print("\nComputing the difference between the two filtered results...")

    # Calculate the absolute difference between the two images, pixel by pixel.
    # The images are floats in the [0, 1] range.
    difference_image = cv2.absdiff(filtered_original, filtered_approximated)

    # Find the maximum value in the difference image. This is our max error.
    max_pixel_error = np.max(difference_image)

    print(f"\n>>> The MAXIMUM PIXEL-WISE ERROR introduced by the approximation is: {max_pixel_error:.6f}")
    
    # ==============================================================================
    
    # 7. Display all the results, including the new difference image
    print("\nDisplaying comparison...")
    
    # We'll use a 2x2 grid to show everything
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(filtered_original, cmap='gray')
    plt.title(f'Filtered with Original {kernel_name}')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(filtered_approximated, cmap='gray')
    plt.title('Filtered with Separable Approximation')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    # Display the difference image. A 'hot' colormap makes errors easy to see.
    # Black/dark areas have low error; bright yellow/white areas have high error.
    im = plt.imshow(difference_image, cmap='hot')
    plt.title(f'Absolute Difference (Max Error: {max_pixel_error:.4f})')
    plt.axis('off')
    plt.colorbar(im) # Add a colorbar to show the scale of the error

    plt.suptitle(f'Analysis of Non-Separable {kernel_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("-" * 40 + "\n")