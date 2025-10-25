import numpy as np
import cv2

# Load image and convert to grayscale
original_img_color = cv2.imread('bonn.jpg')  # Load 'bonn.jpg'
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
img_gray_float = gray_img.astype(np.float32) 

# Define Kernel 1
K1 = np.array([
    [0.0113, 0.0838, 0.0113],
    [0.0838, 0.6193, 0.0838],
    [0.0113, 0.0838, 0.0113]
], dtype=np.float32)

# Define Kernel 2
K2 = np.array([
    [-0.8984,  0.1472,  1.1410],
    [-1.9075,  0.1566,  2.1359],
    [-0.8659,  0.0573,  1.0337]
], dtype=np.float32)

# Decompose Kernel 1 using OpenCV SVD
w1, u1, vt1 = cv2.SVDecomp(K1)

# Decompose Kernel 2 using OpenCV SVD
w2, u2, vt2 = cv2.SVDecomp(K2)

# --- Analysis ---
print("Singular values for Kernel 1 (K1):")
print(w1.flatten())
# flatten() converts the 2D array of singular values into a 1D array for easier reading.
print("-" * 35)

print("Singular values for Kernel 2 (K2):")
print(w2.flatten())
print("-" * 35)


# Determine separability based on the number of non-zero singular values.
# A small tolerance is used to account for floating-point inaccuracies.
tolerance = 1e-6

rank_K1 = np.sum(w1 > tolerance)
rank_K2 = np.sum(w2 > tolerance)
# w1 > tolerance checks each singular value of the array w1 against the tolerance,
# returning a boolean array as 1s and 0s.
# np.sum then adds up the 1s to give the count of rank.


print("Analysis Results:")
if rank_K1 == 1:
    print("Kernel 1 is separable (Rank = 1).")
else:
    print(f"Kernel 1 is not separable (Rank = {rank_K1}).")

if rank_K2 == 1:
    print("Kernel 2 is separable (Rank = 1).")
else:
    print(f"Kernel 2 is not separable (Rank = {rank_K2}).")




# As both kernels are not separable, we will create the best rank-1 separable approximations.
# 1. Create the separable approximation for K1.
# The best rank-1 approximation is found using the highest singular value (s0)
# and its corresponding left (u0) and right (v0t) singular vectors.
# K_approx = s0 * u0 * v0t

# Extract the highest singular value (it's the first one)
s0_1 = w1[0, 0]

# Extract the first left singular vector (first column of u2)
u0_1 = u1[:, 0:1]

# Extract the first right singular vector (first row of vt2)
v0t_1 = vt1[0:1, :]

# Calculate the approximation by performing the outer product and scaling
K1_approx = s0_1 * (u0_1 @ v0t_1)
# @ operator performs matrix multiplication in numpy while * is for element-wise multiplication.

print("Original Kernel K1:\n", K1)
print("\nBest Rank-1 Separable Approximation (K1_approx):\n", K1_approx)

# 2. Filter the image with both the original kernel and the approximation.

# Filter with the original 2D kernel K1
filtered_img_original_K1 = cv2.filter2D(img_gray_float, -1, K1)

# Filter with the separable approximation K1_approx
filtered_img_approx_K1 = cv2.filter2D(img_gray_float, -1, K1_approx)

# cv2.imshow('Filtered Image with Original K1', filtered_img_original_K1)
# cv2.imshow('Filtered Image with Separable Approximation K1_approx', filtered_img_approx_K1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# As both kernels are not separable, we will create the best rank-1 separable approximations.
# Create the separable approximation for K2.
# The best rank-1 approximation is found using the highest singular value (s0)
# and its corresponding left (u0) and right (v0t) singular vectors.
# K_approx = s0 * u0 * v0t

# Extract the highest singular value (it's the first one)
s0_2 = w2[0, 0]

# Extract the first left singular vector (first column of u2)
u0_2 = u2[:, 0:1]

# Extract the first right singular vector (first row of vt2)
v0t_2 = vt2[0:1, :]

# Calculate the approximation by performing the outer product and scaling
K2_approx = s0_2 * (u0_2 @ v0t_2)
# @ operator performs matrix multiplication in numpy while * is for element-wise multiplication.

print("Original Kernel K2:\n", K2)
print("\nBest Rank-1 Separable Approximation (K2_approx):\n", K2_approx)

# 2. Filter the image with both the original kernel and the approximation.

# Filter with the original 2D kernel K2
filtered_img_original_K2 = cv2.filter2D(img_gray_float, -1, K2)

# Filter with the separable approximation K2_approx
filtered_img_approx_K2 = cv2.filter2D(img_gray_float, -1, K2_approx)

# cv2.imshow('Filtered Image with Original K1', filtered_img_original_K2)
# cv2.imshow('Filtered Image with Separable Approximation K1_approx', filtered_img_approx_K2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# --- Code for Part (c) ---

# for Kernel K1
print("\n" + "="*50)
print("Part (c): Computing the Pixel-wise Difference and Error for K1")
print("="*50)
# 1. Compute the absolute pixel-wise difference between the two results.
difference_image_K1 = cv2.absdiff(filtered_img_original_K1, filtered_img_approx_K1)
# Find and print the maximum pixel error.
max_pixel_error_K1 = np.max(difference_image_K1)
print(f"The maximum absolute pixel error between the full kernel and its approximation for K1 is: {max_pixel_error_K1}")



# for Kernel K2

print("\n" + "="*50)
print("Part (c): Computing the Pixel-wise Difference and Error")
print("="*50)

# 1. Compute the absolute pixel-wise difference between the two results.
# cv2.absdiff is ideal for this as it calculates the absolute difference
# per element between two arrays.
difference_image = cv2.absdiff(filtered_img_original_K2, filtered_img_approx_K2)

# Find and print the maximum pixel error.
# The maximum error is simply the highest pixel value in the difference image.
max_pixel_error = np.max(difference_image)

print(f"The maximum absolute pixel error between the full kernel and its approximation for K2 is: {max_pixel_error}")



