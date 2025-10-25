import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

'''
TODO: Load the image 'bonn.jpg' and convert it to grayscale
'''

# Load image and convert to grayscale
original_img_color = cv2.imread('bonn.jpg')  # Load 'bonn.jpg'
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part a)
# ==============================================================================
print("\n--- a) Calculating Integral Image ---")


def calculate_integral_image(img):
    """
    Calculate the integral image (summed area table).
    Each pixel contains the sum of all pixels above and to the left.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Integral image with dimensions (height+1, width+1)
    
    TODO:
    1. Create an integral image array     
    2. Iterate through all pixels and compute integral values
    
        """
    height, width = img.shape
    
    # 1. Create an integral image array with an extra row and column for padding
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    
    # 2. Iterate through all pixels and compute integral values using the summed-area table formula
    for r in range(height):
        for c in range(width):
            integral[r+1, c+1] = img[r, c] + integral[r, c+1] + integral[r+1, c] - integral[r, c]
            # we add img[r, c] to the sum of the area above and to the left,
            # and subtract the overlapping area that was added twice.
            # integral[r, c] is the value at the top-left corner of the current pixel.
            # The area was overlapped be
            
    return integral


# Calculate integral image
integral_img = calculate_integral_image(gray_img)  # Call calculate_integral_image()

print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part b)
# ==============================================================================
print("\n--- b) Computing Mean Using Integral Image ---")


def mean_using_integral(integral, top_left, bottom_right):
    """
    Calculate mean gray value using integral image.
    Time Complexity: O(1)

    Args:
        integral: The integral image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract coordinates from top_left and bottom_right
    2. Adjust indices for integral image (remember it's 1-indexed)
    3. Return Sum / number_of_pixels
    """
    # 1. Extract coordinates from top_left and bottom_right
    r1, c1 = top_left
    r2, c2 = bottom_right
    
    # 2. Adjust indices for integral image to find the sum of the region
    # The sum of a rectangle (r1, c1) to (r2, c2) is calculated as:
    # I(r2, c2) - I(r2, c1-1) - I(r1-1, c2) + I(r1-1, c1-1)
    # Due to our +1 padding, this translates to the following indices:
    bottom_right_sum = integral[r2 + 1, c2 + 1]
    bottom_left_sum = integral[r2 + 1, c1]
    top_right_sum = integral[r1, c2 + 1]
    top_left_sum = integral[r1, c1]
    # The -1 from the formula and the +1 from the padding "cancel out" for the row index, leaving just r1 and c1 as is.

    region_sum = bottom_right_sum - bottom_left_sum - top_right_sum + top_left_sum

    # 3. Calculate the number of pixels and then the mean
    number_of_pixels = (r2 - r1 + 1) * (c2 - c1 + 1)
    
    # Avoid division by zero for an empty region
    if number_of_pixels == 0:
        return 0
        
    return region_sum / number_of_pixels



# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right)  # Call mean_using_integral()

print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")

# ==============================================================================
# 3. Compute Mean by Direct Summation (Part c)
# ==============================================================================
print("\n--- c) Computing Mean by Direct Summation ---")


def mean_by_direct_sum(img, top_left, bottom_right):
    """
    Calculate mean gray value by summing all pixels in region.
    Time Complexity: O(w * h) where w and h are region dimensions

    Args:
        img: The grayscale image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract the region from the image using array slicing
    2. Calculate and return the mean of all pixels in the region
    
      """
    # 1. Extract the region from the image using array slicing
    r1, c1 = top_left
    r2, c2 = bottom_right
    
    # In NumPy slicing, the end index is exclusive, so we add 1 to include the bottom_right pixel.
    region = img[r1:r2+1, c1:c2+1]
    
    # 2. Calculate and return the mean of all pixels in the region
    # The .mean() method automatically sums all elements and divides by the count.
    return np.mean(region)


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)  # Call mean_by_direct_sum()

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part d)
# ==============================================================================
print("\n--- d) Computational Complexity Analysis ---")

'''
TODO:
1. Benchmark both methods by running them multiple times (e.g., 100 iterations)
2. Measure execution time for both methods using time.perf_counter()
3. Compare the execution times
4. Verify that both methods produce the same result
5. Print the results:
   - Method name
   - Average execution time
   - Performance improvement factor


'''

# Benchmark parameters
iterations = 100

print(f"\nBenchmarking with {iterations} iterations...\n")

# TODO: Implement benchmarking code here
# --- Benchmarking mean_using_integral ---
start_time_integral = time.perf_counter()
for _ in range(iterations):
    mean_integral_val = mean_using_integral(integral_img, top_left, bottom_right)
end_time_integral = time.perf_counter()
# time.perf_counter() returns time in seconds

time_integral = (end_time_integral - start_time_integral) / iterations
# we divide by iterations to get average time per call

# --- Benchmarking mean_by_direct_sum ---
start_time_direct = time.perf_counter()
for _ in range(iterations):
    mean_direct_val = mean_by_direct_sum(gray_img, top_left, bottom_right)
end_time_direct = time.perf_counter()
time_direct = (end_time_direct - start_time_direct) / iterations


# TODO: Display results
print(f"Mean value from integral image method: {mean_integral_val:.2f}")
print(f"Mean value from direct sum method:   {mean_direct_val:.2f}")



# TODO: Print theoretical complexity explanation
print("\n--- Theoretical Complexity (Big-O Notation) ---")
print("Method (b) - Using Integral Image:")
print("  - Time Complexity: O(1) (Constant Time)")
# it is 1 because the number of operations does not depend on the size of the region.
print("  - Explanation: Once the integral image is built, calculating the sum of any rectangular region requires only four array lookups and three arithmetic operations, regardless of the size of the region. This makes it incredibly efficient.")
print("\nMethod (c) - Direct Summation:")
print("  - Time Complexity: O(W * H) where W and H are the width and height of the region.")
print("  - Explanation: This method must iterate through every single pixel within the specified region to calculate the sum. The execution time, therefore, grows linearly with the number of pixels (the area) of the region.")

# 5. Print the benchmark results
print("\n--- Performance Results ---")
print(f"-Method (b): Integral Image\n-Average time: {time_integral * 1e6:.4f} microseconds")
print(f"-Method (c): Direct Summation\n-Average time: {time_direct * 1e6:.4f} microseconds")

if time_integral > 0:
    improvement_factor = time_direct / time_integral
    print(f"\nPerformance Improvement: The integral image method is approximately {improvement_factor:.2f} times faster.")




# --- Print theoretical complexity explanation ---
