import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

# Load image and convert to grayscale
original_img_color = cv2.imread('bonn.jpg')  # Load 'bonn.jpg'
gray_img = cv2.cvtColor(original_img_color,cv2.COLOR_BGR2GRAY)          # Convert to grayscale

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part a)
# ==============================================================================
print("\n--- a) Calculating Integral Image ---")


def calculate_integral_image(img):
    h , w = img.shape
    # i have changed the dtype to int56 so that it can hold large sum
    integral_img = np.zeros((h + 1, w + 1), dtype=np.int64)
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            # this formula is dervied with region sum (A + B) + (A + C) - A ...this gives us A+B+C which is what we want 
            # sum of all pixels above and left to (y,x)
            integral_img[y, x] = (
                img[y - 1, x - 1] +
                integral_img[y - 1, x] +
                integral_img[y, x - 1] -
                integral_img[y - 1, x - 1]
            )
    return integral_img


# Calculate integral image
integral_img = calculate_integral_image(gray_img)  # Call calculate_integral_image()

print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part b)
# ==============================================================================
print("\n--- b) Computing Mean Using Integral Image ---")


def mean_using_integral(integral, top_left, bottom_right):
   
    
    y2,x2 = bottom_right
    y1,x1 = top_left
    # Sum(D) = I(r2, c2) - I(r1-1, c2) - I(r2, c1-1) + I(r1-1, c1-1) 
    # where I is integral image and D is the region 
    region_sum = (
        integral_img[y2, x2]
        - integral_img[y1 - 1, x2]
        - integral_img[y2, x1 - 1]
        + integral_img[y1 - 1, x1 - 1]
    )

    # Compute number of pixels in the region
    num_pixels = (y2 - y1 + 1) * (x2 - x1 + 1)
    
    mean_value = region_sum / num_pixels
    
    return mean_value
    


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

    y1, x1 = top_left
    y2, x2 = bottom_right

    region = img[y1:y2+1, x1:x2+1]  # +1 because Python slicing is exclusive on the end
    # i used directly np.mean to calculate mean of the region
    mean_val = np.mean(region)

    return mean_val


mean_direct = mean_by_direct_sum(gray_img,top_left,bottom_right)  # Call mean_by_direct_sum()

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part d)
# ==============================================================================
print("\n--- d) Computational Complexity Analysis ---")

iterations = 100
start_time_integral = time.perf_counter()
for _ in range(iterations):
    # We don't need to store the result, just run the function
    _ = mean_using_integral(integral_img, top_left, bottom_right)
end_time_integral = time.perf_counter()

# Calculate total and average execution time
total_time_integral = end_time_integral - start_time_integral
avg_time_integral = total_time_integral / iterations

#  Benchmark the Direct Summation method
start_time_direct = time.perf_counter()
for _ in range(iterations):
    _ = mean_by_direct_sum(gray_img, top_left, bottom_right)
end_time_direct = time.perf_counter()

total_time_direct = end_time_direct - start_time_direct
avg_time_direct = total_time_direct / iterations

# i'm using np.isclose to check for numerical closeness
results_are_same = np.isclose(mean_integral, mean_direct)

print(f"--- Benchmark Results ({iterations} iterations) ---")
print(f"\nVerification:")
print(f"  > Results from both methods are the same: {results_are_same}")

print("\nTiming Analysis:")
# Method 1: Integral Image
print("  Method 1: Integral Image")
# I am Multipling  by 1,000,000 to show the time in microseconds (μs) for better readability as the times are very small.
print(f"    - Average execution time: {avg_time_integral * 1e6:.4f} microseconds")

# Method 2: Direct Summation
print("\n  Method 2: Direct Summation")
print(f"    - Average execution time: {avg_time_direct * 1e6:.4f} microseconds")

# Performance Improvement Factor
if avg_time_integral > 0:
    performance_factor = avg_time_direct / avg_time_integral
    print("\nPerformance Comparison:")
    print(f"  > The Integral Image method was {performance_factor:.2f} times faster.")
else:
    print("\nPerformance Comparison: Could not calculate speedup (integral method was too fast).")