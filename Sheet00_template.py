"""
Exercise 0 for MA-INF 2201 Computer Vision WS25/26
Introduction to OpenCV - Template
Python 3.12, OpenCV 4.11, NumPy 2.3.3
Image: bonn.jpeg
"""

import cv2 as cv
import numpy as np
import random
import time
import os

# ============================================================================
# Exercise 1: Read and Display Image (0.5 Points)
# ============================================================================
def exercise1():
    """
    Read and display the image bonn.jpeg.
    Print the image dimensions and data type.
    """
    print("Exercise 1: Read and Display Image")
    
    #I have used relative path to read the image else it won't work 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'bonn.jpeg')
    img = cv.imread("bonn.jpeg")
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # img.shape provides the hight , width and channels of the image
    print('Loaded image shape:', img.shape)
    print('Loaded image dtype :', img.dtype)    
    print("Exercise 1 completed!\n")
    return img


# ============================================================================
# Exercise 2: HSV Color Space (0.5 Points)
# ============================================================================
def exercise2(img):
    """
    Convert image to HSV color space and display all three channels separately.
    """
    print("Exercise 2: HSV Color Space")
    
    # TODO: Convert to HSV using cv2.cvtColor() with cv2.COLOR_BGR2HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # TODO: Split HSV into H, S, V channels using cv2.split()
    h, s, v = cv.split(hsv)
    
    # TODO: Display all three channels
    # Hint: You can concatenate them horizontally using cv2.hconcat()
    hsv_concat = cv.hconcat([h, s, v])
    cv.imshow("HSV Channels (H | S | V)", hsv_concat)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
   
    
    print("Exercise 2 completed!\n")
    return hsv


# ============================================================================
# Exercise 3: Brightness Adjustment with Loops (1 Point)
# ============================================================================
def exercise3(img):
    """
    Add 50 to all pixel values and clip to [0, 255] using nested for-loops.
    Display original and brightened images side by side.
    """
    print("Exercise 3: Brightness Adjustment with Loops")
    
    # TODO: Create a copy of the image
    result = img.copy()
    
    # TODO: Get image dimensions
    x,y = result.shape[0],result.shape[1]
    
    # TODO: Use nested for-loops to iterate through each pixel, add 50 to pixel value, and clip pixel value to [0, 255]
    for i in range(x):
        for j in range(y):
            for k in range(3): 
                result[i,j,k] = min(result[i,j,k] + 50, 255)
    # TODO: Display original and result side by side
    concated_image = cv.hconcat([img,result])
    cv.imshow("Original | Brightened", concated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Exercise 3 completed!\n")
    return result


# ============================================================================
# Exercise 4: Vectorized Brightness Adjustment (1 Points)
# ============================================================================
def exercise4(img):
    """
    Perform the same brightness adjustment using NumPy in one line.
    Compare execution time with loop-based approach.
    """
    print("Exercise 4: Vectorized Brightness Adjustment")
    
    # Time the loop-based approach (copy of exercise3 loop)
    # Make a copy so we don't modify the original
    loop_img = img.copy()
    start_time_loop = time.time()
    loop_result = exercise3(loop_img)
    end_time_loop = time.time()

    # Time the vectorized approach
    start_time_vec = time.time()
    # Use np.clip and ensure we operate in a type that won't overflow, then cast back to uint8
    vec_result = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    end_time_vec = time.time()

    # Print execution times
    print(f"Loop-based approach: {end_time_loop - start_time_loop:.4f} seconds")
    print(f"Vectorized approach: {end_time_vec - start_time_vec:.4f} seconds")

    # Display original, loop_result, and vec_result side by side
    try:
        combined = cv.hconcat([img,loop_result, vec_result])
        cv.imshow("Original | Loop Brightened | Vectorized Brightened", combined)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except cv.error:
        # In case the display fails (headless environment), just skip showing
        print("Note: cv.imshow failed (possible headless environment). Skipping display.")

    print("Exercise 4 completed!\n")
    return vec_result


# ============================================================================
# Exercise 5: Extract and Paste Patch (0.5 Points)
# ============================================================================
def exercise5(img):
    """
    Extract a 32×32 patch from top-left corner and paste at 3 random locations.
    """
    print("Exercise 5: Extract and Paste Patch")
    
    # TODO: Extract 32x32 patch from top-left corner (starting at 0,0)
    patch_size = 32
    # Ensure the image is large enough
    h, w = img.shape[0], img.shape[1]
    if h < patch_size or w < patch_size:
        print(f"Image too small for a {patch_size}x{patch_size} patch.")
        return img

    # Extract patch from top-left corner
    patch = img[0:patch_size, 0:patch_size].copy()

    # Create a copy to paste onto
    img_copy = img.copy()

    # Generate 3 random locations and paste the patch
    for i in range(3):
        # Choose top-left corner (y, x) such that patch fits
        max_y = h - patch_size
        max_x = w - patch_size
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        img_copy[y:y+patch_size, x:x+patch_size] = patch
    # Display the result
    try:
        cv.imshow("Patch Pasted", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except cv.error:
        print("Note: cv.imshow failed (possible headless environment). Skipping display.")

    print("Exercise 5 completed!\n")
    return img_copy


# ============================================================================
# Exercise 6: Binary Masking (0.5 Points)
# ============================================================================
def exercise6(img):
    """
    Create masked version showing only bright regions.
    Convert to grayscale, threshold at 128, use as mask.
    """
    print("Exercise 6: Binary Masking")
    
    # TODO: Convert to grayscale using cv2.cvtColor() with cv2.COLOR_BGR2GRAY
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    # TODO: Apply binary threshold at value 128
    # Use cv2.threshold() with cv2.THRESH_BINARY
    _, mask = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
    
    # TODO: Apply mask to original color image
    # Hint: Use cv2.bitwise_and() with the mask
    masked = cv.bitwise_and(img, img, mask=mask)
    
    # TODO: Display original, mask, and masked result
    try:
        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)  # Convert mask to BGR for concatenation
        combined = cv.hconcat([img,  masked])
        cv.imshow("Original | Mask | Masked Result", combined)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except cv.error:
        print("Note: cv.imshow failed (possible headless environment). Skipping display.")
    
    print("Exercise 6 completed!\n")


# ============================================================================
# Exercise 7: Border and Annotations (1 Points)
# ============================================================================
def exercise7(img):
    """
    Add 20-pixel border and draw 5 circles and 5 text labels at random positions.
    """
    print("Exercise 7: Border and Annotations")
    
    # TODO: Add 20-pixel border using cv2.copyMakeBorder()
    # Use cv2.BORDER_CONSTANT with a color of your choice
    border_size = 20
    border_color = (50, 200, 50)  # BGR
    bordered = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                 cv.BORDER_CONSTANT, value=border_color)

    # Get dimensions of bordered image
    h, w = bordered.shape[0], bordered.shape[1]

    # Draw 5 random circles
    for i in range(5):
        center_x = random.randint(border_size, w - border_size - 1)
        center_y = random.randint(border_size, h - border_size - 1)
        radius = random.randint(5, min(50, min(w, h)//10))
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        thickness = random.randint(1, 4)
        cv.circle(bordered, (center_x, center_y), radius, color, thickness)

    # Add 5 random text labels
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(5):
        text = f"P{i+1}"
        org_x = random.randint(border_size, w - border_size - 60)
        org_y = random.randint(border_size + 10, h - border_size - 10)
        font_scale = round(random.uniform(0.5, 1.2), 2)
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        thickness = random.randint(1, 2)
        cv.putText(bordered, text, (org_x, org_y), font, font_scale, color, thickness, cv.LINE_AA)

    # Display the result
    try:
        cv.imshow("Bordered & Annotated", bordered)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except cv.error:
        print("Note: cv.imshow failed (possible headless environment). Skipping display.")

    print("Exercise 7 completed!\n")
    return bordered
    
    


# ============================================================================
# Main function
# ============================================================================
def main():
    """
    Run all exercises.
    """
    print("=" * 60)
    print("Exercise 0: Introduction to OpenCV")
    print("=" * 60 + "\n")
    
    img = exercise1()
    if img is None:
        return
    exercise2(img)
    exercise3(img)
    exercise4(img)
    exercise5(img)
    exercise6(img)
    exercise7(img)
    
    print("=" * 60)
    print("All exercises completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
