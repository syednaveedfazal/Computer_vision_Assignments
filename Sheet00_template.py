"""
Exercise 0 for MA-INF 2201 Computer Vision WS25/26
Introduction to OpenCV - Template
Python 3.12, OpenCV 4.11, NumPy 2.3.3
Image: bonn.jpeg
"""

import cv2
import numpy as np
import random
import time

# ============================================================================
# Exercise 1: Read and Display Image (0.5 Points)
# ============================================================================
def exercise1():
    """
    Read and display the image bonn.jpeg.
    Print the image dimensions and data type.
    """
    print("Exercise 1: Read and Display Image")
    
    # TODO: Read the image 'bonn.jpeg' using cv2.imread()
    img = cv2.imread("bonn.jpeg")
    
    # TODO: Check if image was loaded successfully
    if img is None:
        print("Could not load the image")
        return None
    # TODO: Display the image using cv2.imshow()
    cv2.imshow("Bonn", img)

    # TODO: Wait for a key press using cv2.waitKey(0)
    cv2.waitKey(0)

    # TODO: Close all windows using cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    # TODO: Print image dimensions (height, width, channels)
    height, width, channels = img.shape
    print(f"Image Dimensions: Height={height}, Width={width}, Channels={channels}")

    # TODO: Print image data type
    print(f"Image Data Type: {img.dtype}")
    
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # TODO: Split HSV into H, S, V channels using cv2.split()
    # H: Hue, S: Saturation, V: Value
    h, s, v = cv2.split(hsv)
    
    # TODO: Display all three channels
    cv2.imshow("Hue Channel", h)
    cv2.imshow("Saturation Channel", s)
    cv2.imshow("Value Channel", v)
    # Hint: You can concatenate them horizontally using cv2.hconcat()

    hsv_channels = cv2.hconcat([h, s, v])
    cv2.imshow("Hue, Saturation, and Value Channels", hsv_channels)

    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    height, width, channels = img.shape
    
    # TODO: Use nested for-loops to iterate through each pixel, add 50 to pixel value, and clip pixel value to [0, 255]
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                # Add 50 to the pixel value and ensure it's an integer
                brightened_value = int(result[i, j, c]) + 50
                
                # Clip the value to stay within the [0, 255] range
                if brightened_value > 255:
                    result[i, j, c] = 255
                else:
                    result[i, j, c] = brightened_value


    # TODO: Display original and result side by side
    # np.hstack places two images next to each other.
    combined_image = np.hstack((img, result))
    cv2.imshow('Original vs. Brightened', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
    
    # TODO: Time the loop-based approach (from exercise 3)
    start_time_loop = time.time()
    # ... (implement or copy loop code)
    result_loop = img.copy()
    height, width, channels = img.shape
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                # Add 50 and ensure the value is an integer
                brightened_value = int(result_loop[i, j, c]) + 50
                # Clip the value to stay within the [0, 255] range
                if brightened_value > 255:
                    result_loop[i, j, c] = 255
                else:
                    result_loop[i, j, c] = brightened_value

    end_time_loop = time.time()
    
    # TODO: Time the vectorized approach
    start_time_vec = time.time()

    # TODO: Add 50 and clip in one line using np.clip()
    # np.clip function "clamps" all the values in the array to be within a specified range
    # img.astype(np.int16) was used to prevent overflow during addition
    # astype(np.uint8) was used to convert back to original data type so that image can be displayed correctly
    result = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)

    end_time_vec = time.time()
    
    # TODO: Print execution times
    # .4f formats the float to 4 decimal places
    print(f"Loop-based approach: {end_time_loop - start_time_loop:.4f} seconds")
    print(f"Vectorized approach: {end_time_vec - start_time_vec:.4f} seconds")
    
    # TODO: Display the result
    cv2.imshow('Vectorized Brightened Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 4 completed!\n")
    return result


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
    # img[0:patch_size, 0:patch_size] means rows 0 to 31 and columns 0 to 31
    patch = img[0:patch_size, 0:patch_size]

    
    # TODO: Create a copy of the image
    img_copy = img.copy()
    
    # TODO: Get image dimensions
    # _ means we ignore the number of channels
    height, width, _ = img.shape

    
    
    # TODO: Generate 3 random locations and paste the patch
    # Use random.randint() and ensure patch fits within boundaries
    for i in range(3):
        # Determine the valid range for the top-left corner of the patch
        max_y = height - patch_size
        max_x = width - patch_size
        
        # Generate random coordinates for the top-left corner of the paste location
        rand_y = random.randint(0, max_y)
        rand_x = random.randint(0, max_x)
        
        print(f"Pasting patch {i+1} at (y,x): ({rand_y}, {rand_x})")
        
        # Paste the patch onto the image copy
        # rand_y : rand_y + patch_size means rows from rand_y to rand_y + 31
        # rand_x : rand_x + patch_size means columns from rand_x to rand_x + 31
        # This assigns the patch to the specified region in img_copy
        img_copy[rand_y : rand_y + patch_size, rand_x : rand_x + patch_size] = patch
    
    # TODO: Display the result
    cv2.imshow('Image with Patches', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # TODO: Apply binary threshold at value 128
    # Use cv2.threshold() with cv2.THRESH_BINARY
    # cv2.THRESH_BINARY means pixels above the threshold are set to max value (255), below to 0
    # _ is used to ignore the first return value (threshold used)
    # mask: pixels above 128 are set to 255, below to 0
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # TODO: Apply mask to original color image
    # Hint: Use cv2.bitwise_and() with the mask
    # where mask is 255, original pixel is kept; where mask is 0, pixel becomes black
    masked = cv2.bitwise_and(img, img, mask=mask)
    
    # TODO: Display original, mask, and masked result
    # To display them side-by-side, they must have the same number of channels.
    # We convert the single-channel mask to a 3-channel BGR image.
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Concatenate the images horizontally for a combined view
    combined_display = np.hstack((img, mask_bgr, masked))
    

    cv2.imshow('Original, Mask, Masked Result', combined_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Exercise 6 completed!\n")

    return masked, mask


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

    border_color = [255, 0, 0]  # Blue in BGR
    bordered_image = cv2.copyMakeBorder(
        img, 
        top=border_size, 
        bottom=border_size, 
        left=border_size, 
        right=border_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=border_color
    )

    
    # TODO: Get dimensions of bordered image
    height, width, _ = bordered_image.shape
    
    # TODO: Draw 5 random circles
    # Use random.randint() and cv2.circle(img, center, radius, color, thickness)
    for i in range(5):
        # Defining random properties for the circle
        radius = random.randint(10, 50)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = 2
        
        # Generate a center point ensuring the circle is fully within the image
        # The center's x must be between radius and width-radius
        # The center's y must be between radius and height-radius
        center_x = random.randint(radius, width - radius)
        center_y = random.randint(radius, height - radius)
        center = (center_x, center_y)

        # Draw the circle on the bordered image
        cv2.circle(bordered_image, center, radius, color, thickness)
    
    # TODO: Add 5 random text labels
    # Use random.randint() and cv2.putText(img, text, org, font, fontScale, color, thickness)
    for i in range(5):
        # Define random properties for the text
        text = f"Hello {i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = 2

        # Generate the origin (bottom-left corner of the text)
        # Ensure the text doesn't run off the screen
        org_x = random.randint(10, width - 200) # Leave space for text width
        org_y = random.randint(30, height - 30) # Leave space for text height
        org = (org_x, org_y)
        
        # Draw the text on the bordered image (in-place)
        cv2.putText(bordered_image, text, org, font, font_scale, color, thickness)

    
    # TODO: Display the result
    cv2.imshow('Image with Border and Annotations', bordered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    print("Exercise 7 completed!\n")
    return bordered_image


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
    
    # Uncomment the exercises you want to run:
    img = exercise1()
    # if img is None:
    #     return
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
