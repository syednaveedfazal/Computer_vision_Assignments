# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.
    """
    # In OpenCV, if the kernel size is specified as (0, 0), it is automatically
    # calculated from the sigma value. This is the recommended practice.
    # The kernel is created based on the Gaussian function:
    # G(x,y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    # This kernel is then convolved with the image to produce the smoothed result.
    
    smoothed_img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    return smoothed_img


def compute_gradients(img):
    """
    Compute gradient magnitude and direction (Sobel-based).
    Return gradient_magnitude, gradient_angle.
    """


    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Calculate the vertical gradient (Gy)
    # dx=0, dy=1 specifies a first-order derivative in the y-direction.
    # The kernel used is equivalent to:
    # K_Gy = [[-1, -2, -1],
    #         [ 0,  0,  0],
    #         [ 1,  2,  1]]
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Use L1 norm (|Gx| + |Gy|) to match cv2.Canny's default behavior
    gradient_magnitude = np.abs(sobelx) + np.abs(sobely)
    
    # Normalize magnitude to a 0-255 scale for easier processing later,
    # converting it back to an 8-bit unsigned integer type.
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    # Calculate the Gradient Direction
    # We use arctan2(Gy, Gx) which handles all quadrants correctly and avoids
    # division by zero. The result is in radians, ranging from -pi to +pi.
    gradient_angle = np.arctan2(sobely, sobelx)

    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    """
    # Get the dimensions of the input images
    height, width = mag.shape
    suppressed_img = np.zeros_like(mag, dtype=np.float32)
    ang[ang < 0] += np.pi  # Map all angles to [0, pi]

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = ang[i, j]

            # Horizontal gradient (vertical edge)
            if (0 <= angle < np.pi / 4) or (3 * np.pi / 4 <= angle <= np.pi):
                t = np.abs(np.tan(angle))
                p1 = (1 - t) * mag[i, j + 1] + t * mag[i - 1, j + 1]
                p2 = (1 - t) * mag[i, j - 1] + t * mag[i + 1, j - 1]
            # Vertical gradient (horizontal edge)
            else:
                t = np.abs(1 / np.tan(angle))
                p1 = (1 - t) * mag[i - 1, j] + t * mag[i - 1, j + 1]
                p2 = (1 - t) * mag[i + 1, j] + t * mag[i + 1, j - 1]

            if mag[i, j] >= p1 and mag[i, j] >= p2:
                suppressed_img[i, j] = mag[i, j]
                
    return suppressed_img.astype(np.uint8)


def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.
    Return thresholded edge map.
    """
    
    # Define constants for pixel intensity values for clarity.
    # These will be the labels in our new map.
    strong_pixel_val = 255
    weak_pixel_val = 70  # An arbitrary intermediate value.

    height, width = nms.shape
    # nms is the input image after non-maximum suppression.
    
    # Create an image to store the result of the classification.
    classified_map = np.zeros((height, width), dtype=np.uint8)

    # Find the coordinates (indices) of pixels that fall into each category.
    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms < high) & (nms >= low))
    
    # 3. Set the corresponding pixel values (labels) in our classified map.
    classified_map[strong_i, strong_j] = strong_pixel_val
    classified_map[weak_i, weak_j] = weak_pixel_val
    
    return classified_map, weak_pixel_val, strong_pixel_val


def hysteresis(edge_map, weak, strong):
    """
    Perform edge tracking by hysteresis.
    Return final binary edge map.
    """


    height, width = edge_map.shape
    
    # The recursive function that traces and promotes connected weak pixels.
    def trace_and_promote(i, j):
        # Iterate over the 8-connected neighborhood of the pixel (i, j).
        for x in range(max(0, i-1), min(height, i+2)):
            for y in range(max(0, j-1), min(width, j+2)):
                # If a neighbor is a weak pixel...
                if edge_map[x, y] == weak:
                    # ...promote it to a strong pixel...
                    edge_map[x, y] = strong
                    # ...and continue the trace from this new strong pixel.
                    trace_and_promote(x, y)

    # The main loop iterates through every pixel of the classified map.
    for i in range(height):
        for j in range(width):
            # If a pixel is identified as a strong pixel, start the tracking.
            if edge_map[i, j] == strong:
                trace_and_promote(i, j)
    
    # After tracing, create the final binary map.
    # Any pixel that is not 'strong' is suppressed.
    final_edge_map = np.zeros_like(edge_map)
    final_edge_map[edge_map == strong] = 255  # Set final edge pixels to white.
    
    return final_edge_map


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.
    """
        # --- Compute Mean Absolute Difference (MAD) ---
    # Convert images to float to avoid data type issues during subtraction
    # The difference is normalized by 255 to get a value between 0 and 1
    mad = np.mean(np.abs(manual_edges.astype(float) - cv_edges.astype(float))) / 255.0

    # --- Compute F1-Score ---
    # Convert images to boolean arrays (True for edge, False for non-edge)
    my_bool = manual_edges > 0
    opencv_bool = cv_edges > 0

    # True Positives (TP): Correctly identified edge pixels
    tp = np.sum(np.logical_and(my_bool, opencv_bool))
    
    # False Positives (FP): Pixels you identified as an edge, but aren't
    fp = np.sum(np.logical_and(my_bool, np.logical_not(opencv_bool)))
    
    # False Negatives (FN): Actual edge pixels that you missed
    fn = np.sum(np.logical_and(np.logical_not(my_bool), opencv_bool))

    # Add a small epsilon to denominators to prevent division by zero
    epsilon = 1e-10

    # Calculate Precision and Recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score, mad


def plot_results(images, labels, figsize=(15, 10), cols=3):
    """
    A standalone function to plot a list of images with their corresponding labels in a grid.

    Args:
        images (list): A list of image arrays to be displayed.
        labels (list): A list of string titles for each image.
        figsize (tuple): The default size of the matplotlib figure.
        cols (int): The number of columns in the subplot grid.
    """
    # Ensure that the number of images matches the number of labels.
    if len(images) != len(labels):
        print("Error: The number of images and labels must be the same.")
        return

    # Automatically calculate the number of rows required to display all images.
    # np.ceil ensures that we have enough rows for all images, even if len(images) is not a multiple of cols.
    rows = int(np.ceil(len(images) / cols))

    plt.figure(figsize=figsize)

    # Iterate through the images and labels to create a subplot for each.
    for i, (image, label) in enumerate(zip(images, labels)):
        # The subplot index starts from 1.
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off') # Hide the axes for a cleaner look.

    # Adjust the layout to prevent titles and images from overlapping.
    plt.tight_layout()
    plt.show()





SIGMA = 0.3
LOW_THRESHOLD_RATIO = 0.1
HIGH_THRESHOLD_RATIO = 0.15




# ==========================================================

# TODO: 1. Load the grayscale image 'bonn.jpg'
img_path = 'data/bonn.jpg'
original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)



# TODO: 2. Smooth the image using your Gaussian function
smoothed_image = gaussian_smoothing(original_image, SIGMA)

# cv2.imshow('Smoothed Image', smoothed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# TODO: 3. Compute gradients (magnitude and direction)

gradient_magnitude, gradient_angle = compute_gradients(smoothed_image)


# cv2.imshow('Gradient Magnitude', gradient_magnitude)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# TODO: 4. Apply non-maximum suppression
suppressed_image = nonmax_suppression(gradient_magnitude, gradient_angle)

# cv2.imshow('Non-Maximum Suppression', suppressed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# TODO: 5. Apply double threshold (choose suitable low/high values)
high_threshold = original_image.max() * HIGH_THRESHOLD_RATIO
low_threshold = high_threshold * LOW_THRESHOLD_RATIO


classified_image, weak_val, strong_val = double_threshold(suppressed_image, low_threshold, high_threshold)  

# cv2.imshow('Double Thresholding', classified_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# TODO: 6. Perform hysteresis to obtain final edges
manual_edges = hysteresis(classified_image, weak_val, strong_val)



# TODO: 7. Compare your result with cv2.Canny using MAD and F1-score
opencv_canny_edges = cv2.Canny(original_image, low_threshold, high_threshold)

f1, mad = compute_metrics(manual_edges, opencv_canny_edges)

# Print the results
print("\n--- Comparison Metrics ---")
print(f"Mean Absolute Difference (MAD): {mad:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nNote: MAD represents the average per-pixel difference (0=identical, 1=completely different).")
print("F1-Score is the harmonic mean of precision and recall (closer to 1 is better).")


# TODO: 8. Display original image, your edges, and OpenCV edges

# cv2.imshow('Original Image', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('My Edges', manual_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('OpenCV Canny Edges', opencv_canny_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plot_results(
    images=[original_image, manual_edges, opencv_canny_edges],
    labels=['Original Image', 'My Canny Edges', 'OpenCV Canny Edges'],
    figsize=(12, 6),
    cols=2
)




