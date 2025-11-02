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
    Compute gradient magnitude and direction using manual Sobel convolution.

    This implementation is based on the formulas shown in the image:
    - Gx and Gy are calculated by convolving the image with Sobel kernels.
    - Magnitude M(x,y) = sqrt(gx^2 + gy^2)
    - Phase/Angle α(x,y) = arctan(gy / gx)

    This function does not use the cv2 library.

    Args:
        img: A 2D numpy array representing the grayscale image.

    Returns:
        A tuple containing:
        - gradient_magnitude (np.ndarray): The gradient magnitude, normalized to 0-255.
        - gradient_angle (np.ndarray): The gradient direction in radians (-pi to +pi).
    """

    # Define the 3x3 Sobel kernels for approximating derivatives in x and y
    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])
                   
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    # Get the dimensions of the input image
    height, width = img.shape
    
    # Initialize arrays to store the gradients in x and y directions
    sobelx = np.zeros_like(img, dtype=np.float64)
    sobely = np.zeros_like(img, dtype=np.float64)

    # Manually convolve the image with the Sobel kernels
    # We iterate over each pixel, ignoring a 1-pixel border to handle the kernel size
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Select a 3x3 neighborhood of pixels
            neighborhood = img[i-1:i+2, j-1:j+2]
            
            # Compute the gradient in the x-direction (gx)
            gx = np.sum(neighborhood * Kx)
            sobelx[i, j] = gx
            
            # Compute the gradient in the y-direction (gy)
            gy = np.sum(neighborhood * Ky)
            sobely[i, j] = gy

    # Calculate the gradient magnitude using the formula from the image (L2 Norm)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the magnitude to a 0-255 scale for easier processing later.
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    # Calculate the gradient angle.
    # We use arctan2(gy, gx) as it correctly handles all quadrants and
    # avoids division-by-zero errors, which is a robust implementation
    # of the formula α = tan⁻¹(gy/gx).
    gradient_angle = np.arctan2(sobely, sobelx)

    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    """
    # Get the dimensions of the input images
    height, width = mag.shape
    
    # Initialize an output image of the same size with zeros
    suppressed_img = np.zeros_like(mag, dtype=mag.dtype)
    
    # Convert angles from radians to degrees for easier quantization.
    # We map all angles to the range [0, 180) because opposite directions
    # (e.g., 10° and 190°) lie on the same line.
    angles_deg = np.degrees(ang)
    # np.degrees returns angles in the range [-180, 180].
    angles_deg[angles_deg < 0] += 180
    # Here, angles are now in [0, 180).
    
    # Iterate over each pixel, ignoring the 1-pixel border to ensure
    # all neighbors are within bounds.
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            
            angle = angles_deg[i, j]
            current_mag = mag[i, j]

            # --- Quantize the angle to one of 4 directions ---
            
            # Direction 1: Horizontal edge (gradient is vertical)
            # Check neighbors above and below.
            if (67.5 <= angle < 112.5):
                neighbor1 = mag[i - 1, j]
                neighbor2 = mag[i + 1, j]
            
            # Direction 2: +45° diagonal edge (gradient is at 135°)
            # Check neighbors at top-right and bottom-left.
            elif (112.5 <= angle < 157.5):
                neighbor1 = mag[i - 1, j + 1]
                neighbor2 = mag[i + 1, j - 1]
            
            # Direction 3: -45° diagonal edge (gradient is at 45°)
            # Check neighbors at top-left and bottom-right.
            elif (22.5 <= angle < 67.5):
                neighbor1 = mag[i - 1, j - 1]
                neighbor2 = mag[i + 1, j + 1]
            
            # Direction 0: Vertical edge (gradient is horizontal)
            # Check neighbors to the left and right.
            # This covers angles from [0, 22.5) and [157.5, 180].
            else:
                neighbor1 = mag[i, j - 1]
                neighbor2 = mag[i, j + 1]

            # --- Perform the suppression check ---
            # If the current pixel's magnitude is greater than its neighbors
            # along the gradient direction, keep it. Otherwise, suppress it.
            if current_mag > neighbor1 and current_mag > neighbor2:
                suppressed_img[i, j] = current_mag
            else:
                suppressed_img[i, j] = 0
                
    return suppressed_img


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


    high_threshold = edge_map.max() * 0.09
    low_threshold = high_threshold * 0.05

    # Get the dimensions of the input image
    height, width = edge_map.shape

    # Create a new array to store the final edge map
    final_edge_map = np.zeros((height, width), dtype=np.uint8)

    # Define constants for weak and strong pixels for clarity
    WEAK_PIXEL = 75
    STRONG_PIXEL = 255

    # --- Step 1: Initial Classification ---
    # Identify strong pixels (above high threshold)
    strong_i, strong_j = np.where(edge_map >= high_threshold)
    final_edge_map[strong_i, strong_j] = STRONG_PIXEL

    # Identify weak pixels (between low and high thresholds)
    weak_i, weak_j = np.where((edge_map >= low_threshold) & (edge_map < high_threshold))
    final_edge_map[weak_i, weak_j] = WEAK_PIXEL

    # --- Step 2: Edge Tracking by Hysteresis ---
    # The image illustrates that a weak pixel (center) is promoted to a strong one
    # if it is connected to a strong pixel. We iterate through the image and check
    # the 8-pixel neighborhood of each weak pixel.
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if final_edge_map[i, j] == WEAK_PIXEL:
                # Check the 3x3 neighborhood for any strong pixels
                if np.any(final_edge_map[i-1:i+2, j-1:j+2] == STRONG_PIXEL):
                    final_edge_map[i, j] = STRONG_PIXEL
                else:
                    # If no strong pixel is found in the neighborhood, suppress this weak pixel
                    final_edge_map[i, j] = 0

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





SIGMA = 0.4
LOW_THRESHOLD_RATIO = 0.3
HIGH_THRESHOLD_RATIO = 0.9




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


# cv2.imshow('Gradient Magnitude', gradient_angle)
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
