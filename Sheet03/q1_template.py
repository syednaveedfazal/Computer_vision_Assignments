# """
# Task 1: Distance Transform using Chamfer 5-7-11
# Template for MA-INF 2201 Computer Vision WS25/26
# Exercise 03
# """

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os



# def chamfer_distance_transform_5_7_11(binary_image):
#     """
#     Compute Chamfer distance transform using 5-7-11 mask.
    
#     Based on Borgefors "Distance transformations in digital images" (1986).
    
#     Chamfer 5-7-11:
#     - Horizontal/vertical neighbors: weight = 5
#     - Diagonal neighbors: weight = 7
#     - Knight's move neighbors: weight = 11
    
#     Args:
#         binary_image: Binary image where features are 0, background is non-zero
    
#     Returns:
#         Distance transform image
#     """
#     h, w = binary_image.shape
#     dt = np.full((h, w), np.inf, dtype=np.float32)
    
#     # Initialize: 0 if feature pixel, infinity otherwise
#     dt[binary_image == 0] = 0
    
#     # Define forward and backward masks with relative offsets and weights
#     # Forward mask
#     forward_mask = [
#         (-2, -1, 11), (-1, -2, 11), (-1, -1, 7), (0, -1, 5),
#         (1, -2, 11), (2, -1, 11), (1, -1, 7), (-1, 0, 5)
#     ]
    
#     # Backward mask
#     backward_mask = [
#         (1, 0, 5), (-1, 1, 7), (-2, 1, 11), (-1, 2, 11),
#         (0, 1, 5), (1, 1, 7), (2, 1, 11), (1, 2, 11)
#     ]
    
#     # Forward pass
#     for y in range(h):
#         for x in range(w):
#             for dy, dx, weight in forward_mask:
#                 ny, nx = y + dy, x + dx
#                 if 0 <= ny < h and 0 <= nx < w:
#                     if dt[ny, nx] + weight < dt[y, x]:
#                         dt[y, x] = dt[ny, nx] + weight
    
#     # Backward pass
#     for y in range(h - 1, -1, -1):
#         for x in range(w - 1, -1, -1):
#             for dy, dx, weight in backward_mask:
#                 ny, nx = y + dy, x + dx
#                 if 0 <= ny < h and 0 <= nx < w:
#                     if dt[ny, nx] + weight < dt[y, x]:
#                         dt[y, x] = dt[ny, nx] + weight
    
#     return dt


# def main():    
    
#     print("=" * 70)
#     print("Task 1: Distance Transform using Chamfer 5-7-11")
#     print("=" * 70)
    
#     img_path = 'data/bonn.jpg'
#     # img_path = 'data/circle.png'      # play with different images
#     # img_path = 'data/square.png'      
#     # img_path = 'data/triangle.png'    
    
#     if not os.path.exists(img_path):
#         print(f"Error: {img_path} not found!")
#         return
    
#     # Load image and convert to grayscale
#     # TODO
#     original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    
#     # Apply Canny edge detection
#     # TODO
#     # Compute distance transform with the function chamfer_distance_transform_5_7_11
#     # TODO

#     # Compute distance transform using cv2.distanceTransform
#     # TODO
#     edges = cv2.Canny(original_img, 100, 200)

#     chamfer_dt = chamfer_distance_transform_5_7_11(edges)

#     # Compare with OpenCV’s distance transform
#     # cv2.distanceTransform requires a binary image where the features to be measured are non-zero
#     opencv_dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

#     # Normalize the distance transforms for visualization
#     chamfer_dt_normalized = cv2.normalize(chamfer_dt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#     opencv_dt_normalized = cv2.normalize(opencv_dt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


#     # Visualize the results
#     plt.figure(figsize=(15, 10))

#     plt.subplot(2, 2, 1)
#     plt.imshow(original_img, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(2, 2, 2)
#     plt.imshow(edges, cmap='gray')
#     plt.title('Canny Edge Image')
#     plt.axis('off')

#     plt.subplot(2, 2, 3)
#     plt.imshow(chamfer_dt_normalized, cmap='viridis')
#     plt.title('Chamfer 5-7-11 Distance Transform')
#     plt.axis('off')

#     plt.subplot(2, 2, 4)
#     plt.imshow(opencv_dt_normalized, cmap='viridis')
#     plt.title("OpenCV's Distance Transform (L2)")
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()
#     # Visualize results
    
#     # TODO
#     # 1. Original image
#     # 2. Edge image
#     # 3. Distance transform
#     # 4. Distance transform using OpenCV
        
#     print("\n" + "=" * 70)
#     print("Task 1 complete!")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
    
    Based on Borgefors "Distance transformations in digital images" (1986).
    
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are non-zero, background is 0
    
    Returns:
        Distance transform image
    """
    h, w = binary_image.shape
    dt = np.full((h, w), np.inf, dtype=np.float32)
    
    # BUG FIX 1: Initialize feature pixels (edges) to 0, not background.
    # Canny edges are > 0.
    dt[binary_image > 0] = 0
    
    # BUG FIX 2: Correctly define forward and backward masks.
    # Forward mask looks at neighbors "up" and "left".
    forward_mask = [
        (-1,  0, 5),  # Up
        ( 0, -1, 5),  # Left
        (-1, -1, 7),  # Up-Left
        (-1,  1, 7),  # Up-Right (Note: checks a pixel to the right, but in the previous row)
        (-2, -1, 11), # Knight's move
        (-1, -2, 11), # Knight's move
        ( 1, -2, 11), # Knight's move
        (-2,  1, 11)  # Knight's move
    ]
    
    # Backward mask looks at neighbors "down" and "right".
    backward_mask = [
        ( 1,  0, 5),  # Down
        ( 0,  1, 5),  # Right
        ( 1,  1, 7),  # Down-Right
        ( 1, -1, 7),  # Down-Left
        ( 2,  1, 11), # Knight's move
        ( 1,  2, 11), # Knight's move
        (-1,  2, 11), # Knight's move
        ( 2, -1, 11)  # Knight's move
    ]
    
    # Forward pass: top-to-bottom, left-to-right
    for y in range(h):
        for x in range(w):
            for dy, dx, weight in forward_mask:
                # Check neighbor's coordinates
                ny, nx = y + dy, x + dx
                # Ensure neighbor is within image bounds
                if 0 <= ny < h and 0 <= nx < w:
                    # If a path through the neighbor is shorter, update the distance
                    if dt[ny, nx] + weight < dt[y, x]:
                        dt[y, x] = dt[ny, nx] + weight
    
    # Backward pass: bottom-to-top, right-to-left
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            for dy, dx, weight in backward_mask:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if dt[ny, nx] + weight < dt[y, x]:
                        dt[y, x] = dt[ny, nx] + weight
    
    return dt

# The rest of your main function is correct and can remain the same.
# Just make sure to use the corrected function above.
def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    img_path = 'data/bonn.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        # Create a dummy image if file is not found
        original_img = np.zeros((400, 600), dtype=np.uint8)
        cv2.putText(original_img, "bonn.jpg not found", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    else:
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    edges = cv2.Canny(original_img, 100, 200)

    # Use the corrected function
    chamfer_dt = chamfer_distance_transform_5_7_11(edges)

    opencv_dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    chamfer_dt_normalized = cv2.normalize(chamfer_dt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    opencv_dt_normalized = cv2.normalize(opencv_dt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(chamfer_dt_normalized, cmap='viridis')
    plt.title('Corrected Chamfer 5-7-11 Distance Transform')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(opencv_dt_normalized, cmap='viridis')
    plt.title("OpenCV's Distance Transform (L2)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
        
    print("\n" + "=" * 70)
    print("Task 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory. Please place 'bonn.jpg' inside it.")
    main()