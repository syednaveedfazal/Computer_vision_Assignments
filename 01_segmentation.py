import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import os

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TODO your implementation



# Load the Image
file_path = 'data/img_mosaic.tif'
if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
    exit()

img = cv2.imread(file_path)


img_float = img.astype(np.float32) / 255.0
# This converts them to decimal numbers between 0 and 1




# OpenCV loads images as BGR

nir = img_float[:, :, 2]
# Extracts the Near-Infrared band (stored in the Red channel)
red = img_float[:, :, 1]
# Extracts the Red band (stored in the Green channel)
green = img_float[:, :, 0]
# Extracts the Green band (stored in the Blue channel)

# OpenCV loads images as BGR
# R=NIR, G=Red, B=Green 
# Satellite sensors capture different "bands" of light



epsilon = 1e-8
# A tiny number to prevent division by zero


ndvi = (nir - red) / (nir + red + epsilon)
# ndvi is a 2D NumPy array
# NDVI (Normalized Difference Vegetation Index) is a standard formula to find plants. 
# Plants reflect a lot of NIR light but absorb Red light
# Vegetation typically has high NDVI (> 0)
# Buildings/Roads/Shadows have low NDVI



veg_threshold = 0.1
# Threshold NDVI to remove vegetation

non_veg_mask = ndvi < veg_threshold
# non_veg_mask is a boolean mask, a 2D NumPy array of True/False


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
saturation = hsv[:, :, 1]
# Filter Roads based on Saturation (Gray color)
# Roads are gray, meaning they have low saturation.
# Convert to HSV to extract Saturation channel.


road_saturation_threshold = 68
# The cutoff for "greyness"
road_mask = saturation < road_saturation_threshold
# road_mask returns 2D array of True/False for road pixels
# If saturation is low (very grey), we assume it is a road




clean_mask = non_veg_mask & (~road_mask)
# We want pixels that are NOT plants (non_veg_mask).
# AND (&) that are NOT (~) roads

# 5. Clustering on Cleaned pixels
# We want to separate buildings from whatever is left (shadows, other artifacts).
# We can use K-Means on the RGB (actually NIR-Red-Green) values of the clean pixels.

pixels = img_float[clean_mask]
# list of only the valid pixels that haven't filtered out yet


if len(pixels) > 0:
# len(pixels) > 0 make sure there are some pixels to cluster


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2) # Increased iterations/precision
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER means we combine two flags
    # Stop based on Accuracy (EPS) OR based on Count (MAX_ITER)
    # 100 is max iterations
    # 0.2 is required accuracy
    
    
    K = 5 # Increased K to capture more variance
    # K = 5 means we want to cluster pixels into 5 groups

    # Apply KMeans
    # cv2.kmeans expects float32
    ret, label, center = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # label list telling us which group (0-4) every pixel belongs to
    # center is the "average color" of each of the 5 groups
    # ret is the compactness score (not used here)
    # 10: The algorithm runs 10 separate times with different starting points and picks the best result
    # cv2.KMEANS_RANDOM_CENTERS: Randomly choose initial cluster centers




    # Reconstruct the label image
    seg_map = np.full(ndvi.shape, K, dtype=np.uint8)
    # Create an empty image for label

    
    
    seg_map[clean_mask] = label.flatten()
    # Assign labels to clean pixels
    # label is a column vector, flatten it to 1D array for indexing

    
    label_colors = np.random.randint(0, 255, (K + 1, 3), dtype=np.uint8)
    label_colors[K] = [0, 0, 0] # Background is black
    cluster_vis = label_colors[seg_map]
    # Save cluster map for debugging
    # Map labels to colors
    # cv2.imwrite('cluster_map.png', cluster_vis)







    # Heuristic: Buildings are bright. Shadows are dark. Roads are grey.
    
    centers_intensity = np.mean(center, axis=1)
    # Calculate mean intensity of each cluster
    print("Cluster centers intensity:", centers_intensity)
    

    sorted_indices = np.argsort(centers_intensity)[::-1]
    # Sorts the groups from Brightest to Darkest
    

    # Select the top 2 brightest clusters as buildings (heuristic)
    # We can also check if the intensity is above a certain threshold
    building_clusters = []
    intensity_threshold = 0.06
    for idx in sorted_indices:
    # Loops through the groups
        if centers_intensity[idx] > intensity_threshold:
        # If a group is bright enough, it is added to building_clusters
            building_clusters.append(idx)
            print(f"Cluster {idx} (Intensity {centers_intensity[idx]:.2f}) selected as building.")
    
    if not building_clusters:
        print("No bright clusters found. Fallback to brightest.")
        building_clusters.append(sorted_indices[0])

    
    building_mask = np.isin(seg_map, building_clusters).astype(np.uint8) * 255
    # Create binary mask for buildings
    # Checks every pixel in the map. If its group ID is in our "Building List," make it White (255). 
    # Otherwise, make it Black (0)
    
    building_mask = building_mask & (clean_mask.astype(np.uint8) * 255)
    # Mask out areas that were not in clean_mask (background)
    # Double checks that we aren't accidentally including pixels that were originally roads or vegetation


    # Post-processing (Morphological operations) to clean up
    kernel_close = np.ones((4,4), np.uint8) 
    kernel_open = np.ones((1,1), np.uint8)

    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # Opening to remove small isolated noise and disconnect roads

    
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # Closing to fill holes within buildings



    # Area Filtering
    min_area = 2000
    max_area = 88000                                        
    # To remove tiny noise segments and overly large segments that are unlikely to be buildings


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(building_mask, connectivity=8)
    new_mask = np.zeros_like(building_mask)
    

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Gets the size (in pixels) of the current blob
        if min_area <= area <= max_area:
        # If it passed the size test, copy that segment onto the new_mask
            new_mask[labels == i] = 255
    
    building_mask = new_mask

    
    cv2.imwrite('data/final_mask.png', building_mask)
    # Save results
    # cv2.imwrite('ndvi_vis.png', (ndvi + 1) / 2 * 255) # Normalize NDVI to 0-255
    
    plt.figure(figsize=(15, 8))
 
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Final Building Mask
    plt.subplot(1, 2, 2)
    plt.imshow(building_mask, cmap='gray')
    plt.title('Final Building Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Results saved to 'segmentation_result.png', 'building_mask.png', and 'cluster_map.png'")

else:
    print("No non-vegetation pixels found. Check NDVI threshold or band mapping.")