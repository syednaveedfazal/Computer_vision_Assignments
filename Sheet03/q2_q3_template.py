

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
    Returns:
        list of detected circles as (x, y, r, v), accumulator as [r_idx, y_c, x_c]
    """
    h, w = edges.shape
    
    # 1. Setup accumulator and parameter ranges
    radii = np.arange(min_radius, max_radius, r_ssz)
    n_radii = len(radii)
    thetas = np.deg2rad(np.arange(0, 360, theta_ssz))
    
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # Accumulator: dimensions are (radius, y_center, x_center)
    accumulator = np.zeros((n_radii, h, w), dtype=np.uint64)
    
    # Get coordinates of all edge points
    y_idxs, x_idxs = np.nonzero(edges)
    
    # 2. Voting process
    print("Voting in Hough space...")
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        # For each edge point, iterate through all possible radii
        for r_idx, r in enumerate(radii):
            # Calculate potential centers (a, b) for a circle of radius r passing through (x, y)
            # a = x - r*cos(theta), b = y - r*sin(theta)
            b_coords = y - r * sin_thetas
            a_coords = x - r * cos_thetas
            
            # Round to integer coordinates for accumulator bins
            b_coords = b_coords.astype(np.int32)
            a_coords = a_coords.astype(np.int32)
            
            # Filter out centers that are outside the image boundaries
            valid_indices = (a_coords >= 0) & (a_coords < w) & (b_coords >= 0) & (b_coords < h)
            
            # Increment accumulator for valid center locations
            # np.add.at performs indexed addition, which is crucial here
            np.add.at(accumulator, (r_idx, b_coords[valid_indices], a_coords[valid_indices]), 1)
            
    print("Voting complete.")
    
    # 3. Peak detection and filtering (Non-Maximal Suppression)
    print("Finding and filtering peaks...")
    detected_circles = []
    
    # Find all bins in the accumulator with votes above the threshold
    potential_peaks_indices = np.argwhere(accumulator > threshold)
    if potential_peaks_indices.shape[0] == 0:
        print("No circles found above the threshold.")
        return [], accumulator
        
    # Get the vote counts for these potential peaks
    peak_values = accumulator[potential_peaks_indices[:, 0], potential_peaks_indices[:, 1], potential_peaks_indices[:, 2]]
    
    # Combine indices and values into a single array: [r_idx, y, x, value]
    candidates = np.hstack((potential_peaks_indices, peak_values[:, np.newaxis]))
    
    # Sort candidates by vote count in descending order
    candidates = candidates[candidates[:, 3].argsort()[::-1]]
    
    # Filter out peaks that are too close to a better peak
    while candidates.shape[0] > 0:
        # Get the candidate with the highest vote
        r_idx, y, x, v = candidates[0].astype(int)
        r = radii[r_idx]
        
        # Add this circle to our final list
        detected_circles.append((x, y, int(r), int(v)))
        
        # Remove the current candidate from the list
        candidates = candidates[1:]
        
        # If there are candidates left, remove those that are too close
        if candidates.shape[0] > 0:
            # Calculate Euclidean distance between the current circle's center and all others
            distances = np.sqrt((candidates[:, 2] - x)**2 + (candidates[:, 1] - y)**2)
            # Keep only those candidates that are further than min_dist
            candidates = candidates[distances > min_dist]

    print(f"Found {len(detected_circles)} circles.")
    return detected_circles, accumulator


def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift (radius of the search window)
        threshold: Minimum accumulator value to consider as a starting point
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape
    
    if threshold is None:
        threshold = 0.3 * np.max(accumulator)
        
    print(f"Applying Mean Shift with bandwidth={bandwidth}, threshold={threshold:.2f}")

    # Get coordinates of all points in the accumulator above the threshold
    candidate_coords = np.argwhere(accumulator > threshold)
    
    if candidate_coords.shape[0] == 0:
        print("No points above threshold for Mean Shift.")
        return []

    points = [p for p in candidate_coords]
    peaks = []
    
    while len(points) > 0:
        # 1. Select a starting point and remove it from the list
        current_center = points.pop(0).astype(np.float32)
        
        # 2. Iteratively shift the center towards the weighted mean of points in its neighborhood
        for _ in range(30): # Max iterations to ensure convergence
            
            # Find all points from the original list within the bandwidth
            neighbors_coords = [p for p in candidate_coords if np.linalg.norm(p - current_center) < bandwidth]
            
            if not neighbors_coords:
                break
            
            # Calculate the weighted mean of the neighbors
            # The coordinate is the feature, the accumulator value is the weight
            neighbors_coords = np.array(neighbors_coords)
            neighbor_values = accumulator[neighbors_coords[:,0], neighbors_coords[:,1], neighbors_coords[:,2]]
            
            weighted_sum = np.sum(neighbors_coords * neighbor_values[:, np.newaxis], axis=0)
            sum_of_weights = np.sum(neighbor_values)
            
            if sum_of_weights == 0:
                break # Avoid division by zero
                 
            new_center = weighted_sum / sum_of_weights

            # Check for convergence
            if np.linalg.norm(new_center - current_center) < 1e-3:
                break
            
            current_center = new_center

        # 3. After convergence, check if this peak is a new one
        is_new_peak = True
        for peak in peaks:
            dist_to_existing_peak = np.linalg.norm(np.array(peak[:3]) - current_center)
            if dist_to_existing_peak < bandwidth:
                is_new_peak = False
                break
        
        if is_new_peak:
            r_idx, y, x = np.round(current_center).astype(int)
            if 0 <= r_idx < n_r and 0 <= y < h and 0 <= x < w:
                value = accumulator[r_idx, y, x]
                # Format as (x, y, r_idx, value)
                peaks.append((x, y, r_idx, value))

        # 4. Remove points that are now considered part of this converged cluster
        indices_to_remove = [i for i, p in enumerate(points) if np.linalg.norm(p - current_center) < bandwidth]
        for i in sorted(indices_to_remove, reverse=True):
            del points[i]
            
    print(f"Found {len(peaks)} peaks using Mean Shift.")
    return peaks


def main():
    
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        # As a fallback, create a dummy image for demonstration
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(img, "coins.jpg not found", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        img = cv2.imread(img_path)
        
    # Load image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a median blur to reduce noise, which helps Canny
    gray = cv2.medianBlur(gray, 5)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles using myHoughCircles...")
    min_radius = 20
    max_radius = 60
    threshold = 85
    min_dist = 50
    r_ssz = 1
    theta_ssz = 1
    
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    output_img_hough = img.copy()
    for x, y, r, v in detected_circles:
        # Draw the outer circle
        cv2.circle(output_img_hough, (x, y), r, (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(output_img_hough, (x, y), 2, (0, 0, 255), 3)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(output_img_hough, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f'Detected Circles: {len(detected_circles)}')
    ax[1].axis('off')
    plt.show()
    
    # Visualize accumulator slices
    print("\nVisualizing accumulator slices...")
    radii_range = np.arange(min_radius, max_radius, r_ssz)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    slice_indices = np.linspace(0, len(radii_range) - 1, 4, dtype=int)
    for i, r_idx in enumerate(slice_indices):
        r = radii_range[r_idx]
        slice_img = accumulator[r_idx, :, :]
        # Normalize for better visualization
        if np.max(slice_img) > 0:
            slice_img = (slice_img / np.max(slice_img) * 255).astype(np.uint8)
        
        axes[i].imshow(slice_img, cmap='hot')
        axes[i].set_title(f'Accumulator Slice for r = {r} px')
        axes[i].axis('off')
    plt.suptitle('Accumulator Slices for Different Radii')
    plt.show()
    
    # Visualize peak radius slice
    print("\nVisualizing accumulator slice at peak radius...")
    sum_per_radius = np.sum(accumulator, axis=(1, 2))
    peak_r_idx = np.argmax(sum_per_radius)
    peak_radius = radii_range[peak_r_idx]

    peak_slice = accumulator[peak_r_idx, :, :]
    if np.max(peak_slice) > 0:
        peak_slice_vis = (peak_slice / np.max(peak_slice) * 255).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    plt.imshow(peak_slice_vis, cmap='hot')
    plt.title(f'Accumulator Slice at Radius with Max Votes (r = {peak_radius} px)')
    plt.colorbar(label='Vote Intensity')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Parameter Analysis (Task 2):")
    print("  - Canny Thresholds: Affect the quality of the input edges. If too high, weak circles are missed. If too low, noise creates false votes.")
    print("  - Radius Range (min/max): Crucial for performance and accuracy. Must be set according to the objects of interest in the image.")
    print("  - Accumulator Threshold: This is the most sensitive parameter. A higher value leads to fewer, more certain detections (higher precision, lower recall). A lower value detects more circles but also more false positives.")
    print("  - Min Distance: Essential for separating distinct circles and preventing multiple detections for the same object. Should be based on object size.")
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    # =============================================================
    print("\n" + "=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    # Use the accumulator from the previous step
    bandwidth = 15.0  # Defines the radius for clustering in the 3D (r, y, x) space
    ms_threshold = 80 # A lower threshold than the manual one to provide more candidates to the algorithm
    peaks = myMeanShift(accumulator, bandwidth=bandwidth, threshold=ms_threshold)
    
    # Visualize corresponding circles on original image    
    output_img_ms = img.copy()
    for x, y, r_idx, v in peaks:
        if r_idx < len(radii_range):
            r = radii_range[r_idx]
            cv2.circle(output_img_ms, (x, y), int(r), (0, 255, 0), 2)
            cv2.circle(output_img_ms, (x, y), 2, (0, 0, 255), 3)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(output_img_ms, cv2.COLOR_BGR2RGB))
    plt.title(f'Circles Detected via Hough + Mean Shift: {len(peaks)}')
    plt.axis('off')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis (Task 3):")
    print("  - The `bandwidth` in Mean Shift is critical. It defines the size of the region to search for the mean.")
    print("  - If bandwidth is too small, a single circle might be detected as multiple peaks.")
    print("  - If bandwidth is too large, nearby distinct circles might be merged into a single detection.")
    print("  - Mean Shift automates peak detection, removing the need for a manually tuned `min_dist` and `threshold` for non-maximal suppression, replacing them with its own parameters (`bandwidth`, `threshold`).")
    print("=" * 70)
    print("Task 3 complete!")
    

if __name__ == "__main__":
    # Ensure the data directory exists for the script to run
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory. Please place 'coins.jpg' inside it.")
    main()