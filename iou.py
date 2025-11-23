import numpy as np
import cv2
import os

def calculate_iou(prediction_path, target_path):
    """
    Calculates the Intersection over Union (IoU) between a predicted binary mask 
    and a ground truth target mask.
    """
    # 1. Load the Images
    # Load prediction as grayscale (0-255)
    pred_mask = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    
    # Load target unchanged first to inspect channels
    target_raw = cv2.imread(target_path)
    
    # Check if images loaded successfully
    if pred_mask is None:
        raise FileNotFoundError(f"Could not load prediction image: {prediction_path}")
    if target_raw is None:
        raise FileNotFoundError(f"Could not load target image: {target_path}")

    print(f"Loaded Prediction: {pred_mask.shape}, Unique values: {np.unique(pred_mask)}")
    print(f"Loaded Target (Raw): {target_raw.shape}, Unique values: {np.unique(target_raw)}")

    # 2. Preprocess Target Mask (Handle Colors/Classes)
    # If target is RGB/BGR (Height, Width, 3)
    if len(target_raw.shape) == 3:
        # Option A: If it's a label map where blue channel has info
        # Check if it looks like the blue mask provided (Blue > 0)
        b, g, r = cv2.split(target_raw)
        # Create a mask where ANY channel has data (assuming background is black)
        target_mask = cv2.max(b, cv2.max(g, r))
    else:
        target_mask = target_raw

    # 3. Ensure Dimensions Match
    if pred_mask.shape != target_mask.shape:
        print(f"Warning: Shape mismatch. Resizing prediction to match target.")
        print(f"Pred: {pred_mask.shape}, Target: {target_mask.shape}")
        # Resize prediction to match target using Nearest Neighbor (to keep discrete values)
        pred_mask = cv2.resize(pred_mask, (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 4. Binarize Both Masks
    # Convert to boolean (True/False) arrays
    # We assume any non-zero pixel is a building
    
    # Prediction: Threshold at 127 (usually 0 and 255)
    pred_binary = pred_mask > 127
    
    # Target: Threshold at 0 (any non-black pixel is a building)
    target_binary = target_mask > 0
    
    # Debugging: Check overlap
    print(f"Prediction Building Pixels: {np.sum(pred_binary)}")
    print(f"Target Building Pixels:     {np.sum(target_binary)}")
    
    # 5. Compute Intersection and Union
    # Intersection: Pixels where BOTH are True
    intersection = np.logical_and(pred_binary, target_binary)
    
    # Union: Pixels where EITHER is True
    union = np.logical_or(pred_binary, target_binary)
    
    # Sum the true values to get areas
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    print(f"Intersection Area: {intersection_area}")
    print(f"Union Area:        {union_area}")
    
    # 6. Calculate IoU
    if union_area == 0:
        return 0.0 # Avoid division by zero
        
    iou = intersection_area / union_area
    
    return iou

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Main Execution
try:
    # Define paths
    prediction_file = 'data/final_refine_mask.tif'
    target_file = 'data/img_mosaic_label.tif'
    
    # Check if files exist before running
    if not os.path.exists(prediction_file):
        print(f"Error: Prediction file not found at {prediction_file}")
    elif not os.path.exists(target_file):
        print(f"Error: Ground Truth file not found at {target_file}")
    else:
        # Calculate IoU
        score = calculate_iou(prediction_file, target_file)
        
        print("-" * 30)
        print(f"IoU Score Calculation")
        print("-" * 30)
        print(f"Prediction: {prediction_file}")
        print(f"Target:     {target_file}")
        print(f"IoU Score:  {score:.4f}")
        print(f"Percentage: {score * 100:.2f}%")
        print("-" * 30)

except Exception as e:
    print(f"An error occurred: {e}")