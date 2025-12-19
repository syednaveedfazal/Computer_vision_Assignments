import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''

class MOG():
    def __init__(self,height=None, width=None, number_of_gaussians=None, background_thresh=None, lr=None):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.dist_thresh = 20
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height,self.width, self.number_of_gaussians,3)) ## assuming using color frames
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians)) ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array([[122, 122, 122]]*self.number_of_gaussians) ##assuming a [0,255] color channel
                self.sigmaSQs[i,j]=[36.0] * self.number_of_gaussians
                self.omegas[i,j]=[1.0 / self.number_of_gaussians] * self.number_of_gaussians
                
    def updateParam(self, img, BG_pivot): #finish this function
        # Convert inputs to float for calculation
        frame = img.astype(np.float32)
        height, width, _ = frame.shape
        K = self.number_of_gaussians
        alpha = self.lr

        # 1. Calculate Squared Euclidean Distances
        # Broadcast frame (H,W,1,3) against means (H,W,K,3)
        diff = frame[:, :, np.newaxis, :] - self.mus
        dist_sq = np.sum(diff**2, axis=3) # Result shape: (H, W, K)

        # 2. Check for Matches
        # Condition: distance < 2.5 * sigma => dist^2 < 6.25 * sigma^2
        matches = dist_sq < (6.25 * self.sigmaSQs)
        
        # Find the index of the *first* matching Gaussian for each pixel
        # argmax returns the first True index. If all False, it returns 0 (handled by has_match)
        match_idx = np.argmax(matches, axis=2)
        has_match = np.any(matches, axis=2)

        # 3. Update Weights (Omegas)
        # Decay all weights: w = (1-alpha)*w
        self.omegas = (1 - alpha) * self.omegas
        
        # Increase weight for the matched Gaussian: w = w + alpha
        # Use advanced indexing to update only specific (y, x, k) locations
        y_grid, x_grid = np.indices((height, width))
        self.omegas[y_grid[has_match], x_grid[has_match], match_idx[has_match]] += alpha

        # 4. Update Means and Variances for Matched Gaussians
        # We assume rho approx alpha for simplicity and speed
        
        # Get indices of matches
        match_y = y_grid[has_match]
        match_x = x_grid[has_match]
        match_k = match_idx[has_match]
        
        # Update Means: mu = (1-alpha)mu + alpha*pixel
        self.mus[match_y, match_x, match_k] = (1 - alpha) * self.mus[match_y, match_x, match_k] + alpha * frame[has_match]
        
        # Update Sigmas: sigma^2 = (1-alpha)sigma^2 + alpha*dist^2
        self.sigmaSQs[match_y, match_x, match_k] = (1 - alpha) * self.sigmaSQs[match_y, match_x, match_k] + alpha * dist_sq[match_y, match_x, match_k]

        # 5. Handle Unmatched Pixels (Replace lowest priority Gaussian)
        no_match = ~has_match
        # We replace the last Gaussian (index -1) assuming the list will be sorted later
        self.mus[no_match, -1] = frame[no_match]
        self.sigmaSQs[no_match, -1] = 36.0 # High initial variance
        self.omegas[no_match, -1] = alpha  # Low initial weight

        # 6. Normalize Weights
        weight_sum = np.sum(self.omegas, axis=2, keepdims=True)
        self.omegas /= weight_sum

        # 7. Sort Gaussians by Fitness (weight / sigma)
        fitness = self.omegas / np.sqrt(self.sigmaSQs)
        # Get indices that would sort the fitness array in descending order
        sorted_indices = np.argsort(fitness, axis=2)[:, :, ::-1]

        # Reorder parameters
        self.omegas = np.take_along_axis(self.omegas, sorted_indices, axis=2)
        self.sigmaSQs = np.take_along_axis(self.sigmaSQs, sorted_indices, axis=2)
        # Expand sorted_indices for mus (H,W,K,3)
        sorted_indices_exp = sorted_indices[:, :, :, np.newaxis]
        self.mus = np.take_along_axis(self.mus, sorted_indices_exp, axis=2)

        # 8. Background Classification
        # Find which sorted Gaussians match the current pixel
        diff_sorted = frame[:, :, np.newaxis, :] - self.mus
        dist_sq_sorted = np.sum(diff_sorted**2, axis=3)
        matches_sorted = dist_sq_sorted < (6.25 * self.sigmaSQs)

        # Identify Background Models
        # B = argmin_b ( sum(weights[:b]) > T )
        cum_weights = np.cumsum(self.omegas, axis=2)
        # We include a Gaussian in BG if the *previous* cumulative sum was < Threshold
        prev_cum = np.concatenate([np.zeros((height, width, 1)), cum_weights[:,:,:-1]], axis=2)
        is_bg_model = prev_cum < self.background_thresh

        # A pixel is BG if it matches a Gaussian that is part of the BG model
        is_bg_pixel = np.any(matches_sorted & is_bg_model, axis=2)

        # 9. Create Label Image
        # 0 for Background, 255 for Foreground
        label_img = np.where(is_bg_pixel, 0, 255).astype(np.uint8)
        
        return label_img
     
# Define mog variable outside to check existence
mog = None
     
for i in range(1, 3+1):#display first 3 labeled foreground images
    filename = 'imgs/{:04d}.jpg'.format(i)
    if not os.path.exists(filename):
        continue
    img = cv2.imread(filename)
    
    # We check if mog is None so we don't reset the model every frame (Adaptive learning)
    # Also fixes the TypeError by passing correct arguments
    if mog is None:
        mog=MOG(height=img.shape[0], width=img.shape[1], number_of_gaussians=3, background_thresh=0.5, lr=0.01) #finish this line of code
    
    label_img = mog.updateParam(img, np.ones(img.shape[:2]))
    cv2.imwrite('label{:04d}.jpg'.format(i), label_img)

