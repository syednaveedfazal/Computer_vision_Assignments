import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# ==============================================================================
# 0. Setup and Image Loading
# ==============================================================================
print("--- 0. Setup: Loading Images ---")

'''
TODO: Load the original image 'bonn.jpg' and noisy image 'bonn_noisy.jpg'
Convert both to grayscale and prepare the noisy image in float format (0-1 range)
Calculate and print the PSNR of the noisy image compared to the original
'''

# Load images here
original_img_color = cv2.imread('bonn.jpg')
original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
noisy_img = cv2.imread('bonn_noisy.jpg')           # Load bonn_noisy.jpg
# Convert noisy image to grayscale and to float in range 0-1 for custom filters
noisy_img_gray = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
noisy_img_float_01 = noisy_img_gray.astype(np.float32) / 255.0

# Calculate PSNR of noisy image
psnr_noisy = None

# Display original and noisy images
# TODO: Create a figure showing original and noisy images side by side

cv2.imshow('Original Image', original_img_color)
cv2.imshow('Noisy Image', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# i have written this helpter function to display comparions of filters of custom and cv2 filters 
def show_filter_comparison(title, noisy_img, cv2_result, custom_result, psnr_cv2, psnr_custom):
    # Convert custom float result to uint8 for display and PSNR consistency
    custom_result_uint8 = (np.clip(custom_result, 0.0, 1.0) * 255).astype(np.uint8)
    # matlotlib is used to diplsay the results side by side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2_result, cmap='gray')
    plt.title(f'CV2 Result (PSNR: {psnr_cv2:.2f} dB)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(custom_result_uint8, cmap='gray')
    plt.title(f'Custom Result (PSNR: {psnr_custom:.2f} dB)')
    plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.show()
# ==============================================================================
# Custom Filter Definitions (for parts a, b, c)
# ==============================================================================

def custom_gaussian_filter(image, kernel_size, sigma):
       
        center = kernel_size // 2 
        x,y = np.mgrid[-center:center+1,-center:center+1]
        gaussian_kernel = np.exp(-(x**2+y**2)/(2*sigma**2))
        gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)

        padding_size = kernel_size//2
        # i have added padding so the edge pixels can also be processed easily or 
        # else i have to use conditionals for edge pixels 
        padded_image = np.pad(image, pad_width=padding_size,mode='reflect')
        filter_image = np.zeros_like(image)
        height, width = image.shape
        for x in range(height):
            for y in range(width):
                region = padded_image[x:x+kernel_size,y:y+kernel_size]
                filter_image[x,y] = np.sum(region*gaussian_kernel)
        return filter_image
def custom_median_filter(image, kernel_size):
   
    extra_padding = kernel_size //2
    padded_image = np.pad(image, pad_width=extra_padding,mode='reflect')
    filtered_image = np.zeros_like(image)
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            region = padded_image[x:x+kernel_size,y:y+kernel_size]
            filtered_image[x,y] = np.median(region)
    return filtered_image


def custom_bilateral_filter(image, d, sigma_color, sigma_space):
    pad_size = d // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    filtered_image = np.zeros_like(image)
    
    # Pre-calculate the spatial Gaussian kernel
    x, y = np.mgrid[-pad_size:pad_size + 1, -pad_size:pad_size + 1]
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + d, j:j + d]
            center_pixel_intensity = window[pad_size, pad_size]
            
            # Calculate the intensity (range) kernel
            intensity_diff = window - center_pixel_intensity
            range_kernel = np.exp(-(intensity_diff**2) / (2 * sigma_color**2))
            
            # Combine kernels and normalize
            combined_kernel = spatial_kernel * range_kernel
            weights_sum = np.sum(combined_kernel)
            
            filtered_image[i, j] = np.sum(window * combined_kernel) / weights_sum
            
    return filtered_image


# ==============================================================================
# 1. Filter Application (Parts a, b, c)
# ==============================================================================
print("\n--- 1. Filter Application (Parts a, b, c) ---")

# Default Parameters
K_DEFAULT = 7
S_DEFAULT = 2.0
D_DEFAULT = 9
SC_DEFAULT = 100  # cv2 range (0-255)
SS_DEFAULT = 75

# -------------------------- a) Gaussian Filter --------------------------
print("a) Applying Gaussian Filter...")
# Apply Gaussian blur using OpenCV on the grayscale noisy image
gaussian_blur = cv2.GaussianBlur(noisy_img_gray, (K_DEFAULT, K_DEFAULT), S_DEFAULT)

# Apply custom Gaussian filter on the float (0-1) grayscale image
custom_gaussian_filter_blur = custom_gaussian_filter(noisy_img_float_01, K_DEFAULT, S_DEFAULT)

# custom_gaussian_filter_blur is float in [0,1]; convert back to uint8 for display/concatenation
if custom_gaussian_filter_blur.dtype.kind == 'f':
    custom_disp = (np.clip(custom_gaussian_filter_blur, 0.0, 1.0) * 255.0).astype(np.uint8)
else:
    custom_disp = custom_gaussian_filter_blur.astype(np.uint8)

# Now both gaussian_blur and custom_disp are single-channel uint8 images and can be concatenated
comparison = cv2.hconcat([gaussian_blur, custom_disp])
cv2.imshow('Gaussian: OpenCV | Custom', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img_gray,(K_DEFAULT,K_DEFAULT),S_DEFAULT)
psnr_gaussian_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_gaussian_cv2, data_range=255)
denoised_gaussian_custom_float = custom_gaussian_filter(noisy_img_float_01, K_DEFAULT, S_DEFAULT)
denoised_gaussian_custom = (np.clip(denoised_gaussian_custom_float, 0, 1) * 255).astype(np.uint8)
psnr_gaussian_custom = peak_signal_noise_ratio(original_img_gray, denoised_gaussian_custom, data_range=255)

print(f"PSNR (Gaussian CV2): {psnr_gaussian_cv2:.2f} dB")
print(f"PSNR (Gaussian Custom): {psnr_gaussian_custom:.2f} dB")
show_filter_comparison('Gaussian Filter Comparison', noisy_img_gray, denoised_gaussian_cv2, denoised_gaussian_custom_float, psnr_gaussian_cv2, psnr_gaussian_custom)

# Display results here


# -------------------------- b) Median Filter --------------------------
print("b) Applying Median Filter...")



denoised_median_cv2 = cv2.medianBlur(noisy_img_gray,K_DEFAULT)
psnr_median_cv2 = peak_signal_noise_ratio(original_img_gray,denoised_median_cv2,data_range=255)
denoised_median_custom = custom_median_filter(noisy_img_float_01,K_DEFAULT)
denoised_median_float_custom = (np.clip(denoised_median_custom,0,1)*255).astype(np.uint8)
comparion_median = cv2.hconcat([denoised_median_cv2,denoised_median_float_custom])
cv2.imshow('Median: OpenCV | Custom',comparion_median)
cv2.waitKey(0)
cv2.destroyAllWindows()
psnr_median_custom = peak_signal_noise_ratio(original_img_gray,denoised_median_float_custom,data_range=255)
print(f"PSNR (Median CV2): {psnr_median_cv2:.2f} dB")
print(f"PSNR (Median Custom): {psnr_median_custom:.2f} dB")
show_filter_comparison('Median Filter Comparision',noisy_img_gray,denoised_median_cv2,denoised_median_custom,psnr_median_cv2,psnr_median_custom)
# Display results here


# -------------------------- c) Bilateral Filter --------------------------
print("c) Applying Bilateral Filter...")



denoised_bilateral_cv2 = cv2.bilateralFilter(noisy_img_gray,D_DEFAULT,SC_DEFAULT,SS_DEFAULT)
psnr_bilateral_cv2 = peak_signal_noise_ratio(original_img_gray,denoised_bilateral_cv2,data_range=255)

denoised_bilateral_custom = custom_bilateral_filter(noisy_img_float_01,D_DEFAULT,SC_DEFAULT,SS_DEFAULT)
denoised_bilateral_float_custom = (np.clip(denoised_bilateral_custom,0,1)*255).astype(np.uint8)
comparison_bilateral = cv2.hconcat([denoised_bilateral_cv2,denoised_bilateral_float_custom])
cv2.imshow('Bilateral: OpenCV | Custom',comparison_bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
psnr_bilateral_custom = peak_signal_noise_ratio(original_img_gray,denoised_bilateral_float_custom,data_range=255)
comparison_bilateral_show = show_filter_comparison('Bilateral Filter Comparision',noisy_img_gray,denoised_bilateral_cv2,denoised_bilateral_custom,psnr_bilateral_cv2,psnr_bilateral_custom)
print(f"PSNR (Bilateral CV2): {psnr_bilateral_cv2:.2f} dB")
print(f"PSNR (Bilateral Custom): {psnr_bilateral_custom:.2f} dB")
# Display results here


# ==============================================================================
# 2. Performance Comparison (Part d)
# ==============================================================================
print("\n--- d) Performance Comparison ---")
'''
TODO:
1. Compare PSNR values of all three filters
2. Determine which filter performs best
3. Display side-by-side comparison of all filtered images
4. Print the results with the best performing filter highlighted
'''
psnr_results = {
    'Gaussian CV2': psnr_gaussian_cv2,
    'Gaussian Custom':psnr_gaussian_custom,
    'Median CV2': psnr_median_cv2,
    'Median Custom': psnr_median_custom,
    'Bilateral CV2': psnr_bilateral_cv2,
    'Bilateral Custom': psnr_bilateral_custom
}
best_filter = max(psnr_results,key=psnr_results.get)

for filter_name, psnr_value in psnr_results.items():
    result = "Best" if filter_name == best_filter else ""
    print(f'{filter_name}: {psnr_value:.2f} dB {result}')

print(f"\nThe **{best_filter} Filter** performed the best with the default parameters.")

# i have used matplotlib to display all results side by side
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.imshow(noisy_img_gray,cmap='gray')
plt.title(f'Noisy Image (PSNR: {psnr_noisy} dB)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(denoised_gaussian_cv2, cmap='gray')
plt.title(f'Gaussian Filter (PSNR: {psnr_gaussian_cv2} dB)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(denoised_median_cv2, cmap='gray')
plt.title(f'Median Filter (PSNR: {psnr_median_cv2} dB)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(denoised_bilateral_cv2, cmap='gray')
plt.title(f'Bilateral Filter (PSNR: {psnr_bilateral_cv2} dB)')
plt.axis('off')

plt.suptitle('Overall Filter Performance Comparison', fontsize=16)
plt.show()
# ==============================================================================
# 3. Parameter Optimization (Part e)
# ==============================================================================
# Helper function to display results

def run_optimization(original_img, noisy_img):
    """
    Optimize parameters for all three filters to maximize PSNR by performing a grid search.
    """
    results = {
        "gaussian": {"best_psnr": -1, "params": {}},
        "median": {"best_psnr": -1, "params": {}},
        "bilateral": {"best_psnr": -1, "params": {}}
    }

    # 1. Gaussian optimization
    print("Optimizing Gaussian Filter...")
    kernel_sizes_gauss = [3, 5, 7, 9]
    sigmas = [0.5, 1.0, 1.5, 2.0,3.0,4.0]
    for k in kernel_sizes_gauss:
        for s in sigmas:
            denoised = cv2.GaussianBlur(noisy_img, (k, k), s)
            psnr = peak_signal_noise_ratio(original_img, denoised, data_range=255)
            if psnr > results["gaussian"]["best_psnr"]:
                results["gaussian"]["best_psnr"] = psnr
                results["gaussian"]["params"] = {"kernel_size": k, "sigma": s}

    # 2. Median optimization
    print("Optimizing Median Filter...")
    kernel_sizes_median = [3, 5, 7]
    for k in kernel_sizes_median:
        denoised = cv2.medianBlur(noisy_img, k)
        psnr = peak_signal_noise_ratio(original_img, denoised, data_range=255)
        if psnr > results["median"]["best_psnr"]:
            results["median"]["best_psnr"] = psnr
            results["median"]["params"] = {"kernel_size": k}

    # 3. Bilateral optimization (can be slow)
    print("Optimizing Bilateral Filter...")
    diameters = [5, 9]
    sigma_colors = [25, 50, 75]
    sigma_spaces = [25, 50, 75]
    for d in diameters:
        for sc in sigma_colors:
            for ss in sigma_spaces:
                denoised = cv2.bilateralFilter(noisy_img, d, sc, ss)
                psnr = peak_signal_noise_ratio(original_img, denoised, data_range=255)
                if psnr > results["bilateral"]["best_psnr"]:
                    results["bilateral"]["best_psnr"] = psnr
                    results["bilateral"]["params"] = {"d": d, "sigma_color": sc, "sigma_space": ss}
    
    return results


print("--- Starting Parameter Optimization ---")
optimal_results = run_optimization(original_img_gray, noisy_img_gray)
print("--- Optimization Complete ---\n")


# 2. Extract optimal parameters for each filter
gauss_opt = optimal_results['gaussian']
median_opt = optimal_results['median']
bilateral_opt = optimal_results['bilateral']


# 5. Print the optimal parameters clearly
print("--- Optimal Parameters Found ---")
print(f"Gaussian Filter:  Best PSNR = {gauss_opt['best_psnr']:.2f} dB")
print(f"  > Parameters: {gauss_opt['params']}")
print(f"Median Filter:    Best PSNR = {median_opt['best_psnr']:.2f} dB")
print(f"  > Parameters: {median_opt['params']}")
print(f"Bilateral Filter: Best PSNR = {bilateral_opt['best_psnr']:.2f} dB")
print(f"  > Parameters: {bilateral_opt['params']}\n")


# 3. Apply filters using optimal parameters
# Gaussian
k_gauss = gauss_opt['params']['kernel_size']
s_gauss = gauss_opt['params']['sigma']
denoised_gauss_opt = cv2.GaussianBlur(noisy_img_gray, (k_gauss, k_gauss), s_gauss)

# Median
k_median = median_opt['params']['kernel_size']
denoised_median_opt = cv2.medianBlur(noisy_img_gray, k_median)

# Bilateral
d_bi = bilateral_opt['params']['d']
sc_bi = bilateral_opt['params']['sigma_color']
ss_bi = bilateral_opt['params']['sigma_space']
denoised_bilateral_opt = cv2.bilateralFilter(noisy_img_gray, d_bi, sc_bi, ss_bi)


# 4. Display the optimized results in a 2x2 grid
print("--- Displaying Optimized Results ---")
plt.figure(figsize=(12, 12))

# Plot 1: Noisy Image
plt.subplot(2, 2, 1)
plt.imshow(noisy_img_gray, cmap='gray')
plt.title(f'Noisy Image (PSNR: {psnr_noisy} dB)')
plt.axis('off')

# Plot 2: Optimized Gaussian Filter Result
plt.subplot(2, 2, 2)
plt.imshow(denoised_gauss_opt, cmap='gray')
plt.title(f'Optimized Gaussian (PSNR: {gauss_opt["best_psnr"]} dB)')
plt.axis('off')

# Plot 3: Optimized Median Filter Result
plt.subplot(2, 2, 3)
plt.imshow(denoised_median_opt, cmap='gray')
plt.title(f'Optimized Median (PSNR: {median_opt["best_psnr"]} dB)')
plt.axis('off')

# Plot 4: Optimized Bilateral Filter Result
plt.subplot(2, 2, 4)
plt.imshow(denoised_bilateral_opt, cmap='gray')
plt.title(f'Optimized Bilateral (PSNR: {bilateral_opt["best_psnr"]} dB)')
plt.axis('off')

plt.suptitle('Optimized Filter Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for the main title
plt.show()
