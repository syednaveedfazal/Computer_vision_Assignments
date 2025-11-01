import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    # basically we are creating a 1/k^2 matrix 
    kernel = np.ones((k,k), dtype=np.float32)
    return kernel /(k*k)
   

def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    # this will create a grid of (x,y) coordinates based on center which is k-1/2 
    xx, yy = np.meshgrid(ax, ax)
    # Calculate the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normalize the kernel to ensure its elements sum to 1
    return kernel / np.sum(kernel)



def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    img = img.astype(np.float64) 
    # flipping the kernel for convolution if we use correlation than we dont need to flip
    h = np.flipud(np.fliplr(h))
    k = h.shape[0]
    padding = k//2
    padded_image = np.pad(img,padding,mode='constant',constant_values=0)
    result = np.zeros_like(img)
    height, widht = img.shape
    # iterate over every pixel in the original image and get the region and perform element wise multiplication and sum
    for y in range(height):
        for x in range(widht):
            region = padded_image[y:y+k,x:x+k]
            result[y,x] = np.sum(region * h)
    return result


def freq_linear_conv(img, h):

    img_h, img_w = img.shape
    h_h, h_w = h.shape

    # Pad the image and kernel to a size that is the sum of their dimensions minus one
    padded_h = np.zeros((img_h + h_h - 1, img_w + h_w - 1))
    padded_h[:h_h, :h_w] = h
    padded_img = np.zeros_like(padded_h)
    padded_img[:img_h, :img_w] = img

    # Perform FFT
    img_fft = np.fft.fft2(padded_img)
    h_fft = np.fft.fft2(padded_h)

    # Multiply in the frequency domain
    result_fft = img_fft * h_fft

    # Perform inverse FFT
    result_ifft = np.fft.ifft2(result_fft)
    result = np.real(result_ifft)

    # Crop to the original image size
    start_row = (h_h - 1) // 2
    start_col = (h_w - 1) // 2
    return result[start_row:start_row + img_h, start_col:start_col + img_w]


def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))


original_img_gray = cv2.imread('data/bonn.jpg',0)
kernel_size = 9
sigma = 1.5
box_kernel = make_box_kernel(kernel_size)
gauss_kernel = make_gauss_kernel(kernel_size, sigma)
box_spatial = conv2_same_zero(original_img_gray, box_kernel)
gaussian_spatial = conv2_same_zero(original_img_gray, gauss_kernel)

# 4. Apply both filters in the frequency domain
box_freq = freq_linear_conv(original_img_gray, box_kernel)
gaussian_freq = freq_linear_conv(original_img_gray, gauss_kernel)

# 5. Compute and print MAD between spatial and frequency outputs
mad_box = compute_mad(box_spatial, box_freq)
mad_gaussian = compute_mad(gaussian_spatial, gaussian_freq)

print(f"MAD between spatial and frequency domain for Box Filter: {mad_box}")
print(f"MAD between spatial and frequency domain for Gaussian Filter: {mad_gaussian}")

# 6. Visualize all results
plt.figure(figsize=(12, 12))

plt.subplot(2, 3, 1)
plt.imshow(original_img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(box_spatial, cmap='gray')
plt.title('Box Filter (Spatial)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(gaussian_spatial, cmap='gray')
plt.title('Gaussian Filter (Spatial)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(box_freq, cmap='gray')
plt.title('Box Filter (Frequency)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gaussian_freq, cmap='gray')
plt.title('Gaussian Filter (Frequency)')
plt.axis('off')

# To visualize the spectrum, we can look at the magnitude of the FFT of one of the kernels
box_kernel_fft = np.fft.fft2(box_kernel, s=(256, 256))
box_kernel_fft_mag = np.abs(np.fft.fftshift(box_kernel_fft))

plt.subplot(2, 3, 4)
plt.imshow(np.log(1 + box_kernel_fft_mag), cmap='hot')
plt.title('Box Filter Spectrum (log scale)')
plt.axis('off')


plt.tight_layout()
plt.show()

# 7. Verify that MAD < 1×10⁻⁷ for both filters
mad_threshold = 1e-7
print("\nVerification:")
print(f"Box Filter MAD < {mad_threshold}: {mad_box < mad_threshold} and value is {mad_box}")
print(f"Gaussian Filter MAD < {mad_threshold}: {mad_gaussian < mad_threshold} and value is {mad_gaussian}")

