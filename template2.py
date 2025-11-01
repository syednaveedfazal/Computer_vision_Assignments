# Template for Exercise 2 –  Fourier Transform and Image Reconstruction
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted spectrum
    - The magnitude
    - The phase
    """
    """
    args:
        img: Input grayscale image (numpy array)
    returns:
        f_shifted: Shifted Fourier Transform (numpy array of complex numbers)
        magnitude_spectrum: Magnitude spectrum (numpy array)
        phase_spectrum: Phase spectrum (numpy array)
    """

    # Compute the 2D Discrete Fourier Transform
    # The input image is converted to a float type before FFT
    f_transform = np.fft.fft2(img)
    # f_transform is a 2D array of complex numbers representing the frequency domain of the image.


    f_shifted = np.fft.fftshift(f_transform)
    # f_shifted shifts the low-frequency components to the center of the spectrum for better visualization.

    
    magnitude_spectrum = np.abs(f_shifted)
    # The magnitude spectrum is calculated as the absolute value of these complex numbers.
    # As per the reference, Magnitude A = sqrt(R(ω)² + I(ω)²), which is calculated by np.abs()


    phase_spectrum = np.angle(f_shifted)
    # The phase spectrum is the angle of each complex number.
    # As per the reference, Phase φ = tan⁻¹(I(ω) / R(ω)), which is calculated by np.angle()


    return f_shifted, magnitude_spectrum, phase_spectrum


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    """
    """
    args:
        mag: Magnitude spectrum (numpy array)
        phase: Phase spectrum (numpy array)
    returns:
        img_reconstructed_normalized: Reconstructed image normalized to 0-255 (numpy array)
    """
    
    complex_spectrum = mag * np.exp(1j * phase)
    # this is euler's formula to convert polar to complex form
    # Combine the magnitude and phase to recreate the complex f-domain image.
    # The formula is: complex_value = magnitude * e^(j * phase)

    
    f_ishifted = np.fft.ifftshift(complex_spectrum)
    # f_ishifted shifts the zero-frequency component back to the top-left corner
    # this process is needed before applying the inverse FFT.


    img_reconstructed_complex = np.fft.ifft2(f_ishifted)
    # ifft2 takes the frequency domain representation and converts it back to the spatial domain.
    # resulting into real and imaginary parts.


    # Take the absolute value to get the real-valued image in the spatial domain.
    # here imaginary part is negligible
    img_reconstructed = np.abs(img_reconstructed_complex)
    
    # 5. Normalize the pixel values to the 0-255 range and convert to an 8-bit
    # unsigned integer, which is the standard format for saving and displaying images.
    img_reconstructed_normalized = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_reconstructed_normalized


def compute_mad(a, b):
    
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    """
    args:
        a: first image (numpy array)
        b: second image (numpy array)
    returns:
        mad: Mean Absolute Difference between the two images
    """
    absolute_difference = np.abs(a - b)
    # Calculate the absolute difference between each pair of corresponding pixels
    # abs is necessary to avoid negative differences
    
    
    mad = np.mean(absolute_difference)
    # Compute the mean of all these differences
    
    return mad



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
# ==========================================================

# TODO: 1. Load the two grayscale images (1.png and 2.png)
img1 = cv2.imread('data/1.png', 0)
img2 = cv2.imread('data/2.png', 0)
# The second argument '0' loads the image in grayscale mode



# TODO: 2. Compute magnitude and phase of both images
fft_shifted1, magnitude1, phase1 = compute_fft(img1)
# Compute FFT for the second image
fft_shifted2, magnitude2, phase2 = compute_fft(img2)



# TODO: 3. Swap magnitude and phase between the two images
# To reconstruct an image from its Fourier components, we need to recombine
# the magnitude and phase into a complex number array.
# The formula for a complex number in polar coordinates is: z = r * e^(iθ)
# where 'r' is the magnitude and 'θ' is the phase angle.
# In numpy, this is expressed as: magnitude * np.exp(1j * phase)


combined_spectrum_1 = magnitude1 * np.exp(1j * phase2)
# Magnitude of Image 1 and Phase of Image 2
# This complex array represents the Fourier spectrum of the first reconstructed image.

combined_spectrum_2 = magnitude2 * np.exp(1j * phase1)
# Magnitude of Image 2 and Phase of Image 1
# This complex array represents the Fourier spectrum of the second reconstructed image.



# TODO: 4. Reconstruct and save the swapped results

# Combine the magnitude from 1.png with the phase from 2.png
reconstructed_mag1_phase2 = reconstruct_from_mag_phase(magnitude1, phase2)

# Combine the phase from 1.png with the magnitude from 2.png
reconstructed_mag2_phase1 = reconstruct_from_mag_phase(magnitude2, phase1)

# Save the resulting images as requested
cv2.imwrite('reconstructed_mag1_phase2.png', reconstructed_mag1_phase2)
cv2.imwrite('reconstructed_mag2_phase1.png', reconstructed_mag2_phase1)

# 3. Plot the resulting images alongside the originals for comparison
images_to_plot = [
    img1,
    img2,
    reconstructed_mag1_phase2,
    reconstructed_mag2_phase1
]

labels_for_plot = [
    'Original Image 1', 'Original Image 2',
    'Reconstructed:\nMag(1) + Phase(2)',
    'Reconstructed:\nMag(2) + Phase(1)'
]

plot_results(images=images_to_plot, labels=labels_for_plot, cols=2, figsize=(10, 10))
# Using the plot function to display the results in a 2x2 grid






# TODO: 5. Compute and print the MAD values between originals and reconstructions
mad_phase_dominant = compute_mad(img2, reconstructed_mag1_phase2)
# Compare the image reconstructed with the phase of Image 2 (reconstructed_mag1_phase2)
# to the original Image 2. Since they share the same phase, the MAD should be low.


mad_phase_dominant_2= compute_mad(img1, reconstructed_mag2_phase1)
# Compare the image reconstructed with the phase of Image 1 (reconstructed_mag2_phase1)
# to the original Image 1. Since they share the same phase, this MAD should also be low.



mad_magnitude_minor = compute_mad(img1, reconstructed_mag1_phase2)
mad_magnitude_minor_2 = compute_mad(img2, reconstructed_mag2_phase1)
# compare the reconstructed images to the images they took their magnitude from.
# This MAD is expected to be high, showing magnitude is less important for structure.

print("--- Mean Absolute Difference (MAD) Analysis ---")
print(f"MAD between Original Image 2 and Reconstructed (Mag 1 + Phase 2): {mad_phase_dominant:.2f}")
print(f"MAD between Original Image 1 and Reconstructed (Mag 2 + Phase 1): {mad_phase_dominant_2:.2f}")
print("-" * 20)
print(f"MAD between Original Image 1 and Reconstructed (Mag 1 + Phase 2): {mad_magnitude_minor:.2f}")
print(f"MAD between Original Image 2 and Reconstructed (Mag 2 + Phase 1): {mad_magnitude_minor_2:.2f}")
print("\n--- Analysis Findings ---")
print("The results clearly show a much lower MAD when the phase is preserved. The reconstructed image strongly resembles the original image that provided the phase component, not the magnitude component. This demonstrates that the phase spectrum contains the crucial information about the spatial structure (edges, shapes) of an image, while the magnitude spectrum primarily encodes the frequency content (contrast, brightness levels).")




# TODO: 6. Visualize all images (originals, magnitude, phase, reconstructions)
magnitude_display1 = np.log1p(magnitude1)
magnitude_display2 = np.log1p(magnitude2)
# log1p converts the magnitude to a logarithmic scale for better visualization


# Prepare the lists for plotting all results in one figure
all_images_to_plot = [
    img1, magnitude_display1, phase1, reconstructed_mag2_phase1,
    img2, magnitude_display2, phase2, reconstructed_mag1_phase2
]

all_labels = [
    'Original Image 1', 'Log-Magnitude 1', 'Phase 1', 'Reconstructed (Mag 2 + Phase 1)',
    'Original Image 2', 'Log-Magnitude 2', 'Phase 2', 'Reconstructed (Mag 1 + Phase 2)'
]

# Plot all the results in a 2x4 grid for a comprehensive overview
plot_results(images=all_images_to_plot, labels=all_labels, cols=4, figsize=(20, 10))






