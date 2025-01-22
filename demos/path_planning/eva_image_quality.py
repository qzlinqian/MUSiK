#!/usr/bin/env python
# coding: utf-8

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import numpy as np
from scipy.ndimage import sobel

def preprocess_image(image):
    image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
    image_smoothed = gaussian_filter(image_normalized, sigma=2)  # Adjust sigma as needed
    return image_smoothed

def segment_foreground(image):
    thresh = threshold_otsu(image)
    mask_foreground = image > thresh  # Binary mask for bright regions
    return mask_foreground

def segment_background(image, mask_foreground, buffer=5):
    # Invert the foreground mask
    mask_background = ~mask_foreground
    
    # Optionally remove regions near the target
    from scipy.ndimage import binary_dilation
    mask_foreground_dilated = binary_dilation(mask_foreground, iterations=buffer)
    mask_background = mask_background & ~mask_foreground_dilated
    
    return mask_background

def generate_masks(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Segment the foreground
    mask_foreground = segment_foreground(preprocessed_image)
    
    # Segment the background
    mask_background = segment_background(preprocessed_image, mask_foreground)
    
    return mask_foreground, mask_background

def compute_contrast(image, mask_foreground, mask_background):
    # Extract pixel intensities from foreground and background
    foreground = image[mask_foreground]
    background = image[mask_background]
    
    # Compute mean intensity difference
    contrast = abs(np.mean(foreground) - np.mean(background))
    return contrast

def compute_resolution(image):
    # Apply Gaussian filter to simulate blur and detect spread
    blurred_image = gaussian_filter(image, sigma=1)
    
    # Calculate edge sharpness (e.g., gradient magnitude)
    gradient_magnitude = np.abs(np.gradient(blurred_image))
    resolution = -np.mean(gradient_magnitude)  # Lower spread means better resolution
    
    return resolution

def compute_snr(image, mask_foreground):
    # Extract signal region
    signal = image[mask_foreground]
    
    # Compute mean signal and standard deviation (noise level)
    mean_signal = np.mean(signal)
    noise_std = np.std(signal)
    
    # Calculate SNR
    snr = 10 * np.log10(mean_signal / noise_std + 1e-10)  # Avoid division by zero
    return snr

def compute_edge_clarity(image):
    # Apply Sobel filter to detect edges
    edge_map = sobel(image)
    
    # Calculate edge strength
    edge_clarity = np.mean(np.abs(edge_map))
    return edge_clarity

def evaluate_image_quality(image):
    mask_foreground, mask_background = generate_masks(image)
    contrast = compute_contrast(image, mask_foreground, mask_background)
    resolution = compute_resolution(image)
    snr = compute_snr(image, mask_foreground)
    edge_clarity = compute_edge_clarity(image)
    
    # Return the metrics for multi-objective optimization
    return [contrast, resolution, snr, edge_clarity]

