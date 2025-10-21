import cv2
import pydicom
import os
import nibabel as nib
import numpy as np
import logging
import yaml
from datetime import datetime
from PIL import Image
from pyheatmap.heatmap import HeatMap
from matplotlib.colors import Normalize
import matplotlib

import matplotlib.pyplot as plt
import pandas as pd

# Load configuration file
def load_config(config_path='config.yml'):
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Initialize logging system
def setup_logger(config):
    """Setup logging system"""
    log_dir = config['logging']['log_dir']
    log_file = config['logging']['log_file']
    log_level = config['logging']['level']
    
    # Create log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'thermal_map_whiteBG_{timestamp}_{log_file}')
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def printcolormap(grey_image, hot_map, view, filename, config, logger):
    # Create a pure white background image with the same dimensions as grey_image
    white_background = np.ones((grey_image.shape[0], grey_image.shape[1], 3), dtype=np.uint8) * 255
    
    # List to store heatmap data
    data = []
    # Traverse heatmap data, collect coordinates and corresponding values for non-zero entries
    non_zero_indices = np.where(hot_map > 0)
    l_indices, m_indices = non_zero_indices[0], non_zero_indices[1]

    # According to output, nii label coordinate orientation is LPI, while dicom coordinate orientation is LPS
    for l, m in zip(l_indices, m_indices):
        if view == 'Axial':
            data.append([l, m, hot_map[l, m]])
        else:
            data.append([l, m, hot_map[l, m]])

    image = np.array(white_background)

    if view != 'Axial':
        image = image.transpose((1, 0, 2))

    # Specify max and min values for colorbar
    max_val = np.max(hot_map)
    min_val = 1

    # Get parameters from configuration file
    alpha = config['thermal_map_whiteBG']['alpha']
    colormap_name = config['thermal_map_whiteBG']['colormap']
    figure_size = tuple(config['thermal_map_whiteBG']['figure_size'])
    circle_radius = config['thermal_map_whiteBG']['circle_radius']
    img_path = config['thermal_map_whiteBG']['img_path']
    
    # Create a new figure for drawing images and colorbars
    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    # Display pure white background image on axis
    ax.imshow(image)

    # Create normalization object to map data values to colormap range (0-1)
    norm = Normalize(vmin=min_val, vmax=max_val)

    # Select a colormap
    cmap = matplotlib.colormaps[colormap_name]

    # Traverse data points and draw heat points on the image
    for point in data:
        x, y, value = point
        color = cmap(norm(value))[:3]
        circle = plt.Circle((x, y), radius=circle_radius, facecolor=(*color, alpha), edgecolor=None)
        ax.add_patch(circle)

    # Create colorbar and add to figure
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
    cbar.set_label('Value')

    # Adjust layout for better display of image and colorbar
    plt.tight_layout()

    # Save the drawn figure
    dir_path = img_path + view
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = dir_path + '/' + filename
    plt.savefig(file_name)
    plt.close(fig)

# Main function
def main():
    # Load configuration
    config = load_config()
    logger = setup_logger(config)
    
    logger.info("="*50)
    logger.info("Starting white background heatmap generation")
    logger.info("="*50)
    
    # Read parameters from configuration file
    CT_DIR = config['thermal_map_whiteBG']['ct_dir']
    NII_DIR = config['thermal_map_whiteBG']['nii_dir']
    IMG_PATH = config['thermal_map_whiteBG']['img_path']
    window_width = config['thermal_map_whiteBG']['window_width']
    window_center = config['thermal_map_whiteBG']['window_center']
    image_shape = tuple(config['thermal_map_whiteBG']['image_shape'])
    
    logger.info(f"White background heatmap generation parameters:")
    logger.info(f"  CT image directory: {CT_DIR}")
    logger.info(f"  NII file directory: {NII_DIR}")
    logger.info(f"  Output path: {IMG_PATH}")
    logger.info(f"  Window width: {window_width}, Window center: {window_center}")
    logger.info(f"  Image shape: {image_shape}")
    
    # Check if directory exists, create if not
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
        logger.info(f"Created output directory: {IMG_PATH}")
    
    tmap = np.zeros(image_shape)

    file_names = os.listdir(NII_DIR)
    logger.info(f"Found {len(file_names)} NII files")
    
    for file_name in file_names:
        logger.info(f"Processing file: {file_name}")
        # Split string using underscore '_' as delimiter
        parts = file_name.split('_')
        # Need the second part after splitting, remove extension, split by '.' and take first part
        desired_part = parts[1].split('.')[0]
        logger.info(f"  Patient ID: {desired_part}")
        
        file_path = os.path.join(NII_DIR, file_name)
        img = nib.load(file_path)
        data = img.get_fdata()
        logger.info(f"  Coordinate system: {nib.aff2axcodes(img.affine)}")
        
        a = np.where(data != 0, 1, 0)
        tmap += a

    max_num = np.max(tmap)
    logger.info(f"Heatmap maximum value: {max_num}")

    # Create an empty pixel array
    pixel_array = []
    # Get all filenames in DICOM series
    file_names = os.listdir(CT_DIR)
    logger.info(f"Found {len(file_names)} DICOM files")

    # Traverse each file in DICOM series to get a 3D array
    dicom_data = np.zeros(image_shape)

    window_min = window_center - window_width / 2.0 + 1000
    window_max = window_center + window_width / 2.0 + 1000

    for i, file_name in enumerate(file_names):
        # Build complete file path
        file_path = os.path.join(CT_DIR, file_name)
        # Read DICOM file
        dicom_dataset = pydicom.dcmread(file_path)

        if i == 1:
            # Extract orientation attributes
            orientation = dicom_dataset.ImageOrientationPatient
            position = dicom_dataset.ImagePositionPatient
            spacing = dicom_dataset.PixelSpacing
            logger.info(f"DICOM image information:")
            logger.info(f"  Image Orientation (Patient): {orientation}")
            logger.info(f"  Image Position (Patient): {position}")
            logger.info(f"  Pixel Spacing: {spacing}")
            
        # Extract pixel data
        pixel_data = dicom_dataset.pixel_array
        # Normalization
        pixel_data = ((pixel_data - window_min) / (window_max - window_min) * 255)
        pixel_data = np.where(pixel_data < 0, 0, pixel_data)
        pixel_data = np.where(pixel_data > 255, 255, pixel_data)
        pixel_data = pixel_data.astype(np.uint8)
        dicom_data[:, :, i] = pixel_data

    logger.info("Starting coronal heatmap generation...")
    for i in range(dicom_data.shape[0]):
        view = 'Coronal'
        pixel_data = dicom_data[i, :, :]
        pixel_data = pixel_data.astype(np.uint8)
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1], 1)
        filename = 'heatmap_' + str(i) + '.png'
        printcolormap(pixel_data, tmap[:, i, :], view, filename, config, logger)
    logger.info(f"Coronal heatmap generation completed, total {dicom_data.shape[0]} slices")

    logger.info("Starting sagittal heatmap generation...")
    for i in range(dicom_data.shape[1]):
        view = 'Sagittal'
        pixel_data = dicom_data[:, i, :]
        pixel_data = pixel_data.astype(np.uint8)
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1], 1)
        filename = 'heatmap_' + str(i) + '.png'
        printcolormap(pixel_data, tmap[i, :, :], view, filename, config, logger)
    logger.info(f"Sagittal heatmap generation completed, total {dicom_data.shape[1]} slices")

    logger.info("Starting axial heatmap generation...")
    for i in range(dicom_data.shape[2]):
        view = 'Axial'
        pixel_data = dicom_data[:, :, i]
        pixel_data = pixel_data.astype(np.uint8)
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1], 1)
        filename = 'heatmap_' + str(i) + '.png'
        printcolormap(pixel_data, tmap[:, :, i], view, filename, config, logger)
    logger.info(f"Axial heatmap generation completed, total {dicom_data.shape[2]} slices")
    
    logger.info("="*50)
    logger.info("White background heatmap generation program completed!")
    logger.info("="*50)

if __name__ == "__main__":
    main()