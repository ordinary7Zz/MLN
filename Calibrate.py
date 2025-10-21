import os
import logging
import yaml
from datetime import datetime

import nibabel as nib
import numpy as np

from scipy.spatial.distance import cdist

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
    log_filename = os.path.join(log_dir, f'calibrate_{timestamp}_{log_file}')
    
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

def compute_tps_transform(src, dst, reg=1e-3):
    """
    Compute non-linear registration model using Thin Plate Spline (TPS)
    Input:
        src: (N, 3) source points
        dst: (N, 3) target points
        reg: regularization term to prevent overfitting
    Output:
        A function f(x) that can be used to map new points
    """
    # Force input to float type to avoid errors from string or integer types
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    # Safely convert regularization parameter to float (handle cases where '1e-3' is treated as string)
    try:
        reg = float(reg)
    except Exception as e:
        raise ValueError(f"Regularization parameter reg cannot be converted to float: {reg}") from e

    N = src.shape[0]
    if N == 0:
        raise ValueError("TPS registration failed: Input source point set is empty. Please check annotations or data paths.")
    
    # Calculate radial basis matrix K
    dists = cdist(src, src, 'euclidean')
    K = dists ** 2 * np.log(dists + 1e-20)
    
    # Concatenate matrices
    P = np.hstack((np.ones((N, 1)), src))
    L = np.block([
        [K + reg * np.eye(N), P],
        [P.T, np.zeros((4, 4))]
    ])
    
    V = np.vstack((dst, np.zeros((4, 3))))
    
    # Solve for weight parameters
    params = np.linalg.solve(L, V)
    W, A = params[:N], params[N:]
    
    # Return callable function
    def tps_transform(points):
        points = np.asarray(points, dtype=np.float64)
        d = cdist(points, src, 'euclidean')
        U = d ** 2 * np.log(d + 1e-20)
        return np.dot(U, W) + np.hstack((np.ones((points.shape[0], 1)), points)) @ A

    return tps_transform

def calculate_mse(transformed_points_voxel, target_points_voxel, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Mean Square Error (MSE) of registration points in mm²
    Note: Registration uses voxel coordinates, but MSE is calculated in mm²

    Parameters:
        transformed_points_voxel (numpy.ndarray): Transformed point set in voxel coordinates, shape (N, 3)
        target_points_voxel (numpy.ndarray): Target point set (reference), voxel coordinates, shape (N, 3)
        voxel_spacing (tuple or list): Voxel spacing for each dimension (x_spacing, y_spacing, z_spacing) in mm

    Returns:
        mse_mm2 (float): Mean square error in mm²
    """
    voxel_spacing = np.array(voxel_spacing)

    # Convert voxel coordinates to physical coordinates (mm)
    transformed_mm = transformed_points_voxel * voxel_spacing
    target_mm = target_points_voxel * voxel_spacing

    # Calculate squared Euclidean distance for each point
    squared_diff = np.sum((transformed_mm - target_mm) ** 2, axis=1)

    # Mean square error
    mse_mm2 = np.mean(squared_diff)

    return mse_mm2

def calculate_tre(transformed_points, target_points, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Target Registration Error (TRE) in mm

    Parameters:
    transformed_points (numpy.ndarray): Transformed point set, shape (N, 3)
    target_points (numpy.ndarray): Target point set (reference), shape (N, 3)
    voxel_spacing (tuple or list): Voxel spacing for each dimension (x_spacing, y_spacing, z_spacing) in mm

    Returns:
    tre_mean (float): Mean TRE value (mm)
    tre_std (float): Standard deviation of TRE (mm)
    """
    voxel_spacing = np.array(voxel_spacing)

    # Convert voxel coordinates to physical coordinates (mm)
    transformed_mm = transformed_points * voxel_spacing
    target_mm = target_points * voxel_spacing

    # Calculate Euclidean distance for each point (mm)
    distances = np.linalg.norm(transformed_mm - target_mm, axis=1)

    # Calculate mean and standard deviation
    tre_mean = np.mean(distances)
    tre_std = np.std(distances)

    return tre_mean, tre_std


# Calculate center points of area labels (since annotations are not exact pixel points)
def calculate_label_centers(filename):
    # Load NIfTI file
    img = nib.load(filename)
    data = img.get_fdata()

    # Extract labels
    labels = np.unique(data)

    label_centers = []
    for label in labels:
        if label == 0:
            continue  # Skip background label
        indices = np.where(data == label)
        if label == 2:
            # Split into left and right position points
            left_points_x = indices[0][:len(indices[0]) // 2]
            right_points_x = indices[0][len(indices[0]) // 2:]
            left_points_y = indices[1][:len(indices[0]) // 2]
            right_points_y = indices[1][len(indices[0]) // 2:]
            points_z = indices[2]

            # Calculate left and right center points
            left_center_x = np.mean(left_points_x)
            right_center_x = np.mean(right_points_x)
            left_center_y = np.mean(left_points_y)
            right_center_y = np.mean(right_points_y)
            center_z = np.mean(points_z)
            left_center = np.array((left_center_x, left_center_y, center_z))
            right_center = np.array((right_center_x, right_center_y, center_z))

            label_centers.append(left_center)
            label_centers.append(right_center)
            continue
        center = np.mean(indices, axis=1)
        label_centers.append(center)
    return label_centers

# Main function
def main():
    # Load configuration
    config = load_config()
    logger = setup_logger(config)
    
    logger.info("="*50)
    logger.info("Starting registration program")
    logger.info("="*50)
    
    # Read parameters from configuration file
    patients_path = config['calibrate']['patients_path']
    filebase = config['calibrate']['filebase']
    save_path = config['calibrate']['save_path']
    tps_reg = config['calibrate']['tps_reg']
    image_shape = tuple(config['calibrate']['image_shape'])
    
    logger.info(f"Registration parameters:")
    logger.info(f"  Patient data path: {patients_path}")
    logger.info(f"  Reference patient: {filebase}")
    logger.info(f"  Save path: {save_path}")
    logger.info(f"  TPS regularization parameter: {tps_reg}")
    logger.info(f"  Image shape: {image_shape}")
    
    # Check if directory exists, create if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created save directory: {save_path}")

    dir_patients = os.listdir(patients_path)
    logger.info(f"Found {len(dir_patients)} patient data folders")
    
    filenames = []
    filename_mapping = []
    for dir in dir_patients:
        path = os.path.join(patients_path, dir)
        files = os.listdir(path)
        filename = [None, None]
        for file in files:
            if 'area' in file:
                filename[0] = os.path.join(path, file)
            else:
                filename[1] = os.path.join(path, file)
        if dir == filebase:
            filenames.insert(0, filename)
            filename_mapping.insert(0, dir)
        else:
            filenames.append(filename)
            filename_mapping.append(dir)

    logger.info(f"Reference patient: {filename_mapping[0]}")
    logger.info(f"Number of patients to register: {len(filenames) - 1}")

    # Create registered file array
    after_calibrate = np.zeros(image_shape)

    # Get registration points from reference file
    centers_base = calculate_label_centers(filenames[0][0])
    centers_base = np.array(centers_base)

    # For collecting error statistics across all patients
    all_mse_values = []
    all_tre_mean_values = []
    all_tre_std_values = []

    for idx in range(1, len(filenames)):
        logger.info("-"*50)
        logger.info(f"Processing patient {idx}/{len(filenames)-1}: {filename_mapping[idx]}")
        
        # Get registration points from file to be registered
        centers_calibrate = calculate_label_centers(filenames[idx][0])
        centers_calibrate = np.array(centers_calibrate)
        
        # Calculate affine transformation matrix
        logger.info("  Computing TPS transform...")
        f_tps = compute_tps_transform(centers_calibrate, centers_base, reg=tps_reg)
        transform_matrix = f_tps(centers_calibrate)

        # Calculate TRE error
        voxel_spacing = nib.load(filenames[idx][0]).header.get_zooms()[:3]  # (x_spacing, y_spacing, z_spacing)

        # Error calculation
        mse_mm2 = calculate_mse(f_tps(centers_calibrate), centers_base, voxel_spacing)
        tre_mean_mm, tre_std_mm = calculate_tre(f_tps(centers_calibrate), centers_base, voxel_spacing)
        logger.info(f"  MSE (mm²): {mse_mm2:.6f}")
        logger.info(f"  TRE (mm): {tre_mean_mm:.6f} ± {tre_std_mm:.6f}")
        
        # Collect error data for statistics
        all_mse_values.append(mse_mm2)
        all_tre_mean_values.append(tre_mean_mm)
        all_tre_std_values.append(tre_std_mm)

        # Apply transform to lymph node data
        img = nib.load(filenames[idx][1])
        data = img.get_fdata()

        lymphnode_labels = np.unique(data)
        coords = np.where(data != 0)
        coords_h = np.stack([coords[0], coords[1], coords[2]], axis=-1)
        # Apply affine transformation
        transformed_coords_h = f_tps(coords_h)  # Transformed coordinates

        # Convert transformed points back to regular coordinates
        transformed_coords = transformed_coords_h[:, :3]  # Take first three dimensions

        # Create a new empty array to store results
        transformed_data = np.zeros_like(data)

        # Note: Transformed coordinates may exceed original boundaries, need to handle
        for i in range(transformed_coords.shape[0]):
            x_new, y_new, z_new = transformed_coords[i].astype(int)
            if 0 <= x_new < transformed_data.shape[0] and 0 <= y_new < transformed_data.shape[1] and 0 <= z_new < transformed_data.shape[2]:
                transformed_data[x_new, y_new, z_new] = data[coords_h[i][0].astype(int), coords_h[i][1].astype(int), coords_h[i][2].astype(int)]
        
        new_img = nib.Nifti1Image(transformed_data, img.affine)  # Keep original affine matrix
        img_name = 'lymphnode_' + filename_mapping[idx] + '.nii.gz'
        img_path = os.path.join(save_path, img_name)
        nib.save(new_img, img_path)
        logger.info(f"  Saved registration result: {img_path}")
    
    logger.info("="*50)
    logger.info("Registration program completed!")
    logger.info("="*50)
    
    # Calculate and output average error metrics
    if len(all_mse_values) > 0:
        avg_mse = np.mean(all_mse_values)
        avg_tre_mean = np.mean(all_tre_mean_values)
        avg_tre_std = np.mean(all_tre_std_values)
        
        logger.info("")
        logger.info("="*50)
        logger.info("Registration Error Statistics for All Patients")
        logger.info("="*50)
        logger.info(f"Number of registered patients: {len(all_mse_values)}")
        logger.info(f"Average MSE (mm²): {avg_mse:.6f}")
        logger.info(f"Average TRE (mm): {avg_tre_mean:.6f} ± {avg_tre_std:.6f}")
        logger.info(f"MSE range: [{min(all_mse_values):.6f}, {max(all_mse_values):.6f}]")
        logger.info(f"TRE range: [{min(all_tre_mean_values):.6f}, {max(all_tre_mean_values):.6f}]")
        logger.info("="*50)
    else:
        logger.warning("No patient data was registered")

if __name__ == "__main__":
    main()