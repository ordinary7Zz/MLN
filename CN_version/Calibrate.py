import os
import logging
import yaml
from datetime import datetime

import nibabel as nib
import numpy as np

from scipy.spatial.distance import cdist

# 加载配置文件
def load_config(config_path='config.yml'):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 初始化日志系统
def setup_logger(config):
    """设置日志系统"""
    log_dir = config['logging']['log_dir']
    log_file = config['logging']['log_file']
    log_level = config['logging']['level']
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'calibrate_{timestamp}_{log_file}')
    
    # 配置日志
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
    使用 Thin Plate Spline (TPS) 计算非线性配准模型
    输入:
        src: (N, 3) 源点
        dst: (N, 3) 目标点
        reg: 正则项，防止过拟合
    输出:
        一个函数 f(x) 可用于映射新的点
    """
    # 强制将输入转换为浮点类型，避免字符串或整型导致的数值计算错误
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    # 将正则化参数安全转换为浮点数（处理例如 '1e-3' 被当作字符串的情况）
    try:
        reg = float(reg)
    except Exception as e:
        raise ValueError(f"正则化参数 reg 无法转换为浮点数: {reg}") from e

    N = src.shape[0]
    if N == 0:
        raise ValueError("TPS 配准失败：输入的源点集为空。请检查标注或数据路径是否正确。")
    
    # 计算径向基矩阵 K
    dists = cdist(src, src, 'euclidean')
    K = dists ** 2 * np.log(dists + 1e-20)
    
    # 拼接矩阵
    P = np.hstack((np.ones((N, 1)), src))
    L = np.block([
        [K + reg * np.eye(N), P],
        [P.T, np.zeros((4, 4))]
    ])
    
    V = np.vstack((dst, np.zeros((4, 3))))
    
    # 求解权重参数
    params = np.linalg.solve(L, V)
    W, A = params[:N], params[N:]
    
    # 返回可调用函数
    def tps_transform(points):
        points = np.asarray(points, dtype=np.float64)
        d = cdist(points, src, 'euclidean')
        U = d ** 2 * np.log(d + 1e-20)
        return np.dot(U, W) + np.hstack((np.ones((points.shape[0], 1)), points)) @ A

    return tps_transform

def calculate_mse(transformed_points_voxel, target_points_voxel, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    计算配准点的均方误差（MSE），单位为 mm²
    注意：配准使用的是体素坐标，但计算 MSE 时会转换成 mm²

    参数:
        transformed_points_voxel (numpy.ndarray): 变换后的点集，体素坐标，形状 (N, 3)
        target_points_voxel (numpy.ndarray): 目标点集（基准点集），体素坐标，形状 (N, 3)
        voxel_spacing (tuple or list): 每个维度的体素间距 (x_spacing, y_spacing, z_spacing)，单位 mm

    返回:
        mse_mm2 (float): 均方误差，单位 mm²
    """
    voxel_spacing = np.array(voxel_spacing)

    # 将体素坐标转换为物理坐标（mm）
    transformed_mm = transformed_points_voxel * voxel_spacing
    target_mm = target_points_voxel * voxel_spacing

    # 计算每个点的平方欧氏距离
    squared_diff = np.sum((transformed_mm - target_mm) ** 2, axis=1)

    # 均方误差
    mse_mm2 = np.mean(squared_diff)

    return mse_mm2

def calculate_tre(transformed_points, target_points, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    计算配准点的 Target Registration Error (TRE)，单位为 mm

    参数:
    transformed_points (numpy.ndarray): 变换后的点集，形状为 (N, 3)
    target_points (numpy.ndarray): 目标点集（基准点集），形状为 (N, 3)
    voxel_spacing (tuple or list): 每个维度的体素间距 (x_spacing, y_spacing, z_spacing)，单位 mm

    返回:
    tre_mean (float): 平均 TRE 值（mm）
    tre_std (float): TRE 的标准差（mm）
    """
    voxel_spacing = np.array(voxel_spacing)

    # 将体素坐标转换为物理坐标（mm）
    transformed_mm = transformed_points * voxel_spacing
    target_mm = target_points * voxel_spacing

    # 计算每个点的欧氏距离 (mm)
    distances = np.linalg.norm(transformed_mm - target_mm, axis=1)

    # 计算平均值和标准差
    tre_mean = np.mean(distances)
    tre_std = np.std(distances)

    return tre_mean, tre_std


# 计算area点的中心点（因为在软件上标注的不是一个确切的像素点）
def calculate_label_centers(filename):
    # 读取NIfTI文件
    img = nib.load(filename)
    data = img.get_fdata()

    # 提取标签
    labels = np.unique(data)

    label_centers = []
    for label in labels:
        if label == 0:
            continue  # 跳过背景标签
        indices = np.where(data == label)
        if label == 2:
            # 分割成左右位置点
            left_points_x = indices[0][:len(indices[0]) // 2]
            right_points_x = indices[0][len(indices[0]) // 2:]
            left_points_y = indices[1][:len(indices[0]) // 2]
            right_points_y = indices[1][len(indices[0]) // 2:]
            points_z = indices[2]

            # 计算左右中心点
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

# 主函数
def main():
    # 加载配置
    config = load_config()
    logger = setup_logger(config)
    
    logger.info("="*50)
    logger.info("开始执行配准程序")
    logger.info("="*50)
    
    # 从配置文件读取参数
    patients_path = config['calibrate']['patients_path']
    filebase = config['calibrate']['filebase']
    save_path = config['calibrate']['save_path']
    tps_reg = config['calibrate']['tps_reg']
    image_shape = tuple(config['calibrate']['image_shape'])
    
    logger.info(f"配准参数:")
    logger.info(f"  患者数据路径: {patients_path}")
    logger.info(f"  基准患者: {filebase}")
    logger.info(f"  保存路径: {save_path}")
    logger.info(f"  TPS正则化参数: {tps_reg}")
    logger.info(f"  图像尺寸: {image_shape}")
    
    # 检查目录是否存在，如果不存在则创建目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"创建保存目录: {save_path}")

    dir_patients = os.listdir(patients_path)
    logger.info(f"共找到 {len(dir_patients)} 个患者数据")
    
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

    logger.info(f"基准患者: {filename_mapping[0]}")
    logger.info(f"待配准患者数量: {len(filenames) - 1}")

    # 创建配准后文件
    after_calibrate = np.zeros(image_shape)

    # 获取基准文件的配准点
    centers_base = calculate_label_centers(filenames[0][0])
    centers_base = np.array(centers_base)

    # 用于统计所有患者的误差
    all_mse_values = []
    all_tre_mean_values = []
    all_tre_std_values = []

    for idx in range(1, len(filenames)):
        logger.info("-"*50)
        logger.info(f"处理患者 {idx}/{len(filenames)-1}: {filename_mapping[idx]}")
        
        # 获取待配准文件的配准点
        centers_calibrate = calculate_label_centers(filenames[idx][0])
        centers_calibrate = np.array(centers_calibrate)
        
        # 计算仿射变换矩阵
        logger.info("  计算TPS变换...")
        f_tps = compute_tps_transform(centers_calibrate, centers_base, reg=tps_reg)
        transform_matrix = f_tps(centers_calibrate)

        # 计算TRE误差
        voxel_spacing = nib.load(filenames[idx][0]).header.get_zooms()[:3]  # (x_spacing, y_spacing, z_spacing)

        # 误差计算
        mse_mm2 = calculate_mse(f_tps(centers_calibrate), centers_base, voxel_spacing)
        tre_mean_mm, tre_std_mm = calculate_tre(f_tps(centers_calibrate), centers_base, voxel_spacing)
        logger.info(f"  MSE (mm²): {mse_mm2:.6f}")
        logger.info(f"  TRE (mm): {tre_mean_mm:.6f} ± {tre_std_mm:.6f}")
        
        # 收集误差数据用于统计
        all_mse_values.append(mse_mm2)
        all_tre_mean_values.append(tre_mean_mm)
        all_tre_std_values.append(tre_std_mm)

        # 应用变换到定位点
        img = nib.load(filenames[idx][1])
        data = img.get_fdata()

        lymphnode_labels = np.unique(data)
        coords = np.where(data != 0)
        coords_h = np.stack([coords[0], coords[1], coords[2]], axis=-1)
        # 应用仿射变换
        transformed_coords_h = f_tps(coords_h)  # 变换后的坐标

        # 将变换后的点转换回常规坐标
        transformed_coords = transformed_coords_h[:, :3]  # 取前三维坐标

        # 这里创建一个新的空数组以存储结果
        transformed_data = np.zeros_like(data)

        # 注意：变换后的坐标可能会超出原始边界，需要处理
        for i in range(transformed_coords.shape[0]):
            x_new, y_new, z_new = transformed_coords[i].astype(int)
            if 0 <= x_new < transformed_data.shape[0] and 0 <= y_new < transformed_data.shape[1] and 0 <= z_new < transformed_data.shape[2]:
                transformed_data[x_new, y_new, z_new] = data[coords_h[i][0].astype(int), coords_h[i][1].astype(int), coords_h[i][2].astype(int)]
        
        new_img = nib.Nifti1Image(transformed_data, img.affine)  # 保留原始的仿射矩阵
        img_name = 'lymphnode_' + filename_mapping[idx] + '.nii.gz'
        img_path = os.path.join(save_path, img_name)
        nib.save(new_img, img_path)
        logger.info(f"  保存配准结果: {img_path}")
    
    logger.info("="*50)
    logger.info("配准程序执行完成!")
    logger.info("="*50)
    
    # 计算并输出平均误差指标
    if len(all_mse_values) > 0:
        avg_mse = np.mean(all_mse_values)
        avg_tre_mean = np.mean(all_tre_mean_values)
        avg_tre_std = np.mean(all_tre_std_values)
        
        logger.info("")
        logger.info("="*50)
        logger.info("所有患者配准误差统计")
        logger.info("="*50)
        logger.info(f"配准患者数量: {len(all_mse_values)}")
        logger.info(f"平均 MSE (mm²): {avg_mse:.6f}")
        logger.info(f"平均 TRE (mm): {avg_tre_mean:.6f} ± {avg_tre_std:.6f}")
        logger.info(f"MSE 范围: [{min(all_mse_values):.6f}, {max(all_mse_values):.6f}]")
        logger.info(f"TRE 范围: [{min(all_tre_mean_values):.6f}, {max(all_tre_mean_values):.6f}]")
        logger.info("="*50)
    else:
        logger.warning("没有配准任何患者数据")

if __name__ == "__main__":
    main()