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
    log_filename = os.path.join(log_dir, f'thermal_map_{timestamp}_{log_file}')
    
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

def printcolormap(grey_image, hot_map, view, filename, config, logger, tmap):
    # 将灰度图像复制为三通道彩色图像
    color_image = np.repeat(grey_image, 3, axis=2).astype(np.uint8)

    # 用于存储热图数据的列表
    data = []
    # 遍历热图数据，收集非零值的坐标和对应值
    non_zero_indices = np.where(hot_map > 0)
    l_indices, m_indices = non_zero_indices[0], non_zero_indices[1]

    # 根据输出显示，nii标签的坐标方向是LPI，而dicom坐标方向是LPS
    for l, m in zip(l_indices, m_indices):
        if view == 'Axial':
            data.append([l, m, hot_map[l, m]])
        else:
            data.append([l, m, hot_map[l, m]])

    image = np.array(color_image)

    if view != 'Axial':
        image = image.transpose((1, 0, 2))

    # 指定色条的最大值和最小值
    max_val = np.max(tmap)
    min_val = 1
    
    # 从配置文件获取参数
    alpha = config['thermal_map']['alpha']
    colormap_name = config['thermal_map']['colormap']
    figure_size = tuple(config['thermal_map']['figure_size'])
    circle_radius = config['thermal_map']['circle_radius']
    img_path = config['thermal_map']['img_path']
    
    # 创建一个新的图形，用于绘制图像和色条等元素
    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    # 在轴上显示原始图像
    ax.imshow(image)

    # 创建归一化对象，将数据中的数值映射到颜色映射的范围（0-1）
    norm = Normalize(vmin=min_val, vmax=max_val)

    # 选择一个颜色映射
    cmap = matplotlib.colormaps[colormap_name]

    # 遍历数据点，在图像上绘制热力点
    for point in data:
        x, y, value = point
        color = cmap(norm(value))[:3]
        circle = plt.Circle((x, y), radius=circle_radius, facecolor=(*color, alpha), edgecolor=None)
        ax.add_patch(circle)

    # 创建色条并添加到图形中
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
    cbar.set_label('Value')

    # 调整布局，让图像和色条显示更合理
    plt.tight_layout()

    # 保存绘制好的图形
    dir_path = img_path + view
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = dir_path + '/' + filename
    plt.savefig(file_name)
    plt.close(fig)

def save_csv(data, filename):
    # 创建DataFrame
    df = pd.DataFrame(data)
    # 转置DataFrame，使得列标题成为第一列的值
    df_transposed = df.T
    # 重置索引，以便列标题成为DataFrame的一部分
def save_csv(data, filename, csv_path):
    # 创建DataFrame
    df = pd.DataFrame(data)
    # 转置DataFrame，使得列标题成为第一列的值
    df_transposed = df.T
    # 重置索引，以便列标题成为DataFrame的一部分
    df_transposed.reset_index(inplace=True)
    # 保存为CSV文件
    df_transposed.to_csv(os.path.join(csv_path, filename + '.csv'), header=False, index=False)

# 主函数
def main():
    # 加载配置
    config = load_config()
    logger = setup_logger(config)
    
    logger.info("="*50)
    logger.info("开始生成热图")
    logger.info("="*50)
    
    # 从配置文件读取参数
    CT_DIR = config['thermal_map']['ct_dir']
    NII_DIR = config['thermal_map']['nii_dir']
    IMG_PATH = config['thermal_map']['img_path']
    CSV_PATH = config['thermal_map']['csv_path']
    SAVE_CSV = config['thermal_map']['save_csv']
    window_width = config['thermal_map']['window_width']
    window_center = config['thermal_map']['window_center']
    image_shape = tuple(config['thermal_map']['image_shape'])
    
    logger.info(f"热图生成参数:")
    logger.info(f"  CT图像目录: {CT_DIR}")
    logger.info(f"  NII文件目录: {NII_DIR}")
    logger.info(f"  输出路径: {IMG_PATH}")
    logger.info(f"  保存CSV: {SAVE_CSV}")
    logger.info(f"  窗口宽度: {window_width}, 窗口中心: {window_center}")
    logger.info(f"  图像尺寸: {image_shape}")
    
    # 检查目录是否存在，如果不存在则创建目录
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
        logger.info(f"创建输出目录: {IMG_PATH}")
    
    if SAVE_CSV and not os.path.exists(CSV_PATH):
        os.makedirs(CSV_PATH)
        logger.info(f"创建CSV目录: {CSV_PATH}")
    
    tmap = np.zeros(image_shape)

    if SAVE_CSV:
        file_position = []
        for i in range(image_shape[0]):
            file_position.append([])
            for j in range(image_shape[1]):
                file_position[i].append([])
                for k in range(image_shape[2]):
                    file_position[i][j].append([])

    file_names = os.listdir(NII_DIR)
    logger.info(f"找到 {len(file_names)} 个NII文件")
    
    for file_name in file_names:
        logger.info(f"处理文件: {file_name}")
        # 使用下划线 '_' 作为分隔符分割字符串
        parts = file_name.split('_')
        # 需要的是分割后的第二部分，还需要去掉扩展名，再次使用 '.' 分割这个部分，并取第一部分
        desired_part = parts[1].split('.')[0]
        logger.info(f"  患者ID: {desired_part}")
        
        file_path = os.path.join(NII_DIR, file_name)
        img = nib.load(file_path)
        data = img.get_fdata()
        logger.info(f"  坐标系统: {nib.aff2axcodes(img.affine)}")
        
        a = np.where(data != 0, 1, 0)
        tmap += a

        if SAVE_CSV:
            pos = np.where(data != 0)
            for i in range(len(pos[0])):
                file_position[pos[0][i]][pos[1][i]][pos[2][i]].append(desired_part)

    max_num = np.max(tmap)
    logger.info(f"热图最大值: {max_num}")

    # 创建一个空的像素数组
    pixel_array = []
    # 获取DICOM系列文件中的所有文件名
    file_names = os.listdir(CT_DIR)
    logger.info(f"找到 {len(file_names)} 个DICOM文件")

    # 遍历DICOM系列文件中的每个文件获得一个三维数组
    dicom_data = np.zeros(image_shape)

    window_min = window_center - window_width / 2.0 + 1000
    window_max = window_center + window_width / 2.0 + 1000

    for i, file_name in enumerate(file_names):
        # 构建完整的文件路径
        file_path = os.path.join(CT_DIR, file_name)
        # 读取DICOM文件
        dicom_dataset = pydicom.dcmread(file_path)

        if i == 1:
            # 提取方向属性
            orientation = dicom_dataset.ImageOrientationPatient
            position = dicom_dataset.ImagePositionPatient
            spacing = dicom_dataset.PixelSpacing
            logger.info(f"DICOM图像信息:")
            logger.info(f"  Image Orientation (Patient): {orientation}")
            logger.info(f"  Image Position (Patient): {position}")
            logger.info(f"  Pixel Spacing: {spacing}")
            
        # 提取像素数据
        pixel_data = dicom_dataset.pixel_array
        # 正则化
        pixel_data = ((pixel_data - window_min) / (window_max - window_min) * 255)
        pixel_data = np.where(pixel_data < 0, 0, pixel_data)
        pixel_data = np.where(pixel_data > 255, 255, pixel_data)
        pixel_data = pixel_data.astype(np.uint8)
        dicom_data[:, :, i] = pixel_data

    logger.info("开始生成冠状面热图...")
    if SAVE_CSV:
        Coronal_data = {}
    for i in range(dicom_data.shape[0]):
        view = 'Coronal'
        pixel_data = dicom_data[i, :, :]
        pixel_data = pixel_data.astype(np.uint8)
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1], 1)
        filename = 'heatmap_' + str(i) + '.png'
        printcolormap(pixel_data, tmap[:, i, :], view, filename, config, logger, tmap)

        if SAVE_CSV:
            pos = np.where(tmap[:, i, :] == max_num)
            if len(pos[0]) == 0:
                continue
            files = set()
            for idx in range(len([pos[0]])):
                for _, x in enumerate(file_position[pos[0][idx]][i][pos[1][idx]]):
                    files.add(x)
            Coronal_data[filename] = list(files)
    if SAVE_CSV:
        save_csv(Coronal_data, 'Coronal_data', CSV_PATH)
    logger.info(f"冠状面热图生成完成，共 {dicom_data.shape[0]} 张")

    logger.info("开始生成矢状面热图...")
    if SAVE_CSV:
        Sagittal_data = {}
    for i in range(dicom_data.shape[1]):
        view = 'Sagittal'
        pixel_data = dicom_data[:, i, :]
        pixel_data = pixel_data.astype(np.uint8)
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1], 1)
        filename = 'heatmap_' + str(i) + '.png'
        printcolormap(pixel_data, tmap[i, :, :], view, filename, config, logger, tmap)

        if SAVE_CSV:
            pos = np.where(tmap[i, :, :] == max_num)
            if len(pos[0]) == 0:
                continue
            files = set()
            for idx in range(len([pos[0]])):
                for _, x in enumerate(file_position[i][pos[0][idx]][pos[1][idx]]):
                    files.add(x)
            Sagittal_data[filename] = list(files)
    if SAVE_CSV:
        save_csv(Sagittal_data, 'Sagittal_data', CSV_PATH)
    logger.info(f"矢状面热图生成完成，共 {dicom_data.shape[1]} 张")

    logger.info("开始生成轴位面热图...")
    if SAVE_CSV:
        Axial_data = {}
    for i in range(dicom_data.shape[2]):
        view = 'Axial'
        pixel_data = dicom_data[:, :, i]
        pixel_data = pixel_data.astype(np.uint8)
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1], 1)
        filename = 'heatmap_' + str(i) + '.png'
        printcolormap(pixel_data, tmap[:, :, i], view, filename, config, logger, tmap)

        if SAVE_CSV:
            pos = np.where(tmap[:, :, i] == max_num)
            if len(pos[0]) == 0:
                continue
            files = set()
            for idx in range(len([pos[0]])):
                for _, x in enumerate(file_position[pos[0][idx]][pos[1][idx]][i]):
                    files.add(x)
            Axial_data[filename] = list(files)
    if SAVE_CSV:
        save_csv(Axial_data, 'Axial_data', CSV_PATH)
    logger.info(f"轴位面热图生成完成，共 {dicom_data.shape[2]} 张")
    
    logger.info("="*50)
    logger.info("热图生成程序执行完成!")
    logger.info("="*50)

if __name__ == "__main__":
    main()