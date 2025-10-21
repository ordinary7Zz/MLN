# MLN项目运行指南

## 环境要求

- Python 3.8
- Conda/Anaconda

## 运行步骤

### 1. 创建虚拟环境

使用conda创建名为`mln`的Python 3.8环境：

```bash
conda create -n mln python=3.8 -y
```

### 2. 激活虚拟环境

```bash
conda activate mln
```

### 3. 安装依赖包

```bash
pip install -r requirements.txt
```

或者逐个安装：

```bash
pip install numpy==1.24.3 scipy==1.10.1 nibabel==5.1.0 pydicom==2.4.3
pip install opencv-python==4.8.1.78 Pillow==10.0.1 matplotlib==3.7.2
pip install pandas==2.0.3 PyYAML==6.0.1 pyheatmap
```

### 4. 配置config.yml

编辑配置文件，设置正确的路径：

```bash
# 如果config.yml不存在，从模板复制
cp config.example.yml config.yml

# 编辑配置文件
vi config.yml
# 或
nano config.yml
```

**需要配置的关键参数：**

```yaml
calibrate:
  patients_path: ./dataset/before_calibrate3D  # 患者数据路径
  filebase: 20230728-WANGCUIRONG           # 基准患者名称
  save_path: ./output/after_calibrate  # 保存路径

thermal_map:
  ct_dir: ./dataset/CT                        # CT图像目录
  nii_dir: ./output/after_calibrate   # 配准后的NII文件目录
  img_path: ./output/Thermal_map  # 热图输出路径

thermal_map_white:
  ct_dir: ./dataset/CT                        # CT图像目录
  nii_dir: ./output/after_calibrate   # 配准后的NII文件目录
  img_path: ./output/whiteBG/Thermal_map_white   # 白色背景热图输出路径
```

### 5. 准备数据

确保数据目录结构正确：

```
MLN/
├── dataset/
│   ├── before_calibrate/      # 患者数据
│   │   ├── 患者1/
│   │   │   ├── *area*.nii      # 标注文件（包含'area'关键字）
│   │   │   └── *.nii           # 淋巴结数据
│   │   ├── 患者2/
│   │   └── ...
│   └── CT/                      # CT DICOM文件
│       ├── 001.dcm
│       ├── 002.dcm
│       └── ...
```

### 6. 运行配准程序

```bash
python Calibrate.py
```

**输出：**
- 配准后的NII文件保存在 `output/after_calibrate/`
- 日志文件保存在 `output/logs/calibrate_*.log`

### 7. 生成热力图

**选项A：生成CT背景热图**

```bash
python Thermal_map.py
```

**选项B：生成白色背景热图**

```bash
python Thermal_map_whiteBG.py
```

**输出：**
- 热图保存在配置文件指定的 `img_path` 目录
- 包含三个视图：Coronal（冠状面）、Sagittal（矢状面）、Axial（轴位面）
- 日志文件保存在 `output/logs/thermal_map_*.log`

## 完整运行示例

```bash
# 1. 创建并激活环境

conda create -n mln python=3.8 -y
conda activate mln

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置（首次运行）
cp config.example.yml config.yml
vi config.yml

# 4. 运行配准
python Calibrate.py

# 5. 生成热图
python Thermal_map.py
# 或
python Thermal_map_whiteBG.py
```

## 查看结果

```bash
# 查看配准结果
ls -lh output/after_calibrate/

# 查看热图
ls -lh output/Thermal_map/Axial/
ls -lh output/Thermal_map/Coronal/
ls -lh output/Thermal_map/Sagittal/

# 查看日志
tail -f /output/logs/*.log
```

## 环境管理

```bash
# 激活环境
conda activate mln

# 停用环境
conda deactivate

# 删除环境
conda env remove -n mln

# 查看已安装的包
conda activate mln
pip list
```


---

**文档版本**: 1.0  
**创建日期**: 2025-10-21  
**Python版本**: 3.8  
**兼容系统**: Linux, macOS, Windows (with conda)
