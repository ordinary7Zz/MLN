````markdown
# MLN Project Running Guide

## Environment Requirements

- Python 3.8
- Conda/Anaconda

## Running Steps

### 1. Create Virtual Environment

Create a Python 3.8 environment named `mln` using conda:

```bash
conda create -n mln python=3.8 -y
```

### 2. Activate Virtual Environment

```bash
conda activate mln
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy==1.24.3 scipy==1.10.1 nibabel==5.1.0 pydicom==2.4.3
pip install opencv-python==4.8.1.78 Pillow==10.0.1 matplotlib==3.7.2
pip install pandas==2.0.3 PyYAML==6.0.1 pyheatmap
```

### 4. Configure config.yml

Edit the configuration file to set correct paths:

```bash
# If config.yml doesn't exist, copy from template
cp config.example.yml config.yml

# Edit configuration file
vi config.yml
# or
nano config.yml
```

**Key parameters to configure:**

```yaml
calibrate:
  patients_path: ./dataset/before_calibrate3D  # Patient data path
  filebase: 20230728-WANGCUIRONG           # Reference patient name
  save_path: ./output/after_calibrate  # Save path

thermal_map:
  ct_dir: ./dataset/CT                        # CT image directory
  nii_dir: ./output/after_calibrate   # Registered NII file directory
  img_path: ./output/Thermal_map  # Heatmap output path

thermal_map_white:
  ct_dir: ./dataset/CT                        # CT image directory
  nii_dir: ./output/after_calibrate   # Registered NII file directory
  img_path: ./output/whiteBG/Thermal_map_white   # White background heatmap output path
```

### 5. Prepare Data

Ensure correct data directory structure:

```
MLN/
├── dataset/
│   ├── before_calibrate/      # Patient data
│   │   ├── Patient1/
│   │   │   ├── *area*.nii      # Annotation file (contains 'area' keyword)
│   │   │   └── *.nii           # Lymph node data
│   │   ├── Patient2/
│   │   └── ...
│   └── CT/                      # CT DICOM files
│       ├── 001.dcm
│       ├── 002.dcm
│       └── ...
```

### 6. Run Registration Program

```bash
python Calibrate.py
```

**Output:**
- Registered NII files saved in `output/after_calibrate/`
- Log files saved in `output/logs/calibrate_*.log`

### 7. Generate Heatmaps

**Option A: Generate CT Background Heatmap**

```bash
python Thermal_map.py
```

**Option B: Generate White Background Heatmap**

```bash
python Thermal_map_whiteBG.py
```

**Output:**
- Heatmaps saved in the `img_path` directory specified in configuration file
- Contains three views: Coronal, Sagittal, Axial
- Log files saved in `output/logs/thermal_map_*.log`

## Complete Running Example

```bash
# 1. Create and activate environment
conda create -n mln python=3.8 -y
conda activate mln

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (first run)
cp config.example.yml config.yml
vi config.yml

# 4. Run registration
python Calibrate.py

# 5. Generate heatmaps
python Thermal_map.py
# or
python Thermal_map_whiteBG.py
```

## View Results

```bash
# View registration results
ls -lh output/after_calibrate/

# View heatmaps
ls -lh output/Thermal_map/Axial/
ls -lh output/Thermal_map/Coronal/
ls -lh output/Thermal_map/Sagittal/

# View logs
tail -f /output/logs/*.log
```

## Environment Management

```bash
# Activate environment
conda activate mln

# Deactivate environment
conda deactivate

# Remove environment
conda env remove -n mln

# View installed packages
conda activate mln
pip list
```


---

**Document Version**: 1.0  
**Created Date**: 2025-10-21  
**Python Version**: 3.8  
**Compatible Systems**: Linux, macOS, Windows (with conda)

````
