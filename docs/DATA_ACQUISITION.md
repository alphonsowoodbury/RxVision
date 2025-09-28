# NIH RxImage Data Acquisition System

This document describes the data acquisition system for downloading and processing the NIH RxImage dataset for RxVision25.

## Overview

The NIH RxImage dataset contains over 131,000 high-quality images of medications across 4,864+ unique National Drug Codes (NDCs). This dataset is hosted by the National Library of Medicine and is available via FTP.

## Data Acquisition Pipeline

### 1. Dataset Download (`scripts/download_data.py`)

**Features:**
- Downloads images from NIH FTP server (`lhcftp.nlm.nih.gov`)
- Supports both sample and full dataset downloads
- Parallel downloading with progress tracking
- Automatic retry and error handling
- Resume capability for interrupted downloads

**Usage:**
```bash
# Sample dataset (15 classes, ~450 images)
python scripts/download_data.py --sample

# Full dataset (4,864+ classes, 131K+ images)
python scripts/download_data.py --full

# Custom configuration
python scripts/download_data.py --sample --classes 20 --images-per-class 50
python scripts/download_data.py --full --max-classes 100 --target-size 224,224
```

**Command Line Options:**
- `--sample`: Download balanced sample dataset
- `--full`: Download full dataset
- `--classes N`: Number of classes for sample dataset (default: 15)
- `--images-per-class N`: Images per class for sample dataset (default: 30)
- `--max-classes N`: Maximum classes for full dataset
- `--max-images-per-class N`: Maximum images per class
- `--data-dir PATH`: Base directory for dataset (default: 'data')
- `--target-size W,H`: Resize images to target size
- `--cleanup`: Remove raw files after processing
- `--workers N`: Number of download workers (default: 4)

### 2. Image Processing

**Format Conversion:**
- Converts CR2 (Canon RAW) files to JPG
- Standardizes PNG/JPG formats
- Optimizes file sizes while maintaining quality
- Handles corrupted files gracefully

**Supported Formats:**
- Input: CR2, PNG, JPG, JPEG
- Output: JPG (optimized)
- Quality: 95% (configurable)

### 3. Dataset Organization

**Directory Structure:**
```
data/
├── raw/                    # Original downloaded files
├── processed/              # Converted and standardized images
├── train/                  # Training split (70%)
│   ├── [NDC1]/            # Class directories
│   ├── [NDC2]/
│   └── ...
├── val/                    # Validation split (15%)
│   ├── [NDC1]/
│   ├── [NDC2]/
│   └── ...
├── test/                   # Test split (15%)
│   ├── [NDC1]/
│   ├── [NDC2]/
│   └── ...
├── directory_of_images.txt # NIH dataset index
└── dataset_info.json      # Dataset metadata
```

**Split Strategy:**
- **Stratified splitting** by NDC to ensure class balance
- **70/15/15** train/validation/test split (configurable)
- **Class preservation** across all splits

### 4. Metadata Generation

**dataset_info.json contains:**
```json
{
  "dataset_info": {
    "total_images": 450,
    "num_classes": 15,
    "creation_date": "2024-01-15T10:30:00",
    "source": "NIH RxImage Dataset"
  },
  "class_info": {
    "00093725401": {
      "drug_name": "GLIMEPIRIDE 1MG",
      "image_count": 30
    }
  },
  "split_stats": {
    "train": 315,
    "val": 67,
    "test": 68
  },
  "download_stats": {
    "total_files": 450,
    "downloaded": 450,
    "failed": 0,
    "duration": "0:05:23"
  }
}
```

## Testing and Validation

### Connection Test
```bash
python scripts/test_connection.py
```
Tests:
- FTP server connectivity
- Required dependencies
- Access to dataset directories

### Full Pipeline Test
```bash
python scripts/test_data_download.py
```
Tests:
- Complete download pipeline
- Image conversion
- Dataset splitting
- Metadata generation

## Integration with Notebooks

The data acquisition system is integrated into the Jupyter notebooks:

**01_data_exploration_preprocessing.ipynb:**
- Automatically detects if dataset is present
- Prompts user to download if missing
- Provides sample vs. full dataset options
- Loads dataset metadata for analysis

**Usage in Notebook:**
```python
# The notebook will automatically prompt:
# "Choose download option (1 for sample, 2 for full): "

# Or run manually:
import subprocess
subprocess.run([
    "python", "scripts/download_data.py", "--sample"
], cwd="../")
```

## Performance and Optimization

### Download Performance
- **Parallel downloading**: 4 workers by default
- **Resume capability**: Skips existing files
- **Progress tracking**: Real-time progress bars
- **Error handling**: Retries failed downloads

### Expected Times
- **Sample dataset**: 2-5 minutes
- **Full dataset**: 2-6 hours (depending on connection)
- **Processing**: ~10% of download time

### Storage Requirements
- **Sample dataset**: ~50 MB
- **Full dataset**: ~15-20 GB
- **Raw files**: 2x processed (if kept)

## Troubleshooting

### Common Issues

**1. FTP Connection Failed**
```
Error: [Errno 8] nodename nor servname provided
```
- Check internet connection
- Verify firewall settings
- Try different network

**2. Permission Errors**
```
PermissionError: [Errno 13] Permission denied
```
- Check write permissions in data directory
- Run with appropriate user permissions

**3. Disk Space**
```
OSError: [Errno 28] No space left on device
```
- Free up disk space
- Use `--cleanup` option to remove raw files
- Consider sample dataset instead of full

**4. Missing Dependencies**
```
ModuleNotFoundError: No module named 'rawpy'
```
- Install requirements: `pip install -r requirements.txt`
- For minimal setup: `pip install rawpy imageio`

### Manual Recovery

**Resume Interrupted Download:**
```bash
# Downloads will automatically resume from where they stopped
python scripts/download_data.py --sample
```

**Reprocess Existing Files:**
```bash
# Delete processed directory and re-run
rm -rf data/processed data/train data/val data/test
python scripts/download_data.py --sample
```

**Verify Dataset Integrity:**
```bash
python -c "
import json
with open('data/dataset_info.json') as f:
    info = json.load(f)
print(f\"Classes: {info['dataset_info']['num_classes']}\")
print(f\"Images: {info['dataset_info']['total_images']}\")
"
```

## NIH Dataset Information

### Source
- **Provider**: National Library of Medicine (NLM)
- **Host**: lhcftp.nlm.nih.gov
- **Path**: Open-Access-Datasets/Pills/
- **License**: Public Domain
- **Documentation**: https://www.nlm.nih.gov/databases/download/pill_image.html

### Dataset Characteristics
- **Total Images**: 131,271
- **Unique NDCs**: 4,864
- **Image Formats**: CR2, PNG, JPG
- **Resolution**: Variable (typically 1000x1000+)
- **Quality**: Professional pharmaceutical photography

### Legacy vs. Modern Approach

**Legacy (v1) Approach:**
- Manual Google Colab setup
- Basic FTP download with minimal error handling
- Manual file organization
- Limited class selection
- No automated splitting

**Modern (RxVision25) Approach:**
- Automated command-line tool
- Robust error handling and retry logic
- Parallel downloading
- Flexible dataset configuration
- Automated splitting and validation
- Integration with modern ML pipeline

## Future Enhancements

### Planned Features
1. **Cloud Storage Integration**: AWS S3, Google Cloud Storage
2. **Incremental Updates**: Sync with latest NIH updates
3. **Data Validation**: Image quality assessment
4. **Augmentation Preview**: Visual preview of augmentations
5. **Distributed Download**: Multi-machine coordination

### Performance Improvements
1. **HTTP/2 Support**: Faster downloads when available
2. **Compression**: On-the-fly image compression
3. **Caching**: Local caching for repeated downloads
4. **Checksum Validation**: Ensure data integrity

---

**Note**: This data acquisition system is designed for research and educational purposes. Please respect the NIH's terms of use and consider the ethical implications of medication image recognition systems.