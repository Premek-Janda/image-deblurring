# Image Deblurring Project

Image deblurring implementation comparing deep learning models with traditional methods on the GoPro dataset.

## 📁 Project Structure

```
image-deblurring/
├── src/                # Core source code (dataset, models, utils, visualize)
├── scripts/            # Executable scripts (train, evaluate)
├── models_trained/     # Trained model checkpoints (.pth)
├── output/             # Generated figures and results
├── docs/               # requirements.txt
└── notebooks/          # Jupyter notebooks
```

## 🚀 Setup

```bash
# Create and activate virtual environment
pip install --user virtualenv
virtualenv -p python3 venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r docs/requirements.txt
```

## 🎯 Dataset

**GoPro Image Deblurring Dataset** - automatically downloaded via kagglehub
- Training: 2,103 pairs (90% train, 10% val)
- Testing: 1,111 pairs

## 🤖 Models

**Deep Learning:**
1. UNet - Encoder-decoder with skip connections
2. DeblurCNN - Multi-layer CNN with residual connections
3. DeblurGANv2 - GAN-based generator
4. DeblurringSimple - Single conv layer baseline

**Traditional Methods:**
- Wiener Deconvolution - Frequency domain
- Richardson-Lucy - Iterative deconvolution
- Stochastic Deconvolution - MCMC-based

## 🏃 Usage

**Train Models:**
Use the Jupyter notebook `notebooks/train-models.ipynb`

**Evaluate Deep Learning (20 images + metrics chart):**
```bash
python scripts/evaluate_dl_models.py
```
Generates metrics chart (PSNR/SSIM) computed over 20 test images.

**Evaluate Traditional Methods (20 images + metrics chart):**
```bash
python scripts/evaluate_traditional_methods.py
```
Generates metrics chart (PSNR/SSIM) computed over 20 test images.

**Single Image Visual Comparison:**
```bash
python scripts/evaluate_dl_models.py --index 5
python scripts/evaluate_traditional_methods.py --index 10
```
Generates side-by-side visual comparison of a single image (blur/deblur/sharp).

**Custom Evaluation:**
```bash
python scripts/evaluate_dl_models.py --num-images 50
python scripts/evaluate_traditional_methods.py --num-images 100
```
Generates metrics chart computed over N images.

**Options:**
- `--index N` - Processes image N and generates **visual comparison** (3 images side-by-side)
- `--num-images N` - Evaluates N images and generates **metrics chart** (average PSNR/SSIM)
- `--save-dir path` - Custom output directory

## 📈 Metrics

- **PSNR** - Peak Signal-to-Noise Ratio (higher is better)
- **SSIM** - Structural Similarity Index (closer to 1 is better)

---
POVa Project - December 2025
