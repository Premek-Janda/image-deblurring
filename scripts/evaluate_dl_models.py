import os
import sys
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import PeakSignalNoiseRatio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import DEVICE, IMAGE_SIZE, BATCH_SIZE
from src.dataset import GoProDataset
from src.models import DeblurringSimple, UNet, DeblurCNN, DeblurGANv2Generator
from src.visualize import plot_one_image_comparison
from scripts.evaluate_traditional_methods import run_comparison

NUM_IMAGES_FOR_METRICS = 20


def generate_dl_model_figures(image_idx=0, save_dir="output/deep_learning"):
    
    print(f"\n{'='*60}")
    print("GENERATING DEEP LEARNING MODEL FIGURES")
    print(f"{'='*60}")
    
    test_dataset = GoProDataset(split="test", image_size=IMAGE_SIZE, augment=False)
    
    if image_idx >= len(test_dataset):
        print(f"⚠️  Index {image_idx} out of range (max: {len(test_dataset)-1}). Using 0.")
        image_idx = 0
    
    test_dataset = Subset(test_dataset, [image_idx])
    selected_image = test_dataset.dataset.blur_images[test_dataset.indices[0]]
    
    print(f"Image: {selected_image}")
    print(f"Index: {image_idx}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    
    models_dict = {
        "DeblurringSimple": DeblurringSimple,
        "UNet": UNet,
        "DeblurCNN": DeblurCNN,
        "DeblurGANv2": DeblurGANv2Generator
    }
    
    name_mapping = {
        "DeblurringSimple": "deblurring_simple_best.pth",
        "UNet": "unet_best.pth",
        "DeblurCNN": "deblur_cnn_best.pth",
        "DeblurGANv2": "deblurganv2_best.pth"
    }
    
    results = []
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_trained')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PROCESSING MODELS")
    print(f"{'='*60}\n")
    
    for model_name, model_class in models_dict.items():
        model_path = os.path.join(models_dir, name_mapping[model_name])
        
        if not os.path.exists(model_path):
            print(f"⚠️  {model_name} not found")
            continue
        
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        save_path = os.path.join(save_dir, f"{model_name}_visual_comparison.png")
        input_psnr, output_psnr, input_ssim, output_ssim = plot_one_image_comparison(
            model, test_loader, psnr_metric, title=f"{model_name} - Test Result", save_path=save_path
        )
        
        results.append({
            'name': model_name,
            'input_psnr': input_psnr,
            'output_psnr': output_psnr,
            'input_ssim': input_ssim,
            'output_ssim': output_ssim,
            'improvement': output_psnr - input_psnr
        })
        
        print(f"✓ {model_name}")
        plt.close('all')
    
    print(f"\n{'='*60}")
    print("DEEP LEARNING MODELS - PSNR COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} | {'Input PSNR':<12} | {'Output PSNR':<12} | {'Improvement':<12}")
    print("-" * 60)
    for res in results:
        print(f"{res['name']:<20} | {res['input_psnr']:<12.2f} | {res['output_psnr']:<12.2f} | {res['improvement']:<12.2f}")
    
    
    print(f"\n{'='*60}")
    print("SAVING CHARTS")
    print(f"{'='*60}")
    
    names = [r['name'] for r in results]
    output_psnrs = [r['output_psnr'] for r in results]
    output_ssims = [r['output_ssim'] for r in results]
    input_psnr = results[0]['input_psnr']
    input_ssim = results[0]['input_ssim']
    x = range(len(names))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Deep Learning Models - Performance Metrics Comparison', fontsize=16)

    psnr_bars = ax1.bar(x, output_psnrs, color='skyblue', label='Output PSNR')
    ax1.axhline(y=input_psnr, color='r', linestyle='--', label=f'Blurred Input ({input_psnr:.2f})')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Peak Signal-to-Noise Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend()
    ax1.bar_label(psnr_bars, padding=3, fmt='%.2f')

    ssim_bars = ax2.bar(x, output_ssims, color='salmon', label='Output SSIM')
    ax2.axhline(y=input_ssim, color='r', linestyle='--', label=f'Blurred Input ({input_ssim:.4f})')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity Index')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.legend()
    ax2.bar_label(ssim_bars, padding=3, fmt='%.4f')
    ax2.set_ylim([min(input_ssim, min(output_ssims)) - 0.05, 1.0])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'dl_models_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart")
    plt.close()



def generate_traditional_methods_figures(image_idx=0, save_dir="output/report_figures"):
    """Generate figures for traditional deblurring methods."""
    
    print(f"\n{'='*60}")
    print("GENERATING TRADITIONAL METHODS FIGURES")
    print(f"{'='*60}")
    
    run_comparison(image_idx=image_idx, save_dir=save_dir)



def evaluate_metrics_on_multiple_images(num_images=20, save_dir="output/deep_learning"):
    """Evaluate all models on multiple test images."""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING METRICS ON {num_images} IMAGES")
    print(f"{'='*60}")
    
    test_dataset = GoProDataset(split="test", image_size=IMAGE_SIZE, augment=False)
    num_images = min(num_images, len(test_dataset))
    
    random_indices = random.sample(range(len(test_dataset)), num_images)
    test_subset = Subset(test_dataset, random_indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    
    models_dict = {
        "DeblurringSimple": DeblurringSimple,
        "UNet": UNet,
        "DeblurCNN": DeblurCNN,
        "DeblurGANv2": DeblurGANv2Generator
    }
    
    name_mapping = {
        "DeblurringSimple": "deblurring_simple_best.pth",
        "UNet": "unet_best.pth",
        "DeblurCNN": "deblur_cnn_best.pth",
        "DeblurGANv2": "deblurganv2_best.pth"
    }
    
    results = {}
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_trained')
    
    for model_name, model_class in models_dict.items():
        model_path = os.path.join(models_dir, name_mapping[model_name])
        
        if not os.path.exists(model_path):
            continue
        
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        input_psnr_sum = output_psnr_sum = input_ssim_sum = output_ssim_sum = count = 0
        
        with torch.no_grad():
            for blur_batch, sharp_batch in test_loader:
                blur_batch, sharp_batch = blur_batch.to(DEVICE), sharp_batch.to(DEVICE)
                deblurred_batch = model(blur_batch)
                
                for i in range(blur_batch.size(0)):
                    from skimage.metrics import structural_similarity
                    import numpy as np
                    
                    input_psnr_sum += psnr_metric(blur_batch[i:i+1], sharp_batch[i:i+1]).item()
                    output_psnr_sum += psnr_metric(deblurred_batch[i:i+1], sharp_batch[i:i+1]).item()
                    
                    blur_np = blur_batch[i].cpu().numpy().transpose(1, 2, 0)
                    sharp_np = sharp_batch[i].cpu().numpy().transpose(1, 2, 0)
                    deblur_np = deblurred_batch[i].cpu().numpy().transpose(1, 2, 0)
                    
                    input_ssim_sum += structural_similarity(blur_np, sharp_np, data_range=1.0, channel_axis=2)
                    output_ssim_sum += structural_similarity(deblur_np, sharp_np, data_range=1.0, channel_axis=2)
                    count += 1
        
        results[model_name] = {
            'avg_input_psnr': input_psnr_sum / count,
            'avg_output_psnr': output_psnr_sum / count,
            'avg_input_ssim': input_ssim_sum / count,
            'avg_output_ssim': output_ssim_sum / count,
            'avg_improvement': (output_psnr_sum - input_psnr_sum) / count,
            'num_images': count
        }
    
    print(f"\n{'='*80}")
    print(f"METRICS EVALUATION RESULTS ({num_images} images)")
    print(f"{'='*80}")
    print(f"{'Model':<20} | {'Avg Input PSNR':<15} | {'Avg Output PSNR':<15} | {'Avg Improvement':<15}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | {metrics['avg_input_psnr']:>13.2f} dB | "
              f"{metrics['avg_output_psnr']:>13.2f} dB | {metrics['avg_improvement']:>13.2f} dB")
    
    print("="*80)
    
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['avg_output_psnr'])
        print(f"\n🏆 Best: {best_model[0]} ({best_model[1]['avg_output_psnr']:.2f} dB)")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAVING METRICS CHART")
    print(f"{'='*60}")
    
    model_names = list(results.keys())
    avg_psnrs = [results[name]['avg_output_psnr'] for name in model_names]
    avg_ssims = [results[name]['avg_output_ssim'] for name in model_names]
    avg_input_psnr = results[model_names[0]]['avg_input_psnr']
    avg_input_ssim = results[model_names[0]]['avg_input_ssim']
    x = range(len(model_names))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Deep Learning Models - Performance Metrics Comparison', fontsize=16)

    psnr_bars = ax1.bar(x, avg_psnrs, color='skyblue', label='Avg Output PSNR')
    ax1.axhline(y=avg_input_psnr, color='r', linestyle='--', label=f'Blurred Input ({avg_input_psnr:.2f})')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Peak Signal-to-Noise Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha="right")
    ax1.legend()
    ax1.bar_label(psnr_bars, padding=3, fmt='%.2f')

    ssim_bars = ax2.bar(x, avg_ssims, color='salmon', label='Avg Output SSIM')
    ax2.axhline(y=avg_input_ssim, color='r', linestyle='--', label=f'Blurred Input ({avg_input_ssim:.4f})')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity Index')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha="right")
    ax2.legend()
    ax2.bar_label(ssim_bars, padding=3, fmt='%.4f')
    ax2.set_ylim([min(avg_input_ssim, min(avg_ssims)) - 0.05, 1.0])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'dl_models_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart saved")
    plt.close()
    
    return results



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate deep learning model figures and metrics')
    parser.add_argument('--index', type=int, default=None, help='Dataset image index for single image visual comparison')
    parser.add_argument('--save-dir', type=str, default='output/deep_learning', help='Output directory (default: output/deep_learning/)')
    parser.add_argument('--num-images', type=int, default=NUM_IMAGES_FOR_METRICS, 
                        help=f'Number of images for metrics evaluation (default: {NUM_IMAGES_FOR_METRICS})')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("GENERATING DEEP LEARNING FIGURES")
    print(f"{'='*60}")
    print(f"Output: {args.save_dir}")
    print(f"{'='*60}")
    
    if args.index is not None:
        generate_dl_model_figures(image_idx=args.index, save_dir=args.save_dir)
    else:
        evaluate_metrics_on_multiple_images(num_images=args.num_images, save_dir=args.save_dir)
    
    print(f"\n{'='*60}")
    print("✅ ALL FIGURES GENERATED")
    print(f"{'='*60}\n")

