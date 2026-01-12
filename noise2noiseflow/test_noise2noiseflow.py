#!/usr/bin/env python3
"""
Noise2NoiseFlow 测试推理脚本
用于评估训练好的模型的去噪效果

使用方法:
    cd /home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow
    python3 test_denoise.py --single_image scene08_frame001.png
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径 - 确保能找到 model、utils 等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# 如果脚本在 noise2noiseflow 目录下
if os.path.basename(current_dir) == 'noise2noiseflow':
    sys.path.insert(0, current_dir)
else:
    # 如果脚本在其他位置，添加 noise2noiseflow 目录
    noise2noiseflow_dir = '/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow'
    sys.path.insert(0, noise2noiseflow_dir)
    os.chdir(noise2noiseflow_dir)

from model.noise2noise_flow import Noise2NoiseFlow


def init_params():
    """初始化模型参数（与训练时保持一致）"""
    npcam = 3
    c_i = 1.0
    beta1_i = -5.0 / c_i
    beta2_i = 0.0
    gain_params_i = np.ndarray([5])
    gain_params_i[:] = -5.0 / c_i
    cam_params_i = np.ndarray([npcam, 5])
    cam_params_i[:, :] = 1.0
    return (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i)


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")
    
    # 创建模型（参数需要与训练时一致）
    model = Noise2NoiseFlow(
        (1, 512, 512),  # 输入形状: (通道数, 高度, 宽度)
        arch='resflow',
        flow_permutation='invconv',
        param_inits=init_params(),
        lu_decomp=True,
        denoiser_model='dncnn',
        dncnn_num_layers=9,
        lmbda=1.0
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理可能的 DataParallel 包装
    state_dict = checkpoint['state_dict']
    # 如果权重是用 DataParallel 保存的，需要去掉 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print("模型加载成功！")
    return model


def calc_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def calc_ssim(img1, img2):
    """计算 SSIM (Structural Similarity Index)"""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def denoise_patch(model, patch, device='cuda'):
    """对单个 patch 进行去噪"""
    # 转换为 tensor: (H, W) -> (1, 1, H, W)
    input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        denoised = model.denoise(input_tensor, clip=True)
    
    return denoised.squeeze().cpu().numpy()


def denoise_full_image(model, noisy_img, patch_size=512, overlap=64, device='cuda'):
    """
    对整张图像进行去噪（使用滑动窗口 + 重叠平均）
    
    参数:
        model: 训练好的模型
        noisy_img: 噪声图像 (H, W)，值域 [0, 1]
        patch_size: patch 尺寸
        overlap: 重叠像素数
        device: 计算设备
    
    返回:
        去噪后的图像
    """
    h, w = noisy_img.shape
    stride = patch_size - overlap
    
    # 创建输出图像和权重图
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    
    # 创建权重掩码（中心权重高，边缘权重低）
    mask = np.ones((patch_size, patch_size), dtype=np.float32)
    if overlap > 0:
        # 使用余弦窗口实现平滑过渡
        ramp = np.linspace(0, 1, overlap)
        mask[:overlap, :] *= ramp[:, np.newaxis]
        mask[-overlap:, :] *= ramp[::-1, np.newaxis]
        mask[:, :overlap] *= ramp[np.newaxis, :]
        mask[:, -overlap:] *= ramp[::-1][np.newaxis, :]
    
    # 滑动窗口处理
    y_positions = list(range(0, h - patch_size + 1, stride))
    if y_positions[-1] + patch_size < h:
        y_positions.append(h - patch_size)
    
    x_positions = list(range(0, w - patch_size + 1, stride))
    if x_positions[-1] + patch_size < w:
        x_positions.append(w - patch_size)
    
    total_patches = len(y_positions) * len(x_positions)
    processed = 0
    
    for y in y_positions:
        for x in x_positions:
            # 提取 patch
            patch = noisy_img[y:y+patch_size, x:x+patch_size]
            
            # 去噪
            denoised_patch = denoise_patch(model, patch, device)
            
            # 累加到输出
            output[y:y+patch_size, x:x+patch_size] += denoised_patch * mask
            weight[y:y+patch_size, x:x+patch_size] += mask
            
            processed += 1
            print(f"\r处理进度: {processed}/{total_patches} patches", end='')
    
    print()
    
    # 归一化
    output = output / (weight + 1e-8)
    output = np.clip(output, 0, 1)
    
    return output


def visualize_results(noisy, denoised, clean, save_path, psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised):
    """可视化去噪结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：完整图像
    axes[0, 0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Noisy\nPSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Denoised\nPSNR: {psnr_denoised:.2f} dB, SSIM: {ssim_denoised:.4f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Clean (Ground Truth)')
    axes[0, 2].axis('off')
    
    # 第二行：差异图和局部放大
    diff_noisy = np.abs(noisy - clean)
    diff_denoised = np.abs(denoised - clean)
    
    axes[1, 0].imshow(diff_noisy, cmap='hot', vmin=0, vmax=0.2)
    axes[1, 0].set_title(f'|Noisy - Clean|\nmean: {diff_noisy.mean():.4f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_denoised, cmap='hot', vmin=0, vmax=0.2)
    axes[1, 1].set_title(f'|Denoised - Clean|\nmean: {diff_denoised.mean():.4f}')
    axes[1, 1].axis('off')
    
    # 改进统计图
    improvement = psnr_denoised - psnr_noisy
    metrics = ['PSNR\n(dB)', 'SSIM']
    noisy_vals = [psnr_noisy, ssim_noisy * 100]  # SSIM 缩放到 0-100
    denoised_vals = [psnr_denoised, ssim_denoised * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[1, 2].bar(x - width/2, noisy_vals, width, label='Noisy', color='salmon')
    bars2 = axes[1, 2].bar(x + width/2, denoised_vals, width, label='Denoised', color='lightgreen')
    
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_title(f'Quality Metrics\nPSNR Improvement: +{improvement:.2f} dB')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {save_path}")


def test_single_image(model, noisy_path, clean_path, output_dir, device='cuda'):
    """测试单张图像"""
    print(f"\n处理图像: {os.path.basename(noisy_path)}")
    
    # 读取图像
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    
    if noisy is None:
        print(f"错误: 无法读取噪声图像 {noisy_path}")
        return None
    if clean is None:
        print(f"错误: 无法读取干净图像 {clean_path}")
        return None
    
    # 归一化到 [0, 1]
    noisy = noisy.astype(np.float32) / 255.0
    clean = clean.astype(np.float32) / 255.0
    
    print(f"图像尺寸: {noisy.shape}")
    
    # 去噪
    print("正在去噪...")
    denoised = denoise_full_image(model, noisy, patch_size=512, overlap=64, device=device)
    
    # 计算指标
    psnr_noisy = calc_psnr(noisy, clean)
    psnr_denoised = calc_psnr(denoised, clean)
    ssim_noisy = calc_ssim(noisy, clean)
    ssim_denoised = calc_ssim(denoised, clean)
    
    print(f"\n评估结果:")
    print(f"  噪声图像 - PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")
    print(f"  去噪图像 - PSNR: {psnr_denoised:.2f} dB, SSIM: {ssim_denoised:.4f}")
    print(f"  PSNR 提升: +{psnr_denoised - psnr_noisy:.2f} dB")
    print(f"  SSIM 提升: +{ssim_denoised - ssim_noisy:.4f}")
    
    # 保存结果
    basename = os.path.splitext(os.path.basename(noisy_path))[0]
    
    # 保存去噪图像
    denoised_save = (denoised * 255).astype(np.uint8)
    denoised_path = os.path.join(output_dir, f"{basename}_denoised.png")
    cv2.imwrite(denoised_path, denoised_save)
    print(f"去噪图像已保存到: {denoised_path}")
    
    # 保存可视化
    vis_path = os.path.join(output_dir, f"{basename}_comparison.png")
    visualize_results(noisy, denoised, clean, vis_path, 
                     psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised)
    
    return {
        'filename': basename,
        'psnr_noisy': psnr_noisy,
        'psnr_denoised': psnr_denoised,
        'ssim_noisy': ssim_noisy,
        'ssim_denoised': ssim_denoised,
        'psnr_improvement': psnr_denoised - psnr_noisy,
        'ssim_improvement': ssim_denoised - ssim_noisy
    }


def test_dataset(model, noisy_dir, clean_dir, output_dir, device='cuda', max_images=None):
    """测试整个数据集"""
    print(f"\n测试数据集:")
    print(f"  噪声图像目录: {noisy_dir}")
    print(f"  干净图像目录: {clean_dir}")
    print(f"  输出目录: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像列表
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
    
    if max_images:
        noisy_files = noisy_files[:max_images]
    
    print(f"  总图像数: {len(noisy_files)}")
    
    results = []
    
    for i, filename in enumerate(noisy_files):
        print(f"\n[{i+1}/{len(noisy_files)}]", end='')
        
        noisy_path = os.path.join(noisy_dir, filename)
        clean_path = os.path.join(clean_dir, filename)
        
        if not os.path.exists(clean_path):
            print(f"  警告: 找不到对应的干净图像 {filename}")
            continue
        
        result = test_single_image(model, noisy_path, clean_path, output_dir, device)
        if result:
            results.append(result)
    
    # 汇总统计
    if results:
        print("\n" + "=" * 60)
        print("汇总统计")
        print("=" * 60)
        
        avg_psnr_noisy = np.mean([r['psnr_noisy'] for r in results])
        avg_psnr_denoised = np.mean([r['psnr_denoised'] for r in results])
        avg_ssim_noisy = np.mean([r['ssim_noisy'] for r in results])
        avg_ssim_denoised = np.mean([r['ssim_denoised'] for r in results])
        avg_psnr_imp = np.mean([r['psnr_improvement'] for r in results])
        avg_ssim_imp = np.mean([r['ssim_improvement'] for r in results])
        
        print(f"测试图像数: {len(results)}")
        print(f"\n平均 PSNR:")
        print(f"  噪声图像: {avg_psnr_noisy:.2f} dB")
        print(f"  去噪图像: {avg_psnr_denoised:.2f} dB")
        print(f"  提升: +{avg_psnr_imp:.2f} dB")
        print(f"\n平均 SSIM:")
        print(f"  噪声图像: {avg_ssim_noisy:.4f}")
        print(f"  去噪图像: {avg_ssim_denoised:.4f}")
        print(f"  提升: +{avg_ssim_imp:.4f}")
        
        # 保存结果到文件
        results_path = os.path.join(output_dir, 'results.txt')
        with open(results_path, 'w') as f:
            f.write("Noise2NoiseFlow 去噪测试结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"测试图像数: {len(results)}\n\n")
            f.write("单张图像结果:\n")
            f.write("-" * 60 + "\n")
            for r in results:
                f.write(f"{r['filename']}:\n")
                f.write(f"  PSNR: {r['psnr_noisy']:.2f} -> {r['psnr_denoised']:.2f} dB (+{r['psnr_improvement']:.2f})\n")
                f.write(f"  SSIM: {r['ssim_noisy']:.4f} -> {r['ssim_denoised']:.4f} (+{r['ssim_improvement']:.4f})\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("汇总统计:\n")
            f.write(f"  平均 PSNR: {avg_psnr_noisy:.2f} -> {avg_psnr_denoised:.2f} dB (+{avg_psnr_imp:.2f})\n")
            f.write(f"  平均 SSIM: {avg_ssim_noisy:.4f} -> {avg_ssim_denoised:.4f} (+{avg_ssim_imp:.4f})\n")
        
        print(f"\n结果已保存到: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Noise2NoiseFlow 测试推理脚本')
    
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow/experiments/paper/grayscale_4crop_full_12scenes_dual_gpu/saved_models/best_model.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--noisy_dir', type=str,
                        default='/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow/data/my_grayscale_dataset/test/noisy',
                        help='噪声图像目录')
    parser.add_argument('--clean_dir', type=str,
                        default='/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow/data/my_grayscale_dataset/test/clean',
                        help='干净图像目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow/experiments/paper/grayscale_4crop_full_12scenes_dual_gpu/test_results',
                        help='输出目录')
    parser.add_argument('--single_image', type=str, default=None,
                        help='测试单张图像（指定文件名，如 scene08_frame001.png）')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最大测试图像数（用于快速测试）')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')
    
    args = parser.parse_args()
    
    # 检查 CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Noise2NoiseFlow 测试推理")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # 加载模型
    model = load_model(args.checkpoint, args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.single_image:
        # 测试单张图像
        noisy_path = os.path.join(args.noisy_dir, args.single_image)
        clean_path = os.path.join(args.clean_dir, args.single_image)
        test_single_image(model, noisy_path, clean_path, args.output_dir, args.device)
    else:
        # 测试整个数据集
        test_dataset(model, args.noisy_dir, args.clean_dir, args.output_dir, 
                    args.device, args.max_images)
    
    print("\n完成！")


if __name__ == '__main__':
    main()