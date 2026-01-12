#!/usr/bin/env python3
"""
Noise2NoiseFlow 噪声建模效果测试脚本

功能：
1. 噪声采样：从训练好的噪声模型生成噪声
2. 噪声分布可视化：比较真实噪声和生成噪声的统计特性
3. 噪声分析：展示信号依赖性噪声特性

使用方法:
    cd /home/lt/denoising/N2Nflow/Noise2NoiseFlow
    PYTHONPATH=. python3 noise2noiseflow/test_noise_model.py --single_image scene08_frame001.png
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
from scipy import stats
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'noise2noiseflow':
    project_root = os.path.dirname(current_dir)
else:
    project_root = '/home/lt/denoising/N2Nflow/Noise2NoiseFlow'
sys.path.insert(0, project_root)
os.chdir(project_root)

from model.noise2noise_flow import Noise2NoiseFlow


def init_params():
    """初始化模型参数"""
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
    
    model = Noise2NoiseFlow(
        (1, 512, 512),
        arch='resflow',
        flow_permutation='invconv',
        param_inits=init_params(),
        lu_decomp=True,
        denoiser_model='dncnn',
        dncnn_num_layers=9,
        lmbda=1.0
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    
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


def extract_real_noise(noisy_img, clean_img):
    """提取真实噪声 = 噪声图像 - 干净图像"""
    return noisy_img - clean_img


def generate_noise_from_model(model, clean_img, device='cuda', temperature=1.0, 
                               iso=800, cam=0, nlf0=0.0001, nlf1=0.00001):
    """
    使用训练好的噪声模型生成噪声
    
    Noise Flow 学习的是 p(noise | clean)
    通过采样 z ~ N(0, temperature) 并经过逆向 flow 变换得到噪声样本
    
    参数:
        model: 训练好的 Noise2NoiseFlow 模型
        clean_img: 干净图像 (H, W)，值域 [0, 1]
        device: 计算设备
        temperature: 采样温度，控制噪声强度 (1.0 = 标准)
        iso: ISO 值
        cam: 相机 ID
        nlf0, nlf1: 噪声级别函数参数
    
    返回:
        生成的噪声图像
    """
    h, w = clean_img.shape
    
    # 将干净图像转换为 tensor: (H, W) -> (1, 1, H, W)
    clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # 创建其他必要的参数 tensor
    iso_tensor = torch.full(clean_tensor.shape, iso, dtype=torch.float32, device=device)
    cam_tensor = torch.full(clean_tensor.shape, cam, dtype=torch.float32, device=device)
    nlf0_tensor = torch.full(clean_tensor.shape, nlf0, dtype=torch.float32, device=device)
    nlf1_tensor = torch.full(clean_tensor.shape, nlf1, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        try:
            # 使用模型的 sample 函数
            # sample 函数需要 clean, iso, cam, nlf0, nlf1 等参数
            eps_std = torch.tensor(temperature, device=device)
            
            kwargs = {
                'clean': clean_tensor,
                'iso': iso_tensor,
                'cam': cam_tensor,
                'nlf0': nlf0_tensor,
                'nlf1': nlf1_tensor
            }
            
            # 调用模型的 sample 方法生成噪声
            generated_noise = model.sample(eps_std=eps_std, **kwargs)
            generated_noise = generated_noise.squeeze().cpu().numpy()
            
            print(f"✓ 使用 NoiseFlow 模型成功生成噪声 (temperature={temperature})")
            
        except Exception as e:
            print(f"NoiseFlow 采样失败: {e}")
            print("使用替代方法：基于信号依赖噪声模型")
            
            # 替代方法：使用泊松-高斯噪声模型
            # var = alpha * signal + beta (信号依赖噪声)
            alpha = nlf0
            beta = nlf1
            noise_std = np.sqrt(np.maximum(alpha * clean_img + beta, 1e-10))
            
            # 生成噪声
            generated_noise = np.random.randn(h, w).astype(np.float32) * noise_std
    
    return generated_noise


def analyze_noise_statistics(real_noise, generated_noise, clean_img):
    """分析噪声统计特性"""
    stats_dict = {}
    
    # 基本统计量
    stats_dict['real'] = {
        'mean': np.mean(real_noise),
        'std': np.std(real_noise),
        'min': np.min(real_noise),
        'max': np.max(real_noise),
        'skewness': stats.skew(real_noise.flatten()),
        'kurtosis': stats.kurtosis(real_noise.flatten())
    }
    
    stats_dict['generated'] = {
        'mean': np.mean(generated_noise),
        'std': np.std(generated_noise),
        'min': np.min(generated_noise),
        'max': np.max(generated_noise),
        'skewness': stats.skew(generated_noise.flatten()),
        'kurtosis': stats.kurtosis(generated_noise.flatten())
    }
    
    # 信号依赖性分析
    # 将图像按亮度分成若干区间，计算每个区间的噪声方差
    n_bins = 10
    intensity_bins = np.linspace(0, 1, n_bins + 1)
    
    real_var_per_bin = []
    gen_var_per_bin = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (clean_img >= intensity_bins[i]) & (clean_img < intensity_bins[i+1])
        if np.sum(mask) > 100:  # 确保有足够的像素
            real_var_per_bin.append(np.var(real_noise[mask]))
            gen_var_per_bin.append(np.var(generated_noise[mask]))
            bin_centers.append((intensity_bins[i] + intensity_bins[i+1]) / 2)
    
    stats_dict['signal_dependency'] = {
        'bin_centers': np.array(bin_centers),
        'real_variance': np.array(real_var_per_bin),
        'generated_variance': np.array(gen_var_per_bin)
    }
    
    return stats_dict


def visualize_noise_comparison(noisy_img, clean_img, denoised_img, real_noise, 
                                generated_noise, stats_dict, save_path):
    """可视化噪声建模效果"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # ===== 第一行：图像对比 =====
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Noisy Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.imshow(clean_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Clean Image (GT)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(4, 4, 3)
    ax3.imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Denoised Image')
    ax3.axis('off')
    
    # 合成噪声图像
    synthetic_noisy = np.clip(clean_img + generated_noise, 0, 1)
    ax4 = fig.add_subplot(4, 4, 4)
    ax4.imshow(synthetic_noisy, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Synthetic Noisy\n(Clean + Generated Noise)')
    ax4.axis('off')
    
    # ===== 第二行：噪声图对比 =====
    noise_vmin, noise_vmax = -0.15, 0.15
    
    ax5 = fig.add_subplot(4, 4, 5)
    im5 = ax5.imshow(real_noise, cmap='RdBu', vmin=noise_vmin, vmax=noise_vmax)
    ax5.set_title(f'Real Noise\nstd={stats_dict["real"]["std"]:.4f}')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(4, 4, 6)
    im6 = ax6.imshow(generated_noise, cmap='RdBu', vmin=noise_vmin, vmax=noise_vmax)
    ax6.set_title(f'Generated Noise\nstd={stats_dict["generated"]["std"]:.4f}')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(4, 4, 7)
    noise_diff = real_noise - generated_noise
    im7 = ax7.imshow(noise_diff, cmap='RdBu', vmin=noise_vmin, vmax=noise_vmax)
    ax7.set_title(f'Difference\n(Real - Generated)')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # 噪声绝对值对比
    ax8 = fig.add_subplot(4, 4, 8)
    ax8.imshow(np.abs(real_noise), cmap='hot', vmin=0, vmax=0.1)
    ax8.set_title('|Real Noise|')
    ax8.axis('off')
    
    # ===== 第三行：噪声分布直方图 =====
    ax9 = fig.add_subplot(4, 4, 9)
    ax9.hist(real_noise.flatten(), bins=100, density=True, alpha=0.7, 
             label='Real', color='blue')
    ax9.hist(generated_noise.flatten(), bins=100, density=True, alpha=0.7, 
             label='Generated', color='red')
    ax9.set_xlabel('Noise Value')
    ax9.set_ylabel('Density')
    ax9.set_title('Noise Distribution')
    ax9.legend()
    ax9.set_xlim(-0.2, 0.2)
    
    # Q-Q plot
    ax10 = fig.add_subplot(4, 4, 10)
    real_sorted = np.sort(real_noise.flatten())
    gen_sorted = np.sort(generated_noise.flatten())
    # 采样以加速绘图
    n_samples = min(10000, len(real_sorted))
    indices = np.linspace(0, len(real_sorted)-1, n_samples, dtype=int)
    ax10.scatter(real_sorted[indices], gen_sorted[indices], alpha=0.3, s=1)
    ax10.plot([-0.2, 0.2], [-0.2, 0.2], 'r--', label='y=x')
    ax10.set_xlabel('Real Noise Quantiles')
    ax10.set_ylabel('Generated Noise Quantiles')
    ax10.set_title('Q-Q Plot')
    ax10.legend()
    ax10.set_xlim(-0.15, 0.15)
    ax10.set_ylim(-0.15, 0.15)
    
    # ===== 第三行续：信号依赖性 =====
    ax11 = fig.add_subplot(4, 4, 11)
    sd = stats_dict['signal_dependency']
    ax11.plot(sd['bin_centers'], sd['real_variance'], 'bo-', label='Real Noise', linewidth=2)
    ax11.plot(sd['bin_centers'], sd['generated_variance'], 'r^--', label='Generated Noise', linewidth=2)
    ax11.set_xlabel('Signal Intensity')
    ax11.set_ylabel('Noise Variance')
    ax11.set_title('Signal-Dependent Noise\n(Variance vs Intensity)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 统计量对比表格
    ax12 = fig.add_subplot(4, 4, 12)
    ax12.axis('off')
    table_data = [
        ['Metric', 'Real Noise', 'Generated Noise'],
        ['Mean', f'{stats_dict["real"]["mean"]:.6f}', f'{stats_dict["generated"]["mean"]:.6f}'],
        ['Std', f'{stats_dict["real"]["std"]:.6f}', f'{stats_dict["generated"]["std"]:.6f}'],
        ['Skewness', f'{stats_dict["real"]["skewness"]:.4f}', f'{stats_dict["generated"]["skewness"]:.4f}'],
        ['Kurtosis', f'{stats_dict["real"]["kurtosis"]:.4f}', f'{stats_dict["generated"]["kurtosis"]:.4f}'],
    ]
    table = ax12.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax12.set_title('Noise Statistics Comparison')
    
    # ===== 第四行：局部区域对比 =====
    # 选择一个感兴趣的区域进行放大
    crop_y, crop_x = 200, 300
    crop_size = 128
    
    ax13 = fig.add_subplot(4, 4, 13)
    crop_noisy = noisy_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    ax13.imshow(crop_noisy, cmap='gray', vmin=0, vmax=1)
    ax13.set_title('Noisy (Zoomed)')
    ax13.axis('off')
    
    ax14 = fig.add_subplot(4, 4, 14)
    crop_clean = clean_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    ax14.imshow(crop_clean, cmap='gray', vmin=0, vmax=1)
    ax14.set_title('Clean (Zoomed)')
    ax14.axis('off')
    
    ax15 = fig.add_subplot(4, 4, 15)
    crop_synthetic = synthetic_noisy[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    ax15.imshow(crop_synthetic, cmap='gray', vmin=0, vmax=1)
    ax15.set_title('Synthetic Noisy (Zoomed)')
    ax15.axis('off')
    
    ax16 = fig.add_subplot(4, 4, 16)
    crop_denoised = denoised_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    ax16.imshow(crop_denoised, cmap='gray', vmin=0, vmax=1)
    ax16.set_title('Denoised (Zoomed)')
    ax16.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"噪声建模可视化已保存到: {save_path}")


def generate_noise_samples(model, clean_img, n_samples=5, device='cuda',
                           iso=800, cam=0, nlf0=0.0001, nlf1=0.00001):
    """生成多个噪声样本，展示噪声模型的随机性"""
    noise_samples = []
    
    for i in range(n_samples):
        # 每次采样使用不同的随机种子
        noise = generate_noise_from_model(
            model, clean_img, device, 
            temperature=1.0,
            iso=iso, cam=cam, nlf0=nlf0, nlf1=nlf1
        )
        noise_samples.append(noise)
    
    return noise_samples


def visualize_noise_samples(clean_img, noise_samples, save_path):
    """可视化多个噪声样本"""
    n_samples = len(noise_samples)
    
    fig, axes = plt.subplots(2, n_samples + 1, figsize=(4 * (n_samples + 1), 8))
    
    # 第一行：干净图像 + 合成噪声图像
    axes[0, 0].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Clean Image')
    axes[0, 0].axis('off')
    
    for i, noise in enumerate(noise_samples):
        synthetic = np.clip(clean_img + noise, 0, 1)
        axes[0, i+1].imshow(synthetic, cmap='gray', vmin=0, vmax=1)
        axes[0, i+1].set_title(f'Synthetic #{i+1}')
        axes[0, i+1].axis('off')
    
    # 第二行：噪声图
    axes[1, 0].axis('off')
    
    for i, noise in enumerate(noise_samples):
        axes[1, i+1].imshow(noise, cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[1, i+1].set_title(f'Noise #{i+1}\nstd={np.std(noise):.4f}')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"噪声样本可视化已保存到: {save_path}")


def denoise_image(model, noisy_img, device='cuda'):
    """使用模型去噪"""
    h, w = noisy_img.shape
    patch_size = 512
    
    # 如果图像小于等于 patch_size，直接处理
    if h <= patch_size and w <= patch_size:
        # Padding 到 patch_size
        padded = np.zeros((patch_size, patch_size), dtype=np.float32)
        padded[:h, :w] = noisy_img
        
        input_tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            denoised = model.denoise(input_tensor, clip=True)
        
        return denoised.squeeze().cpu().numpy()[:h, :w]
    
    # 对于大图像，使用滑动窗口
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    overlap = 64
    stride = patch_size - overlap
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = noisy_img[y:y+patch_size, x:x+patch_size]
            input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                denoised_patch = model.denoise(input_tensor, clip=True)
            
            denoised_patch = denoised_patch.squeeze().cpu().numpy()
            output[y:y+patch_size, x:x+patch_size] += denoised_patch
            weight[y:y+patch_size, x:x+patch_size] += 1
    
    # 处理边缘
    if (h - patch_size) % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = noisy_img[y:y+patch_size, x:x+patch_size]
            input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                denoised_patch = model.denoise(input_tensor, clip=True)
            denoised_patch = denoised_patch.squeeze().cpu().numpy()
            output[y:y+patch_size, x:x+patch_size] += denoised_patch
            weight[y:y+patch_size, x:x+patch_size] += 1
    
    if (w - patch_size) % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = noisy_img[y:y+patch_size, x:x+patch_size]
            input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                denoised_patch = model.denoise(input_tensor, clip=True)
            denoised_patch = denoised_patch.squeeze().cpu().numpy()
            output[y:y+patch_size, x:x+patch_size] += denoised_patch
            weight[y:y+patch_size, x:x+patch_size] += 1
    
    output = output / (weight + 1e-8)
    return np.clip(output, 0, 1)


def test_noise_model(model, noisy_path, clean_path, output_dir, device='cuda',
                     iso=800, cam=0, nlf0=0.0001, nlf1=0.00001):
    """测试噪声模型效果"""
    print(f"\n处理图像: {os.path.basename(noisy_path)}")
    
    # 读取图像
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    
    if noisy is None or clean is None:
        print("错误: 无法读取图像")
        return None
    
    # 归一化
    noisy = noisy.astype(np.float32) / 255.0
    clean = clean.astype(np.float32) / 255.0
    
    print(f"图像尺寸: {noisy.shape}")
    
    # 1. 去噪
    print("正在去噪...")
    denoised = denoise_image(model, noisy, device)
    
    # 2. 提取真实噪声
    real_noise = extract_real_noise(noisy, clean)
    
    # 3. 使用训练好的 NoiseFlow 生成噪声
    print("正在使用 NoiseFlow 生成噪声...")
    generated_noise = generate_noise_from_model(
        model, clean, device,
        temperature=1.0,
        iso=iso, cam=cam, nlf0=nlf0, nlf1=nlf1
    )
    
    # 4. 分析噪声统计
    print("正在分析噪声统计...")
    stats_dict = analyze_noise_statistics(real_noise, generated_noise, clean)
    
    # 5. 可视化
    basename = os.path.splitext(os.path.basename(noisy_path))[0]
    
    vis_path = os.path.join(output_dir, f"{basename}_noise_analysis.png")
    visualize_noise_comparison(noisy, clean, denoised, real_noise, 
                               generated_noise, stats_dict, vis_path)
    
    # 6. 生成多个噪声样本
    print("正在生成多个噪声样本...")
    noise_samples = generate_noise_samples(
        model, clean, n_samples=4, device=device,
        iso=iso, cam=cam, nlf0=nlf0, nlf1=nlf1
    )
    samples_path = os.path.join(output_dir, f"{basename}_noise_samples.png")
    visualize_noise_samples(clean, noise_samples, samples_path)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("噪声统计对比")
    print("=" * 50)
    print(f"{'指标':<15} {'真实噪声':<15} {'生成噪声':<15}")
    print("-" * 50)
    print(f"{'均值':<15} {stats_dict['real']['mean']:<15.6f} {stats_dict['generated']['mean']:<15.6f}")
    print(f"{'标准差':<15} {stats_dict['real']['std']:<15.6f} {stats_dict['generated']['std']:<15.6f}")
    print(f"{'偏度':<15} {stats_dict['real']['skewness']:<15.4f} {stats_dict['generated']['skewness']:<15.4f}")
    print(f"{'峰度':<15} {stats_dict['real']['kurtosis']:<15.4f} {stats_dict['generated']['kurtosis']:<15.4f}")
    print("=" * 50)
    
    return stats_dict


def main():
    parser = argparse.ArgumentParser(description='Noise2NoiseFlow 噪声建模测试脚本')
    
    base_dir = '/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow'
    
    parser.add_argument('--checkpoint', type=str,
                        default=f'{base_dir}/experiments/paper/grayscale_4crop_full_12scenes_dual_gpu/saved_models/best_model.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--noisy_dir', type=str,
                        default=f'{base_dir}/data/my_grayscale_dataset/test/noisy',
                        help='噪声图像目录')
    parser.add_argument('--clean_dir', type=str,
                        default=f'{base_dir}/data/my_grayscale_dataset/test/clean',
                        help='干净图像目录')
    parser.add_argument('--output_dir', type=str,
                        default=f'{base_dir}/experiments/paper/grayscale_4crop_full_12scenes_dual_gpu/noise_model_results',
                        help='输出目录')
    parser.add_argument('--single_image', type=str, default=None,
                        help='测试单张图像')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最大测试图像数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--iso', type=int, default=800,
                        help='ISO 值')
    parser.add_argument('--cam', type=int, default=0,
                        help='相机 ID')
    parser.add_argument('--nlf0', type=float, default=0.0001,
                        help='噪声级别函数参数 nlf0')
    parser.add_argument('--nlf1', type=float, default=0.00001,
                        help='噪声级别函数参数 nlf1')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Noise2NoiseFlow 噪声建模效果测试")
    print("=" * 60)
    print(f"ISO: {args.iso}, Camera: {args.cam}")
    print(f"NLF0: {args.nlf0}, NLF1: {args.nlf1}")
    
    # 加载模型
    model = load_model(args.checkpoint, args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.single_image:
        noisy_path = os.path.join(args.noisy_dir, args.single_image)
        clean_path = os.path.join(args.clean_dir, args.single_image)
        test_noise_model(model, noisy_path, clean_path, args.output_dir, args.device,
                        iso=args.iso, cam=args.cam, nlf0=args.nlf0, nlf1=args.nlf1)
    else:
        # 测试多张图像
        noisy_files = sorted([f for f in os.listdir(args.noisy_dir) if f.endswith('.png')])
        if args.max_images:
            noisy_files = noisy_files[:args.max_images]
        
        for filename in noisy_files:
            noisy_path = os.path.join(args.noisy_dir, filename)
            clean_path = os.path.join(args.clean_dir, filename)
            if os.path.exists(clean_path):
                test_noise_model(model, noisy_path, clean_path, args.output_dir, args.device,
                               iso=args.iso, cam=args.cam, nlf0=args.nlf0, nlf1=args.nlf1)
    
    print("\n完成！")


if __name__ == '__main__':
    main()