import torch
import numpy as np
from torch.utils.data import IterableDataset
import cv2
import os
import glob
from pathlib import Path
import re

class CustomGrayscaleDataset(IterableDataset):
    """
    针对连续抓拍场景的灰度图数据集加载器
    支持：
    - 四区域裁剪（所有区域都可用，已去除亮线）
    - 不同场景的帧数不同（150帧或50帧）
    
    数据集目录结构：
    your_dataset/
    ├── train/
    │   └── noisy/
    │       ├── scene01_frame001.png  (150帧)
    │       ├── ...
    │       ├── scene09_frame001.png  (50帧)
    │       └── ...
    └── test/
        ├── noisy/
        └── clean/  (可选)
    
    四区域裁剪说明：
    原图已去除亮线区域，尺寸为 1280×1024
    裁剪为4个512×512区域，全部可用：
    
         0        512       1024
    0    ┌─────────┬─────────┐
         │ 区域0   │ 区域1   │
         │ (0,0)   │(0,512)  │
         │ ✓可用   │ ✓可用   │
    512  ├─────────┼─────────┤
         │ 区域2   │ 区域3   │
         │(512,0)  │(512,512)│
         │ ✓可用   │ ✓可用   │
    1024 └─────────┴─────────┘
    
    坐标说明（OpenCV格式 img[y:y+h, x:x+w]）：
    - 区域0: img[0:512, 0:512]
    - 区域1: img[0:512, 512:1024]
    - 区域2: img[512:1024, 0:512]
    - 区域3: img[512:1024, 512:1024]
    """
    
    def __init__(self, 
                 dataset_path, 
                 train_or_test='train',
                 num_regions=4,             # 使用4个区域
                 patch_size=(512, 512),     # 裁剪区域大小
                 original_size=(1280, 1024),# 原始图像尺寸（已去除亮线）
                 patch_sampling='uniform',
                 shuffle_patches=False,
                 subtract_images=False,
                 transform=None,
                 iso=100,
                 cam=0,
                 nlf0=0.0001,
                 nlf1=0.00001,
                 scene_frame_counts=None,   # 字典：{'01': 150, '09': 50, ...}
                 default_frames=150,        # 默认帧数
                 scene_based_pairing=True,
                 pair_distance=1):
        
        self.dataset_path = dataset_path
        self.train_or_test = train_or_test
        self.num_regions = num_regions
        self.patch_size = patch_size
        self.original_size = original_size
        self.shuffle_patches = shuffle_patches
        self.patch_sampling = patch_sampling
        self.subtract_images = subtract_images
        self.transform = transform
        self.scene_based_pairing = scene_based_pairing
        self.pair_distance = pair_distance
        self.default_frames = default_frames
        
        # 场景帧数配置
        self.scene_frame_counts = scene_frame_counts or {}
        
        # 定义4个裁剪区域的坐标 (y_start, x_start)
        # 注意：OpenCV中是 img[y, x]，所以先y后x
        self.crop_regions = [
            (0, 0),        # 区域0: 左上
            (0, 512),      # 区域1: 右上
            (512, 0),      # 区域2: 左下
            (512, 512)     # 区域3: 右下
        ]
        
        # 元数据
        self.default_iso = iso
        self.default_cam = cam
        self.default_nlf0 = nlf0
        self.default_nlf1 = nlf1
        
        # 加载图像文件列表并按场景分组
        self.noisy_files = self._load_file_list('noisy')
        self.clean_files = self._load_file_list('clean')
        
        # 按场景分组
        self.scene_groups = self._group_by_scene(self.noisy_files)
        
        # 计算总样本数（图像数 × 区域数）
        self.len = len(self.noisy_files) * self.num_regions
        
        self._print_dataset_info()
        
    def _print_dataset_info(self):
        """打印数据集信息"""
        print("=" * 70)
        print(f"数据集配置 ({self.train_or_test.upper()})")
        print("=" * 70)
        print(f"总图像数: {len(self.noisy_files)}")
        print(f"场景数: {len(self.scene_groups)}")
        print(f"裁剪区域数: {self.num_regions}")
        print(f"总样本数: {self.len} (图像数 × 区域数)")
        print(f"原始尺寸: {self.original_size[0]}×{self.original_size[1]}")
        print(f"裁剪尺寸: {self.patch_size[0]}×{self.patch_size[1]}")
        print(f"\n裁剪区域坐标 (y, x):")
        region_names = ["左上", "右上", "左下", "右下"]
        for idx, (y, x) in enumerate(self.crop_regions):
            print(f"  区域{idx} ({region_names[idx]}): img[{y}:{y+512}, {x}:{x+512}]")
        print(f"\n场景帧数统计:")
        for scene_id in sorted(self.scene_groups.keys()):
            frame_count = len(self.scene_groups[scene_id])
            print(f"  Scene {scene_id}: {frame_count} 帧")
        print("=" * 70)
        
    def _group_by_scene(self, file_list):
        """将文件按场景分组，并处理不同的帧数"""
        scene_groups = {}
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            
            # 提取场景ID和帧ID
            match = re.match(r'scene(\d+)_frame(\d+)', filename)
            if match:
                scene_id = match.group(1)
                frame_id = int(match.group(2))
            else:
                # Fallback
                print(f"警告: 无法从文件名解析场景信息: {filename}")
                continue
            
            if scene_id not in scene_groups:
                scene_groups[scene_id] = []
            scene_groups[scene_id].append((frame_id, filepath))
        
        # 对每个场景内的帧排序
        for scene_id in scene_groups:
            scene_groups[scene_id].sort(key=lambda x: x[0])
        
        return scene_groups
    
    def _load_file_list(self, subdir):
        """加载指定子目录下的图像文件列表"""
        search_path = os.path.join(self.dataset_path, self.train_or_test, subdir)
        
        if not os.path.exists(search_path):
            print(f"警告: 路径不存在 {search_path}")
            return []
        
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(search_path, ext)))
        
        return sorted(files)
    
    def __len__(self):
        # 总样本数 = 图像数 × 区域数
        return self.len
    
    def _load_image_grayscale(self, img_path):
        """
        加载灰度图像
        返回: (H, W) 的 numpy array，值域 [0, 1]
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        # 验证图像尺寸
        if img.shape != (self.original_size[1], self.original_size[0]):
            print(f"警告: 图像尺寸不匹配 {img_path}")
            print(f"  期望: {self.original_size[1]}×{self.original_size[0]}")
            print(f"  实际: {img.shape[0]}×{img.shape[1]}")
        
        # 归一化到 [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
        
        return img
    
    def _crop_region(self, img, region_idx):
        """
        从图像中裁剪指定区域
        
        参数:
            img: (H, W) numpy array
            region_idx: 区域索引 (0-3)
        
        返回:
            (1, 512, 512, 1) numpy array
        """
        y_start, x_start = self.crop_regions[region_idx]
        y_end = y_start + self.patch_size[0]
        x_end = x_start + self.patch_size[1]
        
        # 裁剪 (注意OpenCV的索引顺序)
        cropped = img[y_start:y_end, x_start:x_end]
        
        # 验证裁剪结果
        if cropped.shape != tuple(self.patch_size):
            raise ValueError(f"裁剪区域{region_idx}尺寸不正确: {cropped.shape}")
        
        # 添加通道维度: (H, W) -> (H, W, 1)
        cropped = np.expand_dims(cropped, axis=-1)
        
        # 添加batch维度: (H, W, 1) -> (1, H, W, 1)
        cropped = np.expand_dims(cropped, axis=0)
        
        return cropped
    
    def _get_paired_frame(self, scene_id, current_frame_idx):
        """
        获取配对的帧
        在同一场景内，选择相邻的帧作为配对
        """
        scene_frames = self.scene_groups[scene_id]
        num_frames = len(scene_frames)
        
        # 确保不超出边界
        paired_idx = (current_frame_idx + self.pair_distance) % num_frames
        
        return scene_frames[paired_idx][1]  # 返回文件路径
    
    def patch_generator(self):
        """生成图像patch的迭代器"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            start = 0
            end = len(self.noisy_files)
        else:
            # 多进程数据加载
            per_worker = int(np.ceil(len(self.noisy_files) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.noisy_files))
        
        for img_idx in range(start, end):
            noisy_img1_path = self.noisy_files[img_idx]
            
            # 加载完整图像
            noisy_img1_full = self._load_image_grayscale(noisy_img1_path)
            
            if self.train_or_test == 'train':
                # ============ 训练模式 ============
                if self.scene_based_pairing:
                    # 基于场景的配对
                    scene_id = None
                    frame_idx = None
                    
                    # 查找当前图像属于哪个场景
                    for sid, frames in self.scene_groups.items():
                        for fidx, (fid, fpath) in enumerate(frames):
                            if fpath == noisy_img1_path:
                                scene_id = sid
                                frame_idx = fidx
                                break
                        if scene_id:
                            break
                    
                    if scene_id:
                        # 获取同一场景的配对帧
                        noisy_img2_path = self._get_paired_frame(scene_id, frame_idx)
                        noisy_img2_full = self._load_image_grayscale(noisy_img2_path)
                    else:
                        # Fallback
                        idx2 = (img_idx + 1) % len(self.noisy_files)
                        noisy_img2_full = self._load_image_grayscale(self.noisy_files[idx2])
                else:
                    # 简单配对
                    idx2 = (img_idx + 1) % len(self.noisy_files)
                    noisy_img2_full = self._load_image_grayscale(self.noisy_files[idx2])
                
                # 对每个区域生成样本（4个区域全部使用）
                for region_idx in range(self.num_regions):
                    # 裁剪指定区域
                    crop1 = self._crop_region(noisy_img1_full, region_idx)
                    crop2 = self._crop_region(noisy_img2_full, region_idx)
                    
                    # 转换为 (C, H, W) 格式
                    crop1 = crop1[0].transpose((2, 0, 1))  # (1, 512, 512)
                    crop2 = crop2[0].transpose((2, 0, 1))
                    
                    sample = {
                        'noisy1': torch.from_numpy(crop1).float(),
                        'noisy2': torch.from_numpy(crop2).float(),
                        'nlf0': torch.full(crop1.shape, self.default_nlf0).float(),
                        'nlf1': torch.full(crop1.shape, self.default_nlf1).float(),
                        'iso': torch.full(crop1.shape, self.default_iso).float(),
                        'cam': torch.full(crop1.shape, self.default_cam).float()
                    }
                    
                    if self.transform:
                        sample = self.transform(sample)
                    
                    yield sample
                    
            else:
                # ============ 测试模式 ============
                if img_idx < len(self.clean_files):
                    clean_img_full = self._load_image_grayscale(self.clean_files[img_idx])
                else:
                    print(f"警告: 测试图像 {img_idx} 没有对应的干净图像")
                    clean_img_full = noisy_img1_full.copy()
                
                # 如果subtract_images=True，计算噪声层
                if self.subtract_images:
                    noise_layer_full = noisy_img1_full - clean_img_full
                else:
                    noise_layer_full = noisy_img1_full
                
                # 对每个区域生成样本（4个区域全部使用）
                for region_idx in range(self.num_regions):
                    noise_crop = self._crop_region(noise_layer_full, region_idx)
                    clean_crop = self._crop_region(clean_img_full, region_idx)
                    
                    # 转换为 (C, H, W) 格式
                    noise_crop = noise_crop[0].transpose((2, 0, 1))
                    clean_crop = clean_crop[0].transpose((2, 0, 1))
                    
                    sample = {
                        'noise': torch.from_numpy(noise_crop).float(),
                        'clean': torch.from_numpy(clean_crop).float(),
                        'nlf0': torch.full(noise_crop.shape, self.default_nlf0).float(),
                        'nlf1': torch.full(noise_crop.shape, self.default_nlf1).float(),
                        'iso': torch.full(noise_crop.shape, self.default_iso).float(),
                        'cam': torch.full(noise_crop.shape, self.default_cam).float(),
                        'pid': torch.tensor(img_idx * self.num_regions + region_idx)
                    }
                    
                    if self.transform:
                        sample = self.transform(sample)
                    
                    yield sample
    
    def __iter__(self):
        return self.patch_generator()


# ========== 数据增强 ==========

class GrayscaleAugmentation:
    """针对灰度图的数据增强"""
    
    def __init__(self, 
                 flip_horizontal=True,
                 flip_vertical=True,
                 rotation=False):
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.rotation = rotation
    
    def __call__(self, sample):
        """应用数据增强"""
        
        # 水平翻转
        if self.flip_horizontal and np.random.rand() > 0.5:
            for key in ['noisy1', 'noisy2', 'nlf0', 'nlf1', 'iso', 'cam']:
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[2])
        
        # 垂直翻转
        if self.flip_vertical and np.random.rand() > 0.5:
            for key in ['noisy1', 'noisy2', 'nlf0', 'nlf1', 'iso', 'cam']:
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[1])
        
        # 旋转90度的倍数
        if self.rotation:
            k = np.random.randint(0, 4)
            if k > 0:
                for key in ['noisy1', 'noisy2', 'nlf0', 'nlf1', 'iso', 'cam']:
                    if key in sample:
                        sample[key] = torch.rot90(sample[key], k, dims=[1, 2])
        
        return sample


# ========== 场景帧数检测工具 ==========

def detect_scene_frame_counts(dataset_path, split='train'):
    """
    自动检测每个场景的帧数
    
    返回:
        字典 {'01': 150, '09': 50, ...}
    """
    noisy_dir = os.path.join(dataset_path, split, 'noisy')
    
    if not os.path.exists(noisy_dir):
        print(f"错误: 路径不存在 {noisy_dir}")
        return {}
    
    files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))
    
    scene_counts = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.match(r'scene(\d+)_frame(\d+)', filename)
        if match:
            scene_id = match.group(1)
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
    
    print(f"\n检测到的场景帧数:")
    for scene_id in sorted(scene_counts.keys()):
        print(f"  Scene {scene_id}: {scene_counts[scene_id]} 帧")
    
    return scene_counts


# ========== 数据验证工具 ==========

def validate_dataset_with_crops(dataset_path):
    """验证数据集并检查裁剪区域"""
    print("=" * 70)
    print("数据集验证（四区域裁剪）")
    print("=" * 70)
    
    for split in ['train', 'test']:
        noisy_dir = os.path.join(dataset_path, split, 'noisy')
        
        if not os.path.exists(noisy_dir):
            print(f"\n❌ {split} 目录不存在: {noisy_dir}")
            continue
        
        files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))
        
        print(f"\n【{split.upper()}】")
        print(f"  总图像数: {len(files)}")
        
        # 检测场景帧数
        scene_counts = {}
        for filepath in files:
            filename = os.path.basename(filepath)
            match = re.match(r'scene(\d+)_', filename)
            if match:
                scene_id = match.group(1)
                scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        
        if scene_counts:
            print(f"\n  场景帧数:")
            total_150 = sum(1 for c in scene_counts.values() if c == 150)
            total_50 = sum(1 for c in scene_counts.values() if c == 50)
            print(f"    150帧场景: {total_150} 个")
            print(f"    50帧场景: {total_50} 个")
            
            for scene_id in sorted(scene_counts.keys()):
                count = scene_counts[scene_id]
                print(f"      Scene {scene_id}: {count} 帧")
        
        # 检查第一张图像和裁剪区域
        if files:
            img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
            print(f"\n  图像信息:")
            print(f"    尺寸: {img.shape[0]}×{img.shape[1]}")
            print(f"    数据类型: {img.dtype}")
            print(f"    值域: [{img.min()}, {img.max()}]")
            
            # 定义4个裁剪区域
            crop_regions = [(0, 0), (0, 512), (512, 0), (512, 512)]
            region_names = ["左上", "右上", "左下", "右下"]
            
            print(f"\n  裁剪区域预览（全部可用）:")
            for idx, (y, x) in enumerate(crop_regions):
                crop = img[y:y+512, x:x+512]
                print(f"    ✓ 区域{idx} ({region_names[idx]}) [{y}:{y+512}, {x}:{x+512}]: "
                      f"尺寸={crop.shape}, 值域=[{crop.min()}, {crop.max()}]")
    
    print("\n" + "=" * 70)


# ========== 可视化工具 ==========

def visualize_crop_regions(image_path, save_path='crop_regions_visualization.png'):
    """可视化四个裁剪区域"""
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"错误: 无法加载图像 {image_path}")
        return
    
    regions = [
        (0, 0, "区域0: 左上"),
        (0, 512, "区域1: 右上"),
        (512, 0, "区域2: 左下"),
        (512, 512, "区域3: 右下")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (y, x, label) in enumerate(regions):
        crop = img[y:y+512, x:x+512]
        axes[idx].imshow(crop, cmap='gray')
        axes[idx].set_title(f'{label}\nimg[{y}:{y+512}, {x}:{x+512}]', fontsize=12)
        axes[idx].axis('off')
        
        # 显示统计信息
        info_text = f'均值: {crop.mean():.1f}\n标准差: {crop.std():.1f}'
        axes[idx].text(10, 30, info_text, 
                      fontsize=10, color='yellow',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✓ 可视化已保存: {save_path}')


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("数据集验证工具\n")
    
    dataset_path = './data/my_grayscale_dataset'
    
    # 1. 验证数据集和裁剪区域
    validate_dataset_with_crops(dataset_path)
    
    # 2. 自动检测场景帧数
    print("\n自动检测场景帧数...")
    scene_frame_counts = detect_scene_frame_counts(dataset_path, 'train')
    
    # 3. 可视化裁剪区域
    print("\n生成裁剪区域可视化...")
    first_image = sorted(glob.glob(os.path.join(dataset_path, 'train', 'noisy', '*.png')))[0]
    if first_image:
        visualize_crop_regions(first_image)
    
    # 4. 测试数据加载器
    print("\n测试数据加载器...")
    
    try:
        train_dataset = CustomGrayscaleDataset(
            dataset_path=dataset_path,
            train_or_test='train',
            num_regions=4,  # 使用全部4个区域
            patch_size=(512, 512),
            original_size=(1280, 1024),
            scene_frame_counts=scene_frame_counts,
            scene_based_pairing=True,
            iso=800,
            cam=0
        )
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0)
        
        for batch_idx, batch in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  noisy1 shape: {batch['noisy1'].shape}")  # [8, 1, 512, 512]
            print(f"  noisy2 shape: {batch['noisy2'].shape}")
            print(f"  值域: [{batch['noisy1'].min():.3f}, {batch['noisy1'].max():.3f}]")
            
            if batch_idx == 0:
                break
        
        print("\n✓ 数据加载器测试通过！")
        print(f"总图像数: {len(train_dataset.noisy_files)}")
        print(f"总样本数: {len(train_dataset)} = {len(train_dataset.noisy_files)} 图像 × 4 区域")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()