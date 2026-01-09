import torch
from torch.utils.data import DataLoader
from custom_loader import CustomGrayscaleDataset

dataset = CustomGrayscaleDataset(
    dataset_path='/home/lt/denoising/N2Nflow/Noise2NoiseFlow/noise2noiseflow/data/my_grayscale_dataset',
    train_or_test='train',
    num_regions=4,
    patch_size=(512, 512),
    original_size=(1280, 1024),
    scene_based_pairing=True,
    iso=800,
    cam=0
)

print('测试单个样本...')
for i, sample in enumerate(dataset):
    print(f'Sample {i}:')
    for key, value in sample.items():
        print(f'  {key}: shape={value.shape}, dtype={value.dtype}')
    if i >= 2:
        break

print('\n测试 DataLoader (batch_size=4)...')
loader = DataLoader(dataset, batch_size=4, num_workers=0)
for batch in loader:
    print('Batch:')
    for key, value in batch.items():
        print(f'  {key}: shape={value.shape}, dtype={value.dtype}')
    break

print('\n测试通过')
