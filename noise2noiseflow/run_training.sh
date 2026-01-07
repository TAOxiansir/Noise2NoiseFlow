#!/bin/bash

# ====================================
# Noise2NoiseFlow 训练脚本
# 数据集: 12场景（8×150帧 + 4×50帧）
# 原图: 1280×1024（已去除亮线）
# 裁剪: 4个512×512区域（全部使用）
# ====================================

# ========== 数据集配置 ==========
DATASET_PATH="./data/my_grayscale_dataset"

# 图像配置
ORIGINAL_WIDTH=1280
ORIGINAL_HEIGHT=1024
CROP_SIZE=512
NUM_REGIONS=4  # 使用全部4个区域

# 场景配置
TOTAL_SCENES=12
SCENES_150=8  # 8个场景×150帧
SCENES_50=4   # 4个场景×50帧
TOTAL_IMAGES=$((8 * 150 + 4 * 50))  # 1400张

# 预估样本数（根据实际划分可能不同）
TRAIN_IMAGES=1150  # 假设: 7×150 + 2×50
TEST_IMAGES=250    # 假设: 1×150 + 2×50
TRAIN_SAMPLES=$((TRAIN_IMAGES * NUM_REGIONS))  # 4,600个样本
TEST_SAMPLES=$((TEST_IMAGES * NUM_REGIONS))    # 1,000个样本

echo "========================================"
echo "训练配置"
echo "========================================"
echo "数据集路径: $DATASET_PATH"
echo "总图像数: $TOTAL_IMAGES 张"
echo "预估训练样本: $TRAIN_SAMPLES (图像 × $NUM_REGIONS 区域)"
echo "预估测试样本: $TEST_SAMPLES"
echo ""

# ========== 训练参数 ==========
# 批次大小（根据显存调整）
# 数据量大，可以用更大的batch size
# 4GB显存:  BATCH_SIZE=4
# 8GB显存:  BATCH_SIZE=8-12
# 12GB显存: BATCH_SIZE=16-20
# 16GB+显存: BATCH_SIZE=24-32
BATCH_SIZE=16

# 训练轮数（数据量增加，可适当减少）
EPOCHS=100

# 学习率
LEARNING_RATE=0.0001

echo "训练参数:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# ========== 模型参数 ==========
N_CHANNELS=1
ARCH="resflow"
FLOW_PERMUTATION="invconv"
LU_DECOMP=1
DENOISER="dncnn"
DNCNN_LAYERS=9
LAMBDA=1.0

# ========== 其他参数 ==========
LOGDIR="grayscale_4crop_full_12scenes"
SEED=42
N_BITS_X=8
ISO=800
CAMERA=0
EPOCHS_FULL_VALID=10
DO_SAMPLE=true

# ========== 检查数据集 ==========
echo "========================================"
echo "数据集检查"
echo "========================================"

if [ ! -d "$DATASET_PATH/train/noisy" ]; then
    echo "❌ 错误: 训练集不存在"
    echo ""
    echo "请确保数据集结构:"
    echo "$DATASET_PATH/"
    echo "  ├── train/noisy/"
    echo "  │   ├── scene01_frame001.png"
    echo "  │   └── ..."
    echo "  └── test/"
    echo "      ├── noisy/"
    echo "      └── clean/ (可选)"
    exit 1
fi

# 统计实际图像数
ACTUAL_TRAIN=$(ls $DATASET_PATH/train/noisy/*.png 2>/dev/null | wc -l)
ACTUAL_TEST=$(ls $DATASET_PATH/test/noisy/*.png 2>/dev/null | wc -l)

echo "实际图像数:"
echo "  训练: $ACTUAL_TRAIN 张 → $((ACTUAL_TRAIN * NUM_REGIONS)) 个样本"
echo "  测试: $ACTUAL_TEST 张 → $((ACTUAL_TEST * NUM_REGIONS)) 个样本"

if [ $ACTUAL_TRAIN -eq 0 ]; then
    echo ""
    echo "❌ 错误: 训练集为空"
    exit 1
fi

# 检查第一张图像尺寸
FIRST_IMAGE=$(ls $DATASET_PATH/train/noisy/*.png 2>/dev/null | head -n 1)
if [ -n "$FIRST_IMAGE" ]; then
    IMAGE_INFO=$(python3 -c "
import cv2
img = cv2.imread('$FIRST_IMAGE', 0)
print(f'{img.shape[0]}×{img.shape[1]}')
" 2>/dev/null)
    
    echo ""
    echo "图像信息:"
    echo "  第一张: $(basename $FIRST_IMAGE)"
    echo "  尺寸: $IMAGE_INFO"
    
    # 验证尺寸
    if [ "$IMAGE_INFO" != "1024×1280" ]; then
        echo "  ⚠️  警告: 尺寸可能不匹配，期望 1024×1280"
    else
        echo "  ✓ 尺寸正确"
    fi
fi

# ========== 裁剪配置 ==========
echo ""
echo "裁剪配置:"
echo "  原始尺寸: ${ORIGINAL_WIDTH}×${ORIGINAL_HEIGHT}"
echo "  裁剪为: 4个 ${CROP_SIZE}×${CROP_SIZE} 区域"
echo "  区域列表:"
echo "    ✓ 区域0: [0:512, 0:512] (左上)"
echo "    ✓ 区域1: [0:512, 512:1024] (右上)"
echo "    ✓ 区域2: [512:1024, 0:512] (左下)"
echo "    ✓ 区域3: [512:1024, 512:1024] (右下)"
echo "  说明: 全部4个区域可用（已去除亮线）"

# ========== 显存估算 ==========
echo ""
echo "显存需求估算:"
# 每个样本: 512×512×4字节 ≈ 1MB
# 显存 ≈ batch_size × 1MB × 2(noisy1+noisy2) × 2(前向+反向) × 安全系数
ESTIMATED_GB=$((BATCH_SIZE * 4 / 1024))
echo "  约需显存: ~${ESTIMATED_GB}GB (batch_size=$BATCH_SIZE)"

if [ $BATCH_SIZE -gt 20 ] && [ $ESTIMATED_GB -lt 12 ]; then
    echo "  ⚠️  警告: batch_size较大，请确保有足够显存"
fi

# ========== 运行前确认 ==========
echo ""
echo "========================================"
echo "准备开始训练"
echo "========================================"
echo "日志目录: experiments/paper/$LOGDIR/"
echo ""
read -p "确认开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ========== 开始训练 ==========
echo ""
echo "开始训练..."
echo ""

python train_noise2noiseflow.py \
    --dataset_path $DATASET_PATH \
    --patch_height $CROP_SIZE \
    --n_batch_train $BATCH_SIZE \
    --n_batch_test 8 \
    --n_channels $N_CHANNELS \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --arch $ARCH \
    --flow_permutation $FLOW_PERMUTATION \
    --lu_decomp $LU_DECOMP \
    --denoiser $DENOISER \
    --lmbda $LAMBDA \
    --logdir $LOGDIR \
    --seed $SEED \
    --n_bits_x $N_BITS_X \
    --iso $ISO \
    --camera $CAMERA \
    --epochs_full_valid $EPOCHS_FULL_VALID \
    --no_resume \
    ${DO_SAMPLE:+--do_sample}

# ========== 训练完成 ==========
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 训练完成！"
    echo "========================================"
    echo ""
    echo "结果文件:"
    echo "  模型: experiments/paper/$LOGDIR/saved_models/"
    echo "    - best_model.pth (最佳模型)"
    echo "    - epoch_*_nf_model_net.pth (各轮次)"
    echo ""
    echo "  日志: experiments/paper/$LOGDIR/"
    echo "    - train.txt (训练日志)"
    echo "    - test.txt (测试日志)"
    echo "    - sample.txt (采样日志)"
    echo ""
    echo "  TensorBoard: experiments/paper/$LOGDIR/tensorboard_logs/"
    echo ""
    echo "查看训练曲线:"
    echo "  tensorboard --logdir=experiments/paper/$LOGDIR/tensorboard_logs/"
    echo "  然后访问: http://localhost:6006"
    echo ""
    echo "数据统计:"
    echo "  训练样本: $((ACTUAL_TRAIN * NUM_REGIONS)) (${ACTUAL_TRAIN}图像 × 4区域)"
    echo "  测试样本: $((ACTUAL_TEST * NUM_REGIONS)) (${ACTUAL_TEST}图像 × 4区域)"
    echo ""
    echo "下一步:"
    echo "  1. 查看训练曲线"
    echo "  2. 使用最佳模型进行推理"
    echo "  3. 计算PSNR/SSIM评估性能"
else
    echo ""
    echo "========================================"
    echo "❌ 训练失败"
    echo "========================================"
    echo ""
    echo "可能的原因:"
    echo "  - 显存不足: 尝试减小 batch_size"
    echo "  - 数据加载错误: 检查数据集路径和文件命名"
    echo "  - 依赖缺失: 检查Python环境"
    echo ""
    echo "请查看上方错误信息进行调试"
    exit 1
fi