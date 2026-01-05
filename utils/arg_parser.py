# Copyright (c) 2018-present, Royal Bank of Canada.
import argparse


# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--logdir", type=str,
#                         default='./logdir/', help="Location to save logs")
#     parser.add_argument("--sidd_path", type=str,
#                         default='./data/SIDD_Medium_Raw/Data', help="Location of the SIDD dataset")
#     parser.add_argument("--n_train", type=int,
#                         default=50000, help="Train epoch size")
#     parser.add_argument("--n_test", type=int,
#                         default=-1, help="Valid epoch size")
#     parser.add_argument("--n_batch_train", type=int,
#                         default=128, help="Minibatch size")
#     parser.add_argument("--n_batch_test", type=int,
#                         default=100, help="Minibatch size")
#     parser.add_argument("--epochs", type=int, default=5000,
#                         help="Total number of training epochs")
#     parser.add_argument("--epochs_full_valid", type=int,
#                         default=50, help="Epochs between valid")
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument('--lu_decomp', action='store_true', default=False)
#     parser.add_argument("--width", type=int, default=512,
#                         help="Width of hidden layers")
#     parser.add_argument("--n_bits_x", type=int, default=10,
#                         help="Number of bits of x")
#     parser.add_argument("--do_sample", action='store_true',
#                         help="To sample noisy images from the test set.")
#     # Ablation
#     parser.add_argument("--seed", type=int, default=0, help="Random seed")
#     parser.add_argument("--flow_permutation", type=int, default=1,
#                         help="Type of flow. 0=reverse, 1=1x1conv")
#     # for SIDD
#     parser.add_argument("--dataset_type", type=str, choices=['full', 'medium'],
#                         help="Full or medium")
#     parser.add_argument("--patch_height", type=int,
#                         help="Patch height, width will be the same")
#     parser.add_argument("--patch_sampling", type=str,
#                         help="Patch sampling method form full images (uniform | random)")
#     parser.add_argument("--n_tr_inst", type=int,
#                         help="Number of training scene instances")
#     parser.add_argument("--n_ts_inst", type=int,
#                         help="Number of testing scene instances")
#     parser.add_argument("--n_patches_per_image", type=int,
#                         help="Max. number of patches sampled from each image")
#     parser.add_argument("--start_tr_im_idx", type=int,
#                         help="Starting image index for training")
#     parser.add_argument("--end_tr_im_idx", type=int,
#                         help="Ending image index for training")
#     parser.add_argument("--start_ts_im_idx", type=int,
#                         help="Starting image index for testing")
#     parser.add_argument("--end_ts_im_idx", type=int,
#                         help="Ending image index for testing")
#     parser.add_argument("--camera", type=str,
#                         help="To choose training scenes from one camera")  # to lower image loading frequency
#     parser.add_argument("--iso", type=int,
#                         help="To choose training scenes from one camera")  # to lower image loading frequency
#     parser.add_argument("--arch", type=str, default='',  required=True,
#                         help="Defines a mixture architecture of bijectors")
#     parser.add_argument("--n_train_threads", type=int,
#                         help="Number of training/testing threads")
#     parser.add_argument("--n_channels", type=int, default=4,
#                         help="Number of image channles")
#     parser.add_argument('--no_resume', action='store_true')
#     parser.add_argument("--lmbda", type=int, default=1, help="value for lambda in Noise2NoiseFlow loss term")
#     parser.add_argument("--denoiser", type=str, default='dncnn',
#                         help="Denoiser architecture type, choose between dncnn/unet.")

#     parser.add_argument("--alpha", type=float, default=4, help="Alpha parameter in recorruption")
#     parser.add_argument("--sigma", type=float, default=1/256, help="std of the zero mean noise vector z for recorruption")
#     parser.add_argument("--pretrained_denoiser", default=True)
    
#     hps = parser.parse_args()  # So error if typo
#     return hps
# Copyright (c) 2018-present, Royal Bank of Canada.

def arg_parser():
    parser = argparse.ArgumentParser()
    
    # ========== 基础参数 ==========
    parser.add_argument("--logdir", type=str,
                        default='./logdir/', help="Location to save logs")
    
    # ========== 数据集参数（新增自定义数据集支持）==========
    # 自定义灰度图数据集参数（优先使用）
    parser.add_argument("--dataset_path", type=str,
                        default='./data/my_grayscale_dataset',
                        help="自定义灰度图数据集路径")
    
    parser.add_argument("--original_width", type=int, default=1280,
                        help="原始图像宽度（去除亮线后）")
    
    parser.add_argument("--original_height", type=int, default=1024,
                        help="原始图像高度（去除亮线后）")
    
    parser.add_argument("--num_regions", type=int, default=4,
                        help="裁剪区域数量（1-4），4表示使用全部4个512×512区域")
    
    # SIDD数据集参数（保留以兼容原始代码）
    parser.add_argument("--sidd_path", type=str,
                        default='./data/SIDD_Medium_Raw/Data', 
                        help="Location of the SIDD dataset")
    
    parser.add_argument("--dataset_type", type=str, 
                        choices=['full', 'medium', 'custom'],
                        default='custom',
                        help="数据集类型: full=SIDD完整, medium=SIDD中等, custom=自定义灰度图")
    
    # ========== 训练参数 ==========
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    
    parser.add_argument("--n_test", type=int,
                        default=-1, help="Valid epoch size")
    
    parser.add_argument("--n_batch_train", type=int,
                        default=16,  # 修改默认值，适合512×512图像
                        help="训练batch size（根据显存调整：4GB→4, 8GB→8-12, 12GB→16-20）")
    
    parser.add_argument("--n_batch_test", type=int,
                        default=8,  # 修改默认值
                        help="测试batch size")
    
    parser.add_argument("--epochs", type=int, default=100,  # 修改默认值
                        help="总训练轮数")
    
    parser.add_argument("--epochs_full_valid", type=int,
                        default=10,  # 修改默认值，更频繁验证
                        help="完整验证的间隔轮数")
    
    parser.add_argument("--lr", type=float, default=1e-4,  # 修改默认学习率
                        help="学习率")
    
    parser.add_argument('--lu_decomp', action='store_true', default=False,
                        help="是否使用LU分解")
    
    # ========== 模型参数 ==========
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    
    parser.add_argument("--n_bits_x", type=int, default=8,  # 灰度图通常8bit
                        help="Number of bits of x")
    
    parser.add_argument("--arch", type=str, default='resflow', 
                        help="模型架构: resflow, glow等")
    
    parser.add_argument("--flow_permutation", type=str, default='invconv',
                        help="Flow permutation类型: invconv, reverse等")
    
    parser.add_argument("--denoiser", type=str, default='dncnn',
                        choices=['dncnn', 'unet'],
                        help="去噪器架构类型: dncnn或unet")
    
    parser.add_argument("--pretrained_denoiser", action='store_true', default=False,
                        help="是否使用预训练的去噪器")
    
    parser.add_argument("--lmbda", type=float, default=1.0,  # 改为float
                        help="Noise2NoiseFlow损失项中的lambda值")
    
    # ========== 图像参数 ==========
    parser.add_argument("--patch_height", type=int, default=512,
                        help="Patch高度（宽度相同）")
    
    parser.add_argument("--n_channels", type=int, default=1,  # 修改默认值为1
                        help="图像通道数（灰度图=1, RGB=3, RAW=4）")
    
    parser.add_argument("--patch_sampling", type=str, default='uniform',
                        choices=['uniform', 'random'],
                        help="Patch采样方法: uniform或random")
    
    # ========== SIDD特定参数（保留以兼容）==========
    parser.add_argument("--n_tr_inst", type=int,
                        help="训练场景实例数")
    
    parser.add_argument("--n_ts_inst", type=int,
                        help="测试场景实例数")
    
    parser.add_argument("--n_patches_per_image", type=int,
                        help="每张图像采样的最大patch数")
    
    parser.add_argument("--start_tr_im_idx", type=int,
                        help="训练起始图像索引")
    
    parser.add_argument("--end_tr_im_idx", type=int,
                        help="训练结束图像索引")
    
    parser.add_argument("--start_ts_im_idx", type=int,
                        help="测试起始图像索引")
    
    parser.add_argument("--end_ts_im_idx", type=int,
                        help="测试结束图像索引")
    
    # ========== 相机参数 ==========
    parser.add_argument("--camera", type=int, default=0,  # 改为int
                        help="相机类型ID（0-4）")
    
    parser.add_argument("--iso", type=int, default=800,  # 设置默认值
                        help="ISO值（如: 100, 400, 800, 1600, 3200）")
    
    # ========== 其他参数 ==========
    parser.add_argument("--do_sample", action='store_true', default=False,
                        help="是否从测试集采样生成噪声图像")
    
    parser.add_argument("--seed", type=int, default=42,  # 修改默认值
                        help="随机种子")
    
    parser.add_argument("--n_train_threads", type=int, default=4,
                        help="训练/测试线程数")
    
    parser.add_argument('--no_resume', action='store_true', default=False,
                        help="不恢复之前的训练，重新开始")
    
    # ========== Recorruption参数 ==========
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Recorruption中的Alpha参数")
    
    parser.add_argument("--sigma", type=float, default=1/256,
                        help="Recorruption中零均值噪声向量z的标准差")
    
    hps = parser.parse_args()
    
    # ========== 参数验证和自动调整 ==========
    # 如果使用自定义数据集，确保某些参数设置正确
    if hps.dataset_type == 'custom':
        if hps.n_channels != 1:
            print(f"警告: 自定义灰度图数据集，自动设置 n_channels=1 (当前为{hps.n_channels})")
            hps.n_channels = 1
    
    return hps