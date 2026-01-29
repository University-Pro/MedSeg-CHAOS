import argparse
import os
import torch
import logging
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from medpy import metric

# 引入你的项目模块
from DataLoader import CHAOSMultiModalDataset, CHAOSTransforms

# 导入网络
# from Networks.DLKUNet_S import UNet
# from Networks.SwinUNet import SwinUNet
# from Networks.UNet import UNet
from Networks.Transunet import TransUNet

# CHAOS 数据集通常定义的类别名称
CLASS_NAMES = {
    0: "Background",
    1: "Liver",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Spleen"
}

def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])

def calculate_metric_per_case(pred, gt):
    """
    计算单个切片、单个类别的 Dice 和 HD95
    pred: 二值化预测掩码 (0/1)
    gt: 二值化真值掩码 (0/1)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    # 情况1: 预测和真值都为空 -> 预测完美
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    
    # 情况2: 预测有东西，真值为空 -> 假阳性 (Dice=0, HD95设为一个较大的惩罚值)
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0.0, 100.0  # 这里的100.0是像素距离，可根据需求调整
    
    # 情况3: 预测为空，真值有东西 -> 假阴性 (Dice=0, HD95设为一个较大的惩罚值)
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0.0, 100.0
    
    # 情况4: 都有内容 -> 正常计算
    else:
        dice = metric.binary.dc(pred, gt)
        try:
            hd95 = metric.binary.hd95(pred, gt)
        except RuntimeError:
            # 极少数情况下 medpy 会计算失败
            hd95 = 100.0
        return dice, hd95

def collate_fn_filter_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def test(args):
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Testing on device: {device}")
    logging.info("=" * 60)
    logging.info(f"Testing model: {args.model_path}")

    # 2. 加载数据
    test_transform = CHAOSTransforms(mode="test")
    # test_dataset = CHAOSMultiModalDataset(base_dir=args.base_dir, split="test", transform=test_transform)
    test_dataset = CHAOSMultiModalDataset(base_dir=args.base_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             num_workers=os.cpu_count())
    
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # 3. 加载模型
    model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=16, mlp_dim=3072, block_num=12, patch_dim=16, class_num=5).to(device=device) # Transunet_m
    # model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=16, mlp_dim=4096, block_num=24, patch_dim=16, class_num=5).to(device=device) # Transunet_l
    # model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=8, mlp_dim=512, block_num=8, patch_dim=16, class_num=5).to(device=device) # Transunet_s
    # model = UNet(n_channels=3, n_classes=args.num_classes).to(device)
    # model = SwinUNet(img_size=256, in_chans=3, num_classes=args.num_classes, embed_dim=192, depths=[2,2,6,2], depths_decoder=[2,2,6,2], num_heads=[3,6,12,24], window_size=8).to(device)
    
    if os.path.isfile(args.model_path):
        logging.info(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        
        # 处理 DataParallel 保存的权重
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        logging.error(f"Error: No checkpoint found at {args.model_path}")
        return

    model.eval()

    # 初始化记录器：metric_list[class_idx] = [ (dice, hd95), (dice, hd95), ... ]
    metric_list = {i: [] for i in range(1, args.num_classes)} 

    logging.info("Start Inference...")
    
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            if sampled_batch is None:
                continue

            image = sampled_batch['image'].to(device) # [B, 3, H, W]
            label = sampled_batch['label'].cpu().numpy() # [B, H, W]
            
            # 模型推理
            outputs = model(image)
            outputs = torch.softmax(outputs, dim=1)
            
            # 获取预测结果 [B, H, W]
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()

            # 遍历 Batch 中的每一个样本 (虽然 batch_size 设为了 1，但这是一个好习惯)
            for i in range(image.shape[0]):
                pred_slice = prediction[i]
                label_slice = label[i]

                # 遍历除了背景之外的每一个类别
                for cls_idx in range(1, args.num_classes):
                    pred_cls = (pred_slice == cls_idx).astype(np.uint8)
                    label_cls = (label_slice == cls_idx).astype(np.uint8)
                    
                    # 仅当该切片中存在该器官的标签时，才计入统计（避免大量空切片拉高Dice或导致HD95无意义）
                    # 注意：是否包含空切片取决于评估标准。这里采用了比较严格的标准：
                    # 如果这层GT有器官，计算Dice；如果GT没器官但预测出来了，Dice=0也计入。
                    # 如果GT和预测都没器官，通常跳过不计入平均值，或者记为1。
                    
                    # 策略：如果 GT 中存在该类，或者 Predict 中存在该类（假阳性），则计算指标。
                    if np.sum(label_cls) > 0 or np.sum(pred_cls) > 0:
                        dice, hd95 = calculate_metric_per_case(pred_cls, label_cls)
                        metric_list[cls_idx].append([dice, hd95])

    logging.info("\n" + "="*50)
    logging.info("TEST RESULTS")
    logging.info("="*50)

    total_dice = []
    total_hd95 = []

    # 输出每个类别的结果
    for cls_idx in range(1, args.num_classes):
        metrics = np.array(metric_list[cls_idx])
        
        if len(metrics) > 0:
            avg_dice = np.mean(metrics[:, 0])
            avg_hd95 = np.mean(metrics[:, 1])
            
            # 记录用于计算全局平均
            total_dice.append(avg_dice)
            total_hd95.append(avg_hd95)
            
            class_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
            logging.info(f"{class_name:<15} | Dice: {avg_dice:.4f} | HD95: {avg_hd95:.4f} | Count: {len(metrics)}")
        else:
             logging.info(f"Class {cls_idx:<15} | Not found in test set.")

    logging.info("-" * 50)
    if len(total_dice) > 0:
        logging.info(f"{'Mean Average':<15} | Dice: {np.mean(total_dice):.4f} | HD95: {np.mean(total_hd95):.4f}")
    logging.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./datasets/Chaos", help="Path to CHAOS dataset")
    parser.add_argument("--model_path", type=str, default="./result/Pth/model_epoch_300_checkpoint.pth", help="Path to trained model")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes including background")
    parser.add_argument("--log_path", type=str, default="./result/test_result.log", help="Path to save test log")
    
    args = parser.parse_args()
    
    setup_logging(args.log_path)
    test(args)