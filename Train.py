"""
Training code for CHAOS Dataset
"""

import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# 导入新的进度条显示
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    TaskProgressColumn,
)
from glob import glob

from DataLoader import CHAOSMultiModalDataset, CHAOSTransforms
from Utils.LossFunction import CeDiceLoss

# 导入相关网络
# from Networks.SwinUNet import swin_unet_base
from Networks.SwinUNet import swin_unet_small
# from Networks.DLKUNet_S import UNet
# from Networks.SwinUNet import SwinUNet
# from Networks.UNet import UNet
# from Networks.Transunet import TransUNet

def set_seed(seed_value=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_file):
    """配置日志"""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])
    logging.info("Logging is set up.")

def latest_checkpoint(path):
    """查找最新的权重文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_model(model, model_path, device):
    """加载模型权重"""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # 如果加载的是 DataParallel 保存的模型，移除 'module.' 前缀
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v 
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model

def collate_fn_filter_none(batch):
    """
    自定义整理函数，用于过滤 Dataset 返回的 None 数据
    (对应 CHAOS Dataloader 中文件读取失败的情况)
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_model(model, train_dataset, epochs=300, batch_size=12, learning_rate=1e-4,
                save_path=None, continue_train=False, multi_gpu=False, num_classes=5):
    
    # 根据你的 CPU 核心数调整 num_workers
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=os.cpu_count(), 
                              pin_memory=True,
                              collate_fn=collate_fn_filter_none,
                              drop_last=True) # 建议加上 drop_last=True 防止最后一个batch尺寸不一致导致BN报错

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  
    logging.info(f"Optimizer: {optimizer.__class__.__name__} with parameters: {optimizer.defaults}")

    criterion = CeDiceLoss(num_classes=num_classes, loss_weight=[0.6, 1.4])
    logging.info(f"Criterion: {criterion.__class__.__name__} with parameters: num_classes={num_classes}")

    scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
    logging.info(f"Scheduler: {scheduler.__class__.__name__} with step_size={scheduler.step_size}, gamma={scheduler.gamma}")

    model.train()
    set_seed(42)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if multi_gpu and torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")

    console = Console()
    progress_columns = [
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, complete_style="bright_magenta"),
        TaskProgressColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_columns, console=console, refresh_per_second=10) as progress:
        for epoch in range(epochs):
            running_loss = 0.0
            valid_batches = 0
            task = progress.add_task(f"Epoch {epoch+1}/{epochs}", total=len(train_loader))

            for i_batch, sampled_batch in enumerate(train_loader):
                if sampled_batch is None:
                    progress.advance(task)
                    continue

                images = sampled_batch['image'].to(device)
                labels = sampled_batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss
                valid_batches += 1

                avg_loss = running_loss / valid_batches if valid_batches else 0.0
                progress.update(
                    task,
                    advance=1,
                    description=f"[bold green]Epoch {epoch+1}/{epochs}",
                )
                progress.console.log(f"[yellow]Loss:[/yellow] {current_loss:.4f} [bright_blue]Avg:[/bright_blue] {avg_loss:.4f}")
            
            if valid_batches > 0:
                epoch_loss = running_loss / valid_batches
            else:
                epoch_loss = 0.0
                
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.4f}, Learning Rate: {current_lr}")

            scheduler.step()

            # 每 20 个 epoch 保存一次模型
            if (epoch + 1) % 20 == 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                temp_path = os.path.join(save_path, f'model_epoch_{epoch+1}_checkpoint.pth')
                
                # 如果使用了 DataParallel，保存时建议存 module.state_dict()
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), temp_path)
                else:
                    torch.save(model.state_dict(), temp_path)

                logging.info(f"Saved checkpoint at epoch {epoch+1} at {temp_path}")

    logging.info("Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CHAOS Segmentation Model")
    parser.add_argument("--base_dir", type=str, default="./datasets/Chaos", help="Path to CHAOS dataset root")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--log_path", type=str, default="./result/running.log", help="Path to save running log")
    parser.add_argument("--pth_path", type=str, default='./result/Pth', help="Path to save model checkpoints")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (0:BG + organs)")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from latest checkpoint")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs to train the model")

    option = parser.parse_args()

    setup_logging(option.log_path)
    logging.info(f"Running with parameters: {vars(option)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Now is Going to use {device.type}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # 1. 实例化 Transforms
    # 注意：CHAOSTransforms 内部设置了 Resize(256, 256)
    train_transform = CHAOSTransforms(mode="train")
    
    # 2. 实例化 CHAOS Dataset
    # db_train = CHAOSMultiModalDataset(base_dir=option.base_dir, split="train", transform=train_transform) # transformer写的有问题，暂时不使用
    db_train = CHAOSMultiModalDataset(base_dir=option.base_dir, split="train") # 不使用transformer

    # 3. 实例化模型
    # model = swin_unet_base(img_size=256, in_chans=3, num_classes=5).to(device) # SwinUNet_Base
    model = swin_unet_small(img_size=256, in_chans=3, num_classes=option.num_classes).to(device) # SwinUNet_Small
    # model = SwinUNet(img_size=256, in_chans=3, num_classes=option.num_classes, embed_dim=96, depths=[2, 2, 6, 2], depths_decoder=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=8).to(device=device)
    # model = UNet(n_channels=3, n_classes=option.num_classes).to(device=device)
    # model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=8, mlp_dim=512, block_num=8, patch_dim=16, class_num=5).to(device=device) # Transunet_s
    # model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=16, mlp_dim=3072, block_num=12, patch_dim=16, class_num=option.num_classes).to(device=device) # Transunet_m
    # model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=16, mlp_dim=4096, block_num=24, patch_dim=16, class_num=option.num_classes).to(device=device) # Transunet_l
    

    # 如果继续训练
    if option.continue_train:
        if not os.path.exists(option.pth_path):
             os.makedirs(option.pth_path)
        checkpoint = latest_checkpoint(option.pth_path)
        if checkpoint:
            load_model(model=model, model_path=checkpoint, device=device) 
            logging.info(f"Continuing training from {checkpoint}")
        else:
            logging.info("No checkpoint found, starting a new training session")
    
    if not os.path.exists(option.pth_path):
        os.makedirs(option.pth_path)

    # 4. 开始训练
    train_model(model, 
                db_train, 
                epochs=option.epochs, 
                batch_size=option.batch_size, 
                learning_rate=option.learning_rate, 
                save_path=option.pth_path,
                continue_train=option.continue_train,
                multi_gpu=option.multi_gpu,
                num_classes=option.num_classes)