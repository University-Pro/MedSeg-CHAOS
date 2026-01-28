import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CHAOSTransforms:
    """CHAOS数据集的数据增强"""
    def __init__(self, mode="train"):
        self.mode = mode
        
        # 此时图像已经MinMax归一化到了[0,1]，所以 mean=0.5, std=0.5 会将数据映射到 [-1, 1]
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
        
        if mode == "train":
            self.transform = A.Compose([
                A.Resize(256, 256), # 建议稍微大一点，或者根据显存调整，这里设为256
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # 可以增加一些旋转，增加鲁棒性
                A.Rotate(limit=15, p=0.5),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2()
            ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image 输入进来是 (C, H, W)，Albumentations 需要 (H, W, C)
        image = image.transpose(1, 2, 0)
        
        # 确保 label 是 int 类型用于 mask 处理，防止 resize 时插值产生小数
        augmented = self.transform(image=image, mask=label)
        
        image = augmented['image']
        label = augmented['mask']
        
        # label 输出保持 Long 类型 (int64) 用于 CrossEntropyLoss
        return {'image': image, 'label': label.long()}

class CHAOSMultiModalDataset(Dataset):
    """CHAOS多模态分割数据集 (T1-In, T1-Out, T2-SPIR)"""
    def __init__(self, base_dir="./Datasets/Chaos", split="train", transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        
        train_ids, test_ids = self._get_fold_ids()
        patient_ids = train_ids if split == "train" else test_ids
        
        print(f"[{split.upper()}] 正在扫描文件并构建样本列表...")
        
        # 构建样本列表 (在初始化阶段就确保所有模态文件都存在)
        for patient_id in patient_ids:
            # 基础路径
            t1_dir = os.path.join(self._base_dir, 't1', str(patient_id))
            t2_dir = os.path.join(self._base_dir, 't2', str(patient_id))
            
            if not os.path.isdir(t1_dir) or not os.path.isdir(t2_dir):
                continue
            
            # 扫描 T1 In-phase 文件作为基准
            for filename in os.listdir(t1_dir):
                # 匹配类似 patient1_t1_inphase_5.npz 的文件名
                if 'inphase' in filename and filename.endswith('.npz'):
                    match = re.search(r'_(\d+)\.npz$', filename)
                    if match:
                        slice_idx = int(match.group(1))
                        
                        f_t1_in = os.path.join(t1_dir, f'patient{patient_id}_t1_inphase_{slice_idx}.npz')
                        f_t1_out = os.path.join(t1_dir, f'patient{patient_id}_t1_outphase_{slice_idx}.npz')
                        f_t2_spir = os.path.join(t2_dir, f'patient{patient_id}_t2_{slice_idx}.npz')
                        
                        if os.path.exists(f_t1_in) and os.path.exists(f_t1_out) and os.path.exists(f_t2_spir):
                            self.sample_list.append({
                                'patient_id': patient_id,
                                'slice_idx': slice_idx,
                                'path_t1_in': f_t1_in,
                                'path_t1_out': f_t1_out,
                                'path_t2': f_t2_spir
                            })
        
        self.sample_list.sort(key=lambda x: (x['patient_id'], x['slice_idx']))
        print(f"[{split.upper()}] 构建完成，共找到 {len(self.sample_list)} 个有效多模态样本。")

    def _get_fold_ids(self):
        """定义训练集和测试集的病人ID"""
        training_set = [1, 2, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38]
        testing_set = [3, 39]
        return training_set, testing_set

    def _normalize_slice(self, slice_arr):
        """
        对单张切片进行 Min-Max 归一化到 [0, 1]
        MRI 图像的像素值范围不固定，必须归一化。
        """
        val_min = slice_arr.min()
        val_max = slice_arr.max()
        if val_max - val_min == 0:
            return slice_arr # 避免除以0，通常是全黑图像
        return (slice_arr - val_min) / (val_max - val_min)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        # 加载数据
        # 假设 npz 里的 key 是 'img'
        try:
            img_t1_in = np.load(item['path_t1_in'])['img']
            
            data_t1_out = np.load(item['path_t1_out'])
            img_t1_out = data_t1_out['img']
            label = data_t1_out['lab'] # 标签通常在一个模态里取即可 (已配准)
            
            img_t2_spir = np.load(item['path_t2'])['img']
        except Exception as e:
            print(f"Error loading {item['path_t1_in']}: {e}")
            raise e

        # === 关键步骤：归一化 ===
        img_t1_in = self._normalize_slice(img_t1_in.astype(np.float32))
        img_t1_out = self._normalize_slice(img_t1_out.astype(np.float32))
        img_t2_spir = self._normalize_slice(img_t2_spir.astype(np.float32))

        # 堆叠通道 (3, H, W) -> [T1-In, T1-Out, T2-SPIR]
        image = np.stack([img_t1_in, img_t1_out, img_t2_spir], axis=0)
        
        # 标签处理
        # 原始 label 可能是 int64，不需要归一化，只需转类型
        label = label.astype(np.int32) 

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
            
        return sample

def run_dataloader_test():
    """测试Dataloader功能"""
    print("\n--- 开始测试多模态Dataloader (Antigravity Revised) ---")
    
    # 请确保路径正确，推荐使用绝对路径
    base_dir = os.path.abspath("./Datasets/Chaos") 
    
    if not os.path.exists(base_dir):
        print(f"错误：路径 {base_dir} 不存在，请检查路径配置。")
        return  # 路径不存在直接退出，避免后续错误

    # 实例化数据集
    train_transform = CHAOSTransforms(mode="train")
    test_transform = CHAOSTransforms(mode="test")
    
    train_dataset = CHAOSMultiModalDataset(base_dir=base_dir, split='train', transform=train_transform)
    test_dataset = CHAOSMultiModalDataset(base_dir=base_dir, split='test', transform=test_transform)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # ========== 测试训练集 ==========
    if len(train_dataset) > 0:
        print("\n验证 Train Loader...")
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=min(4, os.cpu_count())  # 避免在某些系统上因 num_workers 过大出错
        )
        try:
            train_batch = next(iter(train_loader))
            images = train_batch['image']
            labels = train_batch['label']
            
            print(f"[Train] Image Shape: {images.shape} (Batch, Channel, H, W)")
            print(f"[Train] Label Shape: {labels.shape} (Batch, H, W)")
            print(f"[Train] Image Range: Min={images.min():.3f}, Max={images.max():.3f}")
            print(f"[Train] Label Unique Values: {torch.unique(labels)}")
            
            # 检查归一化是否正常 (经过Normalize后，值应该在 -1 到 1 附近)
            if images.max() > 100 or images.min() < -100:
                print("⚠️ 警告：图像像素值异常大，归一化可能未生效！")
            else:
                print("✅ 检查：图像数值范围正常 (已归一化)。")
                
        except Exception as e:
            print(f"❌ 训练集读取出错: {e}")
    else:
        print("⚠️ 训练集未找到匹配样本，请检查文件名格式或路径。")

    # ========== 测试测试集 ==========
    if len(test_dataset) > 0:
        print("\n验证 Test Loader...")
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=2,  # 测试时通常 batch_size 较小
            shuffle=False, 
            num_workers=min(2, os.cpu_count())
        )
        try:
            test_batch = next(iter(test_loader))
            images = test_batch['image']
            labels = test_batch['label']
            
            print(f"[Test] Image Shape: {images.shape} (Batch, Channel, H, W)")
            print(f"[Test] Label Shape: {labels.shape} (Batch, H, W)")
            print(f"[Test] Image Range: Min={images.min():.3f}, Max={images.max():.3f}")
            print(f"[Test] Label Unique Values: {torch.unique(labels)}")
            
            # 测试集通常也应归一化（取决于你的 CHAOSTransforms 实现）
            if images.max() > 100 or images.min() < -100:
                print("⚠️ 警告：测试集图像像素值异常大，归一化可能未生效！")
            else:
                print("✅ 检查：测试集图像数值范围正常 (已归一化)。")
                
        except Exception as e:
            print(f"❌ 测试集读取出错: {e}")
    else:
        print("⚠️ 测试集未找到匹配样本，请检查文件名格式或路径。")

    print("\n--- Dataloader 测试完成 ---")


if __name__ == "__main__":
    run_dataloader_test()