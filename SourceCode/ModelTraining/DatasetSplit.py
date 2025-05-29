import os
import shutil
import random
from sklearn.model_selection import train_test_split
# 设置根目录路径

def DatasetSplit(InputDataset, SplitDataset):
    # 创建新的目录结构
    train_dir = os.path.join(SplitDataset, 'train')
    test_dir = os.path.join(SplitDataset, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历每个用户目录
    for user_dir in os.listdir(InputDataset):
        user_path = os.path.join(InputDataset, user_dir)
        if os.path.isdir(user_path):
            npy_files = [f for f in os.listdir(user_path) if f.endswith('.npy')]

            train_data, test_data = train_test_split(npy_files, test_size=0.2, random_state=42)
            
            for file in test_data:
                src = os.path.join(user_path, file)
                dst = os.path.join(test_dir, user_dir, file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

            for file in train_data:
                src = os.path.join(user_path, file)
                dst = os.path.join(train_dir, user_dir, file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)