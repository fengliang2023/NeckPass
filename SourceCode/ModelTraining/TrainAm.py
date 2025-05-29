import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch.nn.functional as F
import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
from lib_siam import train_SiameseNetwork, SiameseNetwork, get_data, UserDataset, initial_SiameseNetwork, save_loaders, hinge_loss, val_func

# sima网络的逻辑是，与一个固定的anchor相减，计算相似性
INITIAL_DATA_LEN = 10

def TrainAm(DatasetPath):
    trainingset = f"{DatasetPath}/train"
    testingset = f"{DatasetPath}/test"
    all_illegal_list =  os.listdir(trainingset)[-1:]
    all_user_num = len(os.listdir(trainingset))
    all_user_list =  [item for item in os.listdir(trainingset) if item not in all_illegal_list]
    loop = len(all_user_list)
    all_user_list_2 = all_user_list + all_user_list
    for i in range(loop):
        train_usr_list = all_user_list_2[i: i + all_user_num]
        test_users_list = all_illegal_list
        positive_users = [train_usr_list[0]]
        model_name = f"{train_usr_list[0]}"
        print(f"{i:<2}/{loop -1}--------------start-----{model_name}-------------", test_users_list)
        loop_control = 1
        loop_j =0
        batch_size = 16
        all_pos_data, initial_pos_data = get_data(trainingset, positive_users, INITIAL_DATA_LEN, "1")
        all_neg_data, _ = get_data(trainingset, train_usr_list, INITIAL_DATA_LEN)
        test_pos_data, _ = get_data(testingset, positive_users, INITIAL_DATA_LEN)
        test_neg_data, _ = get_data(testingset, test_users_list, INITIAL_DATA_LEN)
        init_dataset = UserDataset([initial_pos_data])
        train_dataset = UserDataset([all_pos_data, all_neg_data], "reduce_neg")
        val_dataset = UserDataset([test_pos_data, test_neg_data])
        
        while loop_control and loop_j < 1:
            loop_j += 1
            initial_loader = DataLoader(init_dataset, batch_size=batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            # Initialize model
            input_dim = train_loader.dataset[0][0].shape[-1]

            siamese_model, mean_feature = initial_SiameseNetwork(initial_loader, input_dim=input_dim)
            epochs=1
            loop_control = train_SiameseNetwork(model_name, siamese_model, train_loader, test_loader, epochs)
        save_loaders(model_name, test_loader, mean_feature)