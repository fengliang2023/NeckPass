import torch
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from lib_siam import SiameseNetwork

import torch
import json

def val_func(classifier, test_loader):
    classifier.eval()
    val_correct_pos = 0
    val_correct_neg = 0
    val_total_pos = 0
    val_total_neg = 0
    
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data = test_data.squeeze(1)  
            test_outputs = classifier(test_data)
            test_outputs = test_outputs.squeeze(1) 
            predictions = (test_outputs >= (classifier.tau)).float()
            pos_indices = (test_labels == 1)
            val_correct_pos += (predictions[pos_indices] == test_labels[pos_indices].float()).sum().item()
            val_total_pos += pos_indices.sum().item()
            neg_indices = (test_labels == 0)
            val_correct_neg += (predictions[neg_indices] == test_labels[neg_indices].float()).sum().item()
            val_total_neg += neg_indices.sum().item()


    val_accuracy_pos = val_correct_pos / val_total_pos if val_total_pos > 0 else 0
    val_accuracy_neg = val_correct_neg / val_total_neg if val_total_neg > 0 else 0

    return val_accuracy_pos, val_accuracy_neg, val_total_pos, val_total_neg

def natural_sort_key(filename):
    return [int(part) if part.isdigit() else part for part in filename.replace('.pt', '').split('_')]


def load_data(file_path):
        
    if file_path.endswith('.npy'):
        
        data = np.load(file_path)
        
        data = torch.tensor(data)
    elif file_path.endswith('.pt') or file_path.endswith('.pth'):
        
        data = torch.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return data


def get_data(Input_path, neg_users):
    all_data = []
    for neg_user in neg_users:
        neg_user_dir = os.path.join(Input_path, neg_user)
        all_file_names = os.listdir(neg_user_dir)
        if 'mean_feature.pt' in all_file_names:
            mean_feature = torch.load(os.path.join(neg_user_dir, 'mean_feature.pt'))
            all_data.append((mean_feature, neg_user, 'mean'))
        sorted_file_names = sorted(all_file_names, key=natural_sort_key)
        if 'mean_feature.pt' in all_file_names:
            sorted_file_names.append('mean_feature.pt')
        for file_name in sorted_file_names:
            file_path = os.path.join(neg_user_dir, file_name)
            data = load_data(file_path)
            label = "1"
            all_data.append((data, file_name.split(".pt")[0], label))
    return all_data

def load_model(model_path, input_dim):
    model = SiameseNetwork(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_paired(pos_list, num_time=2):
    final_pos_list = []
    for ii in range(len(pos_list) -num_time + 1):
        tmp_value = 0
        for jj in range(num_time):
            tmp_value += pos_list[ii + jj]
        if tmp_value > 0:
            final_pos_list.append(1)
        else:
            final_pos_list.append(0)
    return final_pos_list


if __name__ == "__main__":
    with open(f'../lib/tpr.json', 'r') as f:
        TPR_dic = json.load(f)
    Input_path =  f'Dataset_NeckPass'
    pos_user_list =  os.listdir(Input_path) 
    loop = len(pos_user_list)
    TPR_list = []
    TNR_list = []
    BAC_list = []
    User_name = []
    for i in range(loop):
        pos_users = [pos_user_list[i]]
        model_name = f"{pos_users[0]}"
        all_users_data = get_data(Input_path, pos_users)
        model_path = f'../8.2_OverallPerformance/Model_Am/Am_{model_name}.pth'
        input_dim = all_users_data[0][0].shape[0]
        model = load_model(model_path, input_dim)
        mean_feature = torch.load(f'../8.2_OverallPerformance/Dataset_NeckPass/{model_name}/mean_feature.pt')
        model.anchor = mean_feature

        model.eval()
        pos_list = []
        neg_list = []
        for ii, (data, file_name, label) in enumerate(all_users_data):
            _,_, outputs = model(data)
            preds_label = (outputs[0] >= model.tau)
            if label == "1":
                if preds_label:
                    pos_list.append(1)
                else:
                    pos_list.append(0)
            elif label == "0":
                if preds_label:
                    neg_list.append(1)
                else:
                    neg_list.append(0)
        pos_list = get_paired(pos_list,2)
        neg_list = get_paired(neg_list,2)
        count_pos = pos_list.count(1)
        count_neg = neg_list.count(1)
        TPR = (count_pos/len(pos_list)*100) if len(pos_list) > 0 else 0
        TPR_list.append(TPR)
        TNR = TPR_dic[model_name][1]
        TNR_list.append(TNR)
        BAC = (TPR + TNR) / 2
        BAC_list.append(BAC)
        User_name.append(model_name)
    print(f"No, User_name      ----   TPR,      TNR,      BAC")
    for i, (user_name, TPR, TNR, BAC) in enumerate(zip(User_name, TPR_list, TNR_list, BAC_list)):
        print(f"{i:<2},{user_name:<15} ----   {TPR:.2f}%,   {np.mean(TNR):.2f}%,   {np.mean(BAC):.2f}%")
    print(f"\n[Average]    ----   {np.mean(TPR_list):.2f}%,   {np.mean(TNR_list):.2f}%,   {np.mean(BAC_list):.2f}%")