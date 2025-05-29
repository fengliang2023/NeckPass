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





INITIAL_DATA_LEN = 10

class BaseNetwork(nn.Module):  
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)  
        x = self.relu(x)  
        x = self.fc1(x)  
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork(input_dim)
        self.anchor = torch.zeros(768) 
        self.tau = nn.Parameter(torch.tensor(0.4))
        self.fc = nn.Linear(input_dim , 1, bias=True)
        self.fc2 = nn.Linear(64 , 64, bias=True)
        self.fc1 = nn.Linear(input_dim , 1, bias=True)

    def forward(self, input1):
        input_x1 = self.base_network(input1)
        input2 = self.anchor.unsqueeze(0).expand(16, -1)
        input_x2 = self.base_network(input2)
        sub_x = abs(input_x1 - input_x2)

        output_x = self.fc1(sub_x)
        output = torch.sigmoid(output_x)
        return input_x1, input_x2, output



def cal_mse(data):
    mean_vector = np.mean(data, axis=0)
    cosine_similarities = []
    for vector in data:
        vector_flat = vector.flatten()
        mean_vector_flat = mean_vector.flatten()
        similarity = np.dot(vector_flat, mean_vector_flat) / (np.linalg.norm(vector_flat) * np.linalg.norm(mean_vector_flat))
        cosine_similarities.append(similarity)
    return np.array(cosine_similarities)


def cal_data(neg_user, each_neg_data, initial_data_len):
    threshold = 0.8
    if initial_data_len >  len(each_neg_data):
        tmp_data = each_neg_data[: -1]
    else:
        left_num = 1
        tmp_data = each_neg_data[ : initial_data_len]
        rest_data = each_neg_data[initial_data_len:]
        while left_num:
            cosine_similarities = cal_mse(tmp_data)
            valid_indices = cosine_similarities > threshold
            tmp_data = [tmp_data[i] for i in range(len(tmp_data)) if valid_indices[i]]
            left_num = initial_data_len - len(tmp_data)
            if left_num > len(rest_data):
                break
            else:
                tmp_data += rest_data[:left_num]
                rest_data = rest_data[left_num:]
    
    cosine_similarities = cal_mse(tmp_data)
    if len(cosine_similarities) and np.all(cosine_similarities > threshold):
        print(f"success---{neg_user}: {cosine_similarities}")
        pass
    else:
        print(f"fail---{neg_user}: {cosine_similarities}")
        tmp_data = each_neg_data[ : initial_data_len]
        cosine_similarities = cal_mse(tmp_data)
        print(f"fail---{neg_user}: {cosine_similarities}")
        pass
    return tmp_data


def get_data(root_dir, neg_users, initial_data_len, label = "0"):
    all_neg_data = []
    initial_neg_data = []
    for neg_user in neg_users:
        neg_user_dir = os.path.join(root_dir, neg_user)
        each_neg_data = []
        for file_name in os.listdir(neg_user_dir):
            file_path = os.path.join(neg_user_dir, file_name)
            data = np.load(file_path)
            
            
            each_neg_data.append(data)
        if label == "1":
            initial_neg_data += cal_data(neg_user, each_neg_data, initial_data_len)
        else:
            initial_neg_data += each_neg_data[:initial_data_len]
        all_neg_data += each_neg_data
    return all_neg_data, initial_neg_data

def data_augmentation(usr_data_list, len_tmp = 100):
    if len(usr_data_list) < len_tmp:
        stop_len = len_tmp - len(usr_data_list)
        augumentation_list = []
        for i, x in enumerate(usr_data_list):
            for y in usr_data_list[i+1:]:
                augumentation_list += [(x + y) / 2]
                if len(augumentation_list) > stop_len:
                    break
            if len(augumentation_list) > stop_len:
                break
        usr_data_list += augumentation_list[:stop_len]
    return usr_data_list


class UserDataset(Dataset):
    def __init__(self, data_list, label = ""):
        self.data = []
        self.labels = []
        if len(data_list) == 1:
            initial_pos_data = data_list[0]
            self.data += initial_pos_data
            self.labels += [1]*len(initial_pos_data)

        if len(data_list) == 2:
            all_pos_data = data_list[0]
            all_neg_data = data_list[1]
            if label == "train":
                len_data = len(all_neg_data) 
                len_tmp = int( len(all_neg_data) / len(all_pos_data) ) + 1
                all_pos_data = all_pos_data*len_tmp
                all_pos_data = all_pos_data[:len_data]
            if label == "reduce_neg":
                all_pos_data = data_augmentation(all_pos_data)
                all_neg_data = random.sample(all_neg_data, len(all_pos_data))

            self.data += all_pos_data
            self.labels += [1]*len(all_pos_data)
            self.data += all_neg_data
            self.labels += [0]*len(all_neg_data)

            self.deal_batch()

    def deal_batch(self):
        len_data = len(self.data) - (len(self.data) % 16)
        self.data = self.data[:len_data]
        self.labels = self.labels[:len_data]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]


def hinge_loss(y_pred, tau, labels, margin=0.3):
    
    loss_pos = torch.clamp(margin - (y_pred - tau), min=0)  
    loss_neg = torch.clamp(margin + (y_pred - tau), min=0)  
    loss = labels * loss_pos + (1 - labels) * loss_neg
    return loss.mean()


def initial_SiameseNetwork(train_loader, input_dim):
    train_features = []
    train_labels = []

    with torch.no_grad():
        for data, labels in train_loader:
            train_features.append(data)
            train_labels.append(labels)
    train_features = torch.cat(train_features).squeeze(1)
    train_labels = torch.cat(train_labels)

    siamese_model =  SiameseNetwork(input_dim=input_dim)

    class_idx = 1
    class_features = train_features[train_labels == class_idx][:1]
    mean_feature = class_features.mean(dim=0)

    siamese_model.anchor = mean_feature
    normalized_feature = mean_feature / mean_feature.norm()
    siamese_model.fc.weight.data[0] = normalized_feature
    return siamese_model, mean_feature


def val_func(classifier, test_loader):
    classifier.eval()
    val_correct_pos = 0
    val_correct_neg = 0
    val_total_pos = 0
    val_total_neg = 0
    
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data = test_data.squeeze(1)  
            _,_,test_outputs = classifier(test_data)
            test_outputs = test_outputs.squeeze(1) 
            predictions = (test_outputs >= classifier.tau).float()

            
            pos_indices = (test_labels == 1)
            val_correct_pos += (predictions[pos_indices] == test_labels[pos_indices].float()).sum().item()
            val_total_pos += pos_indices.sum().item()

            
            neg_indices = (test_labels == 0)
            val_correct_neg += (predictions[neg_indices] == test_labels[neg_indices].float()).sum().item()
            val_total_neg += neg_indices.sum().item()


    val_accuracy_pos = val_correct_pos / val_total_pos if val_total_pos > 0 else 0
    val_accuracy_neg = val_correct_neg / val_total_neg if val_total_neg > 0 else 0

    return val_accuracy_pos, val_accuracy_neg, val_total_pos, val_total_neg


def train_SiameseNetwork(model_name,siamese_model, train_loader, test_loader, epochs):


    optimizer = optim.Adam(siamese_model.parameters(), lr=0.00001, weight_decay=0.0001)
    best_val_accuracy = 0
    patience_counter = 0
    patience= 10
    for epoch in range(epochs):
        siamese_model.train()
        total_loss = 0
        total_correct = 0
        total = 0
        total_steps = 0

        
        for train_data, train_labels in train_loader:
            train_data = train_data.squeeze(1)  
            optimizer.zero_grad()
            output1, output2, train_outputs = siamese_model(train_data)
            train_outputs = train_outputs.squeeze(1)

            hinge_loss_value = hinge_loss(train_outputs, siamese_model.tau, train_labels)  
            
            bce_Loss = nn.BCELoss()(train_outputs, train_labels.float())

            total_loss_term =  hinge_loss_value + bce_Loss
            total_loss_term.backward()
            optimizer.step()
            total_loss += total_loss_term.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        accuracy_pos, accuracy_neg, total_pos, total_neg = val_func(siamese_model, train_loader)
        val_accuracy_pos, val_accuracy_neg, val_total_pos, val_total_neg = val_func(siamese_model, test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.2f}, tau: {siamese_model.tau.item():.2f}, Accuracy: {accuracy_pos:.2f}, {total_pos}--{accuracy_neg:.2f}, {total_neg}  -----  {val_accuracy_pos:.2f}, {val_total_pos}--{val_accuracy_neg:.2f}, {val_total_neg}")
        final_accuracy = val_accuracy_neg + accuracy_pos
        if final_accuracy > best_val_accuracy:
            best_val_accuracy = final_accuracy
            patience_counter = 0
            torch.save(siamese_model.state_dict(), f'save_model_h/classifier_{os.path.basename(__file__).split(".")[0]}_{INITIAL_DATA_LEN}_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                
                torch.save(siamese_model.state_dict(), f'save_model_h/classifier_{os.path.basename(__file__).split(".")[0]}_v2_{INITIAL_DATA_LEN}_{model_name}.pth')
                pass
                break
    
    if val_accuracy_pos < 0.87 or  val_accuracy_neg < 0.9:
        return 1
    return 0


def save_loaders(model_name, test_loader, mean_feature):
    output_dir =f"testdata_h/{model_name}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(mean_feature, f"{output_dir}/mean_feature.pt")
    for batch_idx, (data, labels) in enumerate(test_loader):
        for i in range(data.size(0)):  
            sample = data[i]  
            label = labels[i].item()  
            filename = f"{output_dir}/{label}_{batch_idx}_{i}.pt"
            torch.save(sample, filename)
