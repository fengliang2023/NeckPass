import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
from ast_model import ASTModel
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import zoom
import shutil

def calculate_accuracy(data_loader, model):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():  
        for inputs, labels in data_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            _,outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  
    return correct / total  

def calculate_accuracy_per_class(data_loader, model, num_classes):
    model.eval()  
    correct = np.zeros(num_classes) 
    total = np.zeros(num_classes)

    with torch.no_grad():  
        for inputs, labels in data_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) 
            
            for label in range(num_classes):
                total[label] += (labels == label).sum().item() 
                correct[label] += ((predicted == label) & (labels == label)).sum().item()  

    class_accuracy = correct / total
    return class_accuracy, total

def num_data_augmentation(usr_data_list, len_tmp = 100):
    if len(usr_data_list) < len_tmp:
        stop_len = int( len(usr_data_list) / len_tmp ) + 1
        data_list = usr_data_list * stop_len
        usr_data_list = data_list[:len_tmp]
    return usr_data_list

def avg_data_augmentation(usr_data_list, len_tmp = 50):
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

def get_data(input_path, all_neg_usr, label):
    all_usr = [item for item in os.listdir(input_path) if item not in all_neg_usr]
    data_list = []
    label_list = []
    for usr_name in all_usr:
        usr_path = os.path.join(input_path, usr_name)
        usr_data_list = []
        for file_name in os.listdir( usr_path ):
            file_path = os.path.join(usr_path, file_name)
            data = np.load(file_path)
            
            
            reshaped_array = np.concatenate(data, axis=0).transpose()
            usr_data_list.append(reshaped_array)
        if label == "train":
            usr_data_list = avg_data_augmentation(usr_data_list, 100)
        if label == "test":
            usr_data_list = avg_data_augmentation(usr_data_list, 10)
        data_list += usr_data_list
        label_list += [usr_name]*len(usr_data_list)
    data_tensor = torch.FloatTensor(np.array(data_list))
    print("----223-----", len(data_list))

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(label_list)
    label_tensor = torch.LongTensor(encoded_labels)
    dataset = TensorDataset(data_tensor, label_tensor)
    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_fdim=reshaped_array.shape[1]
    input_tdim=reshaped_array.shape[0]
    return data_loader,label_encoder, input_fdim, input_tdim



def TrainTm(DatasetPath, SaveModelPath):
    train_dir = f'{DatasetPath}/train'
    test_dir = f'{DatasetPath}/test'
    save_model_name = f"Tm"
    all_neg_usr = os.listdir(train_dir)[-1:]    # the attackers

    if os.path.exists(SaveModelPath):
        shutil.rmtree(SaveModelPath)
    os.makedirs(SaveModelPath, exist_ok=True)
    train_loader, label_encoder,input_fdim, input_tdim  = get_data(train_dir, all_neg_usr, "train")
    test_loader, label_encoder,input_fdim, input_tdim  = get_data(test_dir, all_neg_usr, "test")
    print("-------------- start training ------------", len(train_loader))

    num_epochs = 1
    learning_rate = 0.0001
    num_classes=len(label_encoder.classes_)
    model = ASTModel(label_dim=num_classes, input_fdim = input_fdim, input_tdim = input_tdim)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=5e-7, betas=(0.95, 0.999))
    best_val_accuracy = 0
    patience = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_accuracy = calculate_accuracy(train_loader, model)
        test_accuracy = calculate_accuracy(test_loader, model)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f} train/test: {train_accuracy:.3f} {test_accuracy:.3f}')
        final_accuracy = test_accuracy
        if final_accuracy > best_val_accuracy:
            best_val_accuracy = final_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f'{SaveModelPath}/{save_model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping") 
                break
    test_class_accuracy, total = calculate_accuracy_per_class(test_loader, model, num_classes)
    for i, accuracy in enumerate(test_class_accuracy):
        class_name = label_encoder.classes_[i]  
        print(f"{i} {class_name:<15}: {accuracy:.3f}  {int(total[i])}")