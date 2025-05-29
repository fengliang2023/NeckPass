import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
from ast_model import ASTModel
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import zoom
import shutil

def calculate_accuracy_per_class(SaveDatasetPath, data_loader, model, class_list):
    num_classes = len(class_list)
    model.eval()  
    correct = np.zeros(num_classes)  
    total = np.zeros(num_classes)  

    with torch.no_grad():  
        j =0
        for inputs, labels in data_loader:
            j +=1
            inputs = inputs.float()

            encoder_output, outputs = model(inputs)
            for i in range(encoder_output.shape[0]):  
                sample_output = encoder_output[i].cpu().numpy()
                name_fir = os.path.join(SaveDatasetPath, class_list[labels[i]])
                if not os.path.exists(name_fir):
                    os.makedirs(name_fir)
                filename = os.path.join(name_fir,  f"{j}_{i}.npy")
                np.save(filename, sample_output)
            _, predicted = torch.max(outputs.data, 1)  
            
            for label in range(num_classes):
                total[label] += (labels == label).sum().item()  
                correct[label] += ((predicted == label) & (labels == label)).sum().item()  
    class_accuracy = correct / total
    return class_accuracy, total


def get_data(input_path, all_usr):
    all_usr_list = [all_usr]
    data_list = []
    label_list = []
    for usr_name in all_usr_list:
        usr_path = os.path.join(input_path, usr_name)
        usr_data_list = []
        for file_name in os.listdir( usr_path ):
            file_path = os.path.join(usr_path, file_name)
            data = np.load(file_path)
            reshaped_array = np.concatenate(data, axis=0).transpose()
            usr_data_list.append(reshaped_array)
        data_list += usr_data_list
        label_list += [usr_name]*len(usr_data_list)
    data_tensor = torch.FloatTensor(np.array(data_list))
    if np.any(np.isnan(data_tensor.numpy())):
        print("Data contains NaN values")
    if np.any(np.isinf(data_tensor.numpy())):
        print("Data contains infinite values")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(label_list)
    label_tensor = torch.LongTensor(encoded_labels)
    dataset = TensorDataset(data_tensor, label_tensor)
    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_fdim=reshaped_array.shape[1]
    input_tdim=reshaped_array.shape[0]
    return data_loader,label_encoder, input_fdim, input_tdim



def load_model(ModelPath, label_dim, input_fdim, input_tdim):
    model = ASTModel(label_dim=label_dim, input_fdim = input_fdim, input_tdim = input_tdim)
    model.load_state_dict(torch.load(ModelPath))
    model.eval()
    return model

def ExtractRepresentation(DatasetPath, SaveDatasetPath, ModelPath, label_dim = 9):
    for dataset in os.listdir(DatasetPath):
        datasetsubpath = os.path.join(DatasetPath, dataset)
        savedatase = os.path.join(SaveDatasetPath, dataset)
        if os.path.exists(savedatase):
            shutil.rmtree(savedatase)
        os.makedirs(savedatase)
        all_usr = os.listdir(datasetsubpath)
        print("-------------- start training ------------")
        for usr_name in all_usr:
            test_loader, label_encoder, input_fdim, input_tdim  = get_data(datasetsubpath, usr_name)
            num_classes =len(label_encoder.classes_)
            Modelfile= os.listdir(ModelPath)[0]
            modelfilepath = os.path.join(ModelPath, Modelfile)
            model = load_model(modelfilepath, label_dim, input_fdim, input_tdim)
            
            class_list = label_encoder.classes_
            test_class_accuracy, total = calculate_accuracy_per_class(savedatase, test_loader, model, class_list)
            for i, accuracy in enumerate(test_class_accuracy):
                class_name = label_encoder.classes_[i]  
                print(f"{i} {class_name:<15}: {accuracy:.3f}  {int(total[i])}")
