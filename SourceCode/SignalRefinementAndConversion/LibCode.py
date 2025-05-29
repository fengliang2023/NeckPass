import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import librosa
import librosa.display
from scipy.signal import stft

def extract_data(parts):
    data = {}
    num = 21
    data['timestamp'] = float(parts[num].replace("Timestamp:",""))
    if data['timestamp'] < 0.001:
        return {}

    num = 3
    data['angular_acceleration_x'] = float(parts[num].replace("x:",""))
    data['angular_acceleration_y'] = float(parts[num+1].replace("y:",""))
    data['angular_acceleration_z'] = float(parts[num+2].replace("z:",""))
    
    num = 8
    data['angular_velocity_x'] = float(parts[num].replace("x:",""))
    data['angular_velocity_y'] = float(parts[num+1].replace("y:",""))
    data['angular_velocity_z'] = float(parts[num+2].replace("z:",""))
    
    num = 13
    data['linear_acceleration_x'] = float(parts[num].replace("x:",""))
    data['linear_acceleration_y'] = float(parts[num+1].replace("y:",""))
    data['linear_acceleration_z'] = float(parts[num+2].replace("z:",""))
    
    num = 18
    data['linear_velocity_x'] = float(parts[num].replace("x:",""))
    data['linear_velocity_y'] = float(parts[num+1].replace("y:",""))
    data['linear_velocity_z'] = float(parts[num+2].replace("z:",""))
    
    return data

def get_data_from_data_list(data_list):
    timestamps = [data['timestamp'] for data in data_list]
    angular_acceleration_x = [data['angular_acceleration_x'] for data in data_list]
    angular_acceleration_y = [data['angular_acceleration_y'] for data in data_list]
    angular_acceleration_z = [data['angular_acceleration_z'] for data in data_list]
    angular_acceleration = np.sqrt(np.array(angular_acceleration_x)**2 + np.array(angular_acceleration_y)**2 + np.array(angular_acceleration_z)**2)

    angular_velocity_x = [data['angular_velocity_x'] for data in data_list]
    angular_velocity_y = [data['angular_velocity_y'] for data in data_list]
    angular_velocity_z = [data['angular_velocity_z'] for data in data_list]
    angular_velocity = np.sqrt(np.array(angular_velocity_x)**2 + np.array(angular_velocity_y)**2 + np.array(angular_velocity_z)**2)
    linear_acceleration_x = [data['linear_acceleration_x'] for data in data_list]
    linear_acceleration_y = [data['linear_acceleration_y'] for data in data_list]
    linear_acceleration_z = [data['linear_acceleration_z'] for data in data_list]
    linear_acceleration = np.sqrt(np.array(linear_acceleration_x)**2 + np.array(linear_acceleration_y)**2 + np.array(linear_acceleration_z)**2)
    linear_velocity_x = [data['linear_velocity_x'] for data in data_list]
    linear_velocity_y = [data['linear_velocity_y'] for data in data_list]
    linear_velocity_z = [data['linear_velocity_z'] for data in data_list]
    linear_velocity = np.sqrt(np.array(linear_velocity_x)**2 + np.array(linear_velocity_y)**2 + np.array(linear_velocity_z)**2)

    result_list = [angular_acceleration, angular_velocity, linear_acceleration, linear_velocity, timestamps]
    return result_list

def get_data_from_txt(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data_list = [[],[],[],[],[]]
    type_list = ["HEAD", "LEFT", "RIGHT"]
    j_list = [1,2]
    for j in j_list:
        tmp_list = []
        for line in lines:
            if type_list[j] in line:
                parts = line.strip().split()
                if len(parts) < 18:
                    continue
                data = extract_data(parts)
                if not len(data):
                    continue
                tmp_list.append(data)
        result_list = get_data_from_data_list (tmp_list)
        for i in range(len(data_list)):
            data_list[i].append(result_list[i])
    return data_list