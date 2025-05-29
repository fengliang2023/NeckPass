import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks
import tkinter as tk
from LibCode import get_data_from_txt
import math
import json
import shutil

def save_dict_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def cal_freq(data_list):
    sample_rate = 500
    cutoff = [0.8 / (0.5 * sample_rate), 3 / (0.5 * sample_rate)]
    b, a = signal.butter(4, cutoff, btype= "bandpass", analog=False)

    normalized_arrays = []
    for x in [2, 0]:
        ax_data = data_list[x]
        data_filtered = signal.filtfilt(b, a, ax_data)
        min_val = np.min(data_filtered)
        max_val = np.max(data_filtered)
        if max_val > min_val:
            normalized_data = ( (data_filtered - min_val) / (max_val - min_val) )*20 -10
        else:
            normalized_data = np.zeros_like(data_filtered)
        normalized_arrays.append(normalized_data)
    ax_data = np.sum(normalized_arrays, axis=0)
    peaks, _ = find_peaks(ax_data)
    return len(peaks)

def filter_by_value(data_list):
    sample_rate = 500
    cutoff = [0.8 / (0.5 * sample_rate), 3 / (0.5 * sample_rate)]
    b, a = signal.butter(4, cutoff, btype= "bandpass", analog=False)
    
    for item in [0,1,2]:
        data_filtered_0 = signal.filtfilt(b, a, data_list[item])
        if np.max(data_filtered_0) > 1:
            return 0
        if np.max(data_filtered_0) < 0.005:
            return 0

def filter_by_peak(num_peak, peak_list):
    if (num_peak != peak_list[1]) and abs(num_peak -  peak_list[0] ) > 3.5:
        return 0
    else:
        return 1

def SignalRefinement(directory):
    print("+++++ Current file name:+++++++", os.path.basename(__file__))
    folder_path_list = os.listdir(directory)
    usr_freq_dir = {}
    for jj,usr_name in enumerate(folder_path_list):
        usr_directory = os.path.join(directory, usr_name) 
        all_items = os.listdir(usr_directory)
        txt_files = [os.path.join(usr_directory, item) for item in all_items if item.endswith('.txt')]
        print(f"{jj}/{len(folder_path_list)} ++++++++++++++++++++++++++++++++++++++++++++++++++++++++",usr_directory.split("/")[-1])
        tmp_value = 0
        num_peak_list= []
        final_txt_files = []
        all_save_file = []
        for filename in txt_files:
            data_list = get_data_from_txt(filename)
            for left_or_right in [0,1]:
                data_item = [data_list[jj][left_or_right] for jj in [0,1,2]]
                result_value = filter_by_value(data_item)
                if result_value == 0:
                    continue
                all_save_file.append(data_item)

        file_peak_list = []
        cal_peaks= []
        for data_item in all_save_file:
            peaks = cal_freq(data_item)
            file_peak_list.append(peaks)
            if abs(peaks - 20) < 10:
                cal_peaks.append(peaks)
        if len(cal_peaks) == 0:
            print("+++++ no valid data +++++++")
            continue
        avg_peak = sum(cal_peaks) / len(cal_peaks)

        all_save_file_2 = []
        for ii in range(len(all_save_file)):
            file_peak = file_peak_list[ii]
            if abs(file_peak - avg_peak) < 10:
                all_save_file_2.append(all_save_file[ii])

        save_path = os.path.join(directory + "_filtered", usr_name)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        for i, array in enumerate(all_save_file_2):
            file_path = os.path.join(save_path, f'array_{i}.npy')
            np.save(file_path, array)
        usr_freq_dir[usr_name] = 500*10 / (avg_peak / 2)
        print("-----total、final file:", len(txt_files), i, avg_peak,avg_peak, usr_freq_dir[usr_name],"\n")
    save_dict_to_json(usr_freq_dir, f'{directory}.json')

if __name__ == "__main__":
    directory = "family"
    SignalRefinement(directory)

