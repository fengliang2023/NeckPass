import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import librosa
import librosa.display
from skimage import exposure
import json
import scipy.signal as signal
from io import BytesIO

def load_dict_from_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)


def find_values_within_range(heart_beart,  target_frequency):
    peaks, _ = find_peaks(heart_beart)
    valid_peak_pairs = []
    all_list = []

    # 遍历所有峰对
    for i in range(len(peaks) -1):
        for j in range(i + 1, len(peaks)):
            diff1 = abs(   abs(peaks[j] - peaks[i]) - target_frequency   )
            if diff1 <= 41:
                if  all_list and peaks[i] < all_list[-1]:
                    continue
                else:
                    valid_peak_pairs.append((peaks[i], peaks[j]))
                    all_list.append(peaks[j])
                    continue
    return valid_peak_pairs









def cal_mel(input, filename):
    time_tmp = 2
    fs = 500*time_tmp
    n_mels = 128*2
    
    input = signal.resample_poly(input, up=time_tmp, down=1) # 上采样到1000hz
    S = librosa.feature.melspectrogram(y=input, sr=fs, n_fft=512, hop_length=64, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fs / 2)
    indices = np.where((mel_frequencies >= 0) & (mel_frequencies <= 100))[0]
    S_db = S_db[indices, :]

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='linear', cmap='jet')
    # cbar = plt.colorbar()
    # cbar.set_label('Amplitude (dB)')
    # plt.title('Mel Spectrogram (Upsampled)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.tight_layout()

    # # 将图像保存到 BytesIO 对象
    # plt.savefig(filename, dpi=300, bbox_inches='tight')  # 保存为PNG格式，300 DPI
    # plt.close('all')
    
    # print(filename)

    return S_db

def extract_mel(input_data, new_filename):
    fs=500
    input_data = np.array(input_data)

    cutoff = [0.5 / (0.5 * fs), 40 / (0.5 * fs)]
    b, a = signal.butter(4, cutoff, btype= "bandpass", analog=False)
    result = []
    for i in [1,2]:
        input = input_data[:, i]
        input = signal.filtfilt(b, a, input)
        S = cal_mel(input, new_filename)
        result.append(S)
    np.save(new_filename, np.array(result))
    return 1


def find_segment(input_data, target_ferquency):
    fs=500
    input_data = np.array(input_data)
    result_list = []

    cutoff = [0.8 / (0.5 * fs), 3 / (0.5 * fs)]
    b, a = signal.butter(4, cutoff, btype= "bandpass", analog=False)
    heart_beart_list = []
    for ii in [0,2]:
        ax_data = signal.filtfilt(b, a, input_data[:,ii])
        heart_beart_list.append(ax_data)
    heart_beart = np.sum(heart_beart_list, axis=0)
    peak_pairs = find_values_within_range(heart_beart, target_ferquency)
    # print(peak_pairs)
    # plt.figure(figsize=(12, 6))
    # plt.plot(heart_beart, label='Heart Beat Signal', color='blue')
    # plt.title('Heart Beat Signal with Peaks')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # for start, end in peak_pairs:
    #     plt.plot([start, end], [heart_beart[start], heart_beart[end]], 'g--', linewidth=1)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    return peak_pairs

def extract_training_data(data_list, save_path, save_file_index, target_freq, filename):
    input_data_array = np.array(data_list).T

    scaler = MinMaxScaler() # 
    sub_array = scaler.fit_transform(input_data_array)
    peak_pairs = find_segment(sub_array, target_freq)
    count_num = 0
    for i, (start_index, end_index) in enumerate(peak_pairs):
        new_filename = save_path + f"/{save_file_index}_{i}.npy"
        prex = int( ( 1000 - (end_index - start_index) ) /2 )
        start_index = start_index - prex
        end_index = end_index + prex
        if start_index>0 and end_index < len(input_data_array):
            sub_array = input_data_array[start_index:end_index,:]
            sub_array = scaler.fit_transform(sub_array)
            count_num += extract_mel(sub_array, new_filename)
    return count_num

def main_func(filename, save_path, save_file_index, target_freq):
    # 读取 TXT 文件
    data_list = np.load(filename)
    save_file_num_0 = extract_training_data(data_list, save_path, save_file_index, target_freq, filename)
    return save_file_num_0

def SampleConversion(directory):
    print("Current file name:", os.path.basename(__file__))
    save_directory = str(directory) + "_extract"
    tmp_len1 = len(os.listdir(directory))
    loaded_usr_freq_dir = load_dict_from_json(f'{directory.split("_filtered")[0]}.json')
    for k, input_user in enumerate(os.listdir(directory)):
        input_user_path = os.path.join(directory,input_user)
        save_path = os.path.join(save_directory, input_user)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        txt_files = [os.path.join(input_user_path, item) for item in os.listdir(input_user_path) if item.endswith('.npy')]
        print(f"{k}/{tmp_len1}+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",input_user)
        save_file_index = 0
        target_freq = loaded_usr_freq_dir[input_user]
        for filename in txt_files:
            save_file_num = main_func(filename, save_path, save_file_index, target_freq)
            save_file_index += save_file_num
        print("-----total、final file:", len(txt_files), save_file_index,"\n")


if __name__ == "__main__":
    directory = "op" + "_filtered"
    SampleConversion(directory)




