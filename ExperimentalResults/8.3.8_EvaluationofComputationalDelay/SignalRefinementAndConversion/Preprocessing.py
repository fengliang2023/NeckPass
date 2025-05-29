import os
import glob

def count_head_occurrences(file_path, find_string):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.count(find_string)

def main(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    tmp_value = 0
    for txt_file in txt_files:
        head_count_0 = count_head_occurrences(txt_file, 'LEFT')
        head_count_1 = count_head_occurrences(txt_file, 'RIGHT')
        if head_count_0 < 4800 or head_count_1 < 4800:
            tmp_value += 1
            print(f"{txt_file} : LEFT/RIGHT: {head_count_0} : {head_count_1} ")
            os.remove(txt_file)
    print("-----total、final file:", len(txt_files), len(txt_files) -tmp_value,"\n")

def Preprocessing(directory):
    folder_path_list = os.listdir(directory)
    folder_path_list = [os.path.join(directory, i) for i in folder_path_list]
    for folder_path in folder_path_list:
        print("++++++++++++++++++++++++++++++++++【 start 】++++++++++++++++++++++++++++++++++",folder_path.split("/")[-1])
        main(folder_path)



if __name__ == "__main__":
    directory = "l60r70"
    Preprocessing(directory)