import os
import pandas as pd
import shutil


target_folder = 'landpointCollection2'
os.makedirs(target_folder, exist_ok=True)

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('_land_point.csv'):
            parent_folders = root.split(os.sep)[-2:]  # 获取最近的两个父文件夹
            if len(parent_folders) == 2:
                second_parent_folder_name = parent_folders[0]  # 第二近的父文件夹
            else:
                second_parent_folder_name = parent_folders[0]  # 如果只有一个父文件夹


            source_file = os.path.join(root, file)
            new_file_name = f"{second_parent_folder_name}_{file}"
            shutil.copy(source_file, os.path.join(target_folder, new_file_name))
            df = pd.read_csv(os.path.join(target_folder, new_file_name))
            df.columns.values[0] = 'X'  # 修改第一列列名
            df.columns.values[1] = 'Y'  # 修改第二列列名

            df.to_csv(os.path.join(target_folder, new_file_name), index=False)

print("文件复制和列名修改完成。")
