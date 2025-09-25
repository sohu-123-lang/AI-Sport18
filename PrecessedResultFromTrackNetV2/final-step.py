import os
import pandas as pd
import shutil


target_folder = 'landpointCollection'
os.makedirs(target_folder, exist_ok=True)

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('_land_point.csv'):
            parent_folder_name = os.path.basename(root)
            source_file = os.path.join(root, file)
            new_file_name = f"{parent_folder_name}_{file}"
            shutil.copy(source_file, os.path.join(target_folder, new_file_name))
            df = pd.read_csv(os.path.join(target_folder, new_file_name))
            df.columns.values[0] = 'X'  # 
            df.columns.values[1] = 'Y'  #
            
            df.to_csv(os.path.join(target_folder, new_file_name), index=False)

print("文件复制和列名修改完成。")