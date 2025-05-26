import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from helpers.denoise import de_noise_time_series
from helpers.data_split import split_data
from helpers.create_image import create_images_parallel
from helpers.create_csv import make_csv
from helpers.create_image_full import _create_full_imgs

class DataBuilder:
    def __init__(self, data_folder, out_path, column_names, csv_path):


        self.data_folder = data_folder
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        self.column_names = column_names
        self.src_csv_path = csv_path
        self.new_csv_path = "/".join(data_folder.split('/')[:2]) + "/"

    def read_txt(self) -> list[str]:
        try:
            all_files = os.listdir(self.data_folder)
            txt_files = [file for file in all_files if file.endswith(".txt")]
            return txt_files
        except FileNotFoundError:
            print(f"Error: The data folder '{self.data_folder}' was not found.")
            return []
        
    def scaler(self, df: pd.DataFrame, feature_range: tuple = (0, 1)) -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_values = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
        return scaled_df

    def check_data(self, data_path):
        good_nums = []
        error_nums = []
        if not os.path.exists(data_path) or not os.path.isdir(data_path):
            print(f"Warning: Data path '{data_path}' does not exist or is not a directory.")
            return None
        
        all_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        # print("All:", all_folders)
        for folder in all_folders:
            folder_path = os.path.join(data_path, folder)
            # print("Folder Path:", folder_path)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                folder_count = len([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
                if folder_count < 27 or folder_count > 27:
                    # print("FC:", folder_count)
                    error_nums.append(folder)
                else:
                    good_nums.append(folder)
            else:
                error_nums.append(folder)
        return good_nums, error_nums
    
    def build_data(self):
        files = self.read_txt()
        # files = files[:2] # for testing
        for file in files:
            data_path = self.data_folder + file
            time_df = pd.read_csv(data_path, sep=' ', names=self.column_names)
            
            # Denoising
            print(time_df.head(), time_df.shape)
            time_df = de_noise_time_series(
                time_df, 
                swing_threshold_factor=2.0,
                window_size=85,
                min_quiet_duration=40)
            
            # Scaling <-- not nessary
            time_df = self.scaler(time_df)
            print(time_df.head(), time_df.shape)

            # Spliting 
            time_dfs = split_data(time_df)
            # for df in time_dfs:
            #     print(df.shape)

            # Create Images
            print(self.out_path + file.split('.')[0] + '/')
            create_images_parallel(
                dfs = time_dfs, 
                outpath = self.out_path + file.split('.')[0] + '/', 
                target_width_px=448, 
                target_height_px=224, 
                dpi=100,
                num_processes = 6,
            )

        # Create csv
        make_csv(img_path=self.out_path, src_csv_path=self.src_csv_path)

    def build_data_full(self):
        df = pd.read_csv(self.src_csv_path)
        files = df['unique_id'].tolist()
        print('Files:', files)
        _create_full_imgs(data_path=self.data_folder, files=files, outpath=self.out_path)
        return


if __name__ == "__main__":
    data_folder = "data/39_Training_Dataset/train_data/"
    out_path = "data/train/images/"
    csv_path = "data/39_Training_Dataset/train_info.csv"
    column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    train_builder = DataBuilder(data_folder, out_path, column_names, csv_path)
    train_builder.build_data()
    print(train_builder.check_data(out_path))