import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import os

def load_df(folder_path: str, csv_name: str):
    df = pd.read_csv(f"{folder_path}{csv_name}")
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, df, data_files_path='.', split='train', max_seq_len=5000, normalize=True):
        self.data_frame = df
        self.data_files_path = data_files_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.normalize = normalize

    def __len__(self):
        return len(self.data_frame)
    
    def _normalize_time_series(self, time_series_np):
        """
        Normalizes the time series data using Z-score normalization (per-feature/column within the file).
        """
        if time_series_np.size == 0: # Handle empty array
            return time_series_np.astype(np.float32)

        mean = np.mean(time_series_np, axis=0, dtype=np.float32)
        std = np.std(time_series_np, axis=0, dtype=np.float32)
        
        # Avoid division by zero for features with zero standard deviation
        std_no_zero = np.where(std < 1e-8, 1.0, std) # Use a small epsilon for float comparison
        
        normalized_time_series_np = (time_series_np - mean) / std_no_zero
        return normalized_time_series_np.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]

        file_id = row['unique_id']
        file_name = f"{file_id}.txt"
        file_path = os.path.join(self.data_files_path, file_name)

        time_series_np = np.loadtxt(file_path, dtype=np.float32)
        
        # If loadtxt reads a 1D array (e.g. single column file), reshape to (N, 1)
        if time_series_np.ndim == 1:
            time_series_np = time_series_np[:, np.newaxis]

        # Ensure it's float32 before normalization
        time_series_np = time_series_np.astype(np.float32)

        # --- Normalization Step ---
        if self.normalize:
            time_series_np = self._normalize_time_series(time_series_np)
        # --- End Normalization ---

        current_seq_len = time_series_np.shape[0]
        num_features = time_series_np.shape[1]

        if current_seq_len < self.max_seq_len:
            padding_size = self.max_seq_len - current_seq_len
            # Pre-Pad ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
            time_series_np = np.pad(time_series_np, 
                                    ((padding_size, 0), (0, 0)), 
                                    mode='constant', 
                                    constant_values=0)
                                    
        elif current_seq_len > self.max_seq_len:
            # Truncate from the beginning
            time_series_np = time_series_np[:self.max_seq_len, :]
        # If current_seq_len == self.max_seq_len, do nothing.
            
        time_series_tensor = torch.from_numpy(time_series_np)

        if self.split == 'train' or self.split == 'val':
            swing_mode = float(row['mode'])
            gender = float(row['gender'])
            hand = float(row['hold racket handed'])
            years = float(row['play years'])
            level = float(row['level']) - 2.0

            other_features_tensor = torch.tensor([
                swing_mode, gender, hand, years, level
            ], dtype=torch.float32)

            return time_series_tensor.transpose(-2, -1), other_features_tensor
        return time_series_tensor.transpose(-2, -1)
    
def load_data(max_seq_len=5000):
    train_folder_path = "data/39_Training_Dataset/"
    train_csv_name = "train_info.csv"
    train_data_files_dir = os.path.join(train_folder_path, 'train_data')

    test_folder_path = "data/39_Test_Dataset/"
    test_csv_name = "test_info.csv"
    test_data_files_dir = os.path.join(test_folder_path, 'test_data')

    train_val_info_df = load_df(train_folder_path, train_csv_name)
    test_info_df = load_df(test_folder_path, test_csv_name)

    full_train_val_dataset = TimeSeriesDataset(
        df=train_val_info_df,
        data_files_path=train_data_files_dir,
        split='train',
        max_seq_len=max_seq_len
    )

    total_len = len(full_train_val_dataset)
    
    train_len = int(total_len*0.8)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(42) # Ensures reproducibility of split
        
    train_dataset, val_dataset = random_split(
        full_train_val_dataset,
        [train_len, val_len],
        generator=generator
    )

    test_dataset = TimeSeriesDataset(
        df=test_info_df,
        data_files_path=test_data_files_dir,
        split='test',
        max_seq_len=max_seq_len
    )

    return train_dataset, val_dataset, test_dataset
    
if __name__ == '__main__':
    train_ds, val_ds, test_ds = load_data()
        
    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")
    print(f"Number of test samples: {len(test_ds)}")

    if len(train_ds) > 0:
        print("\nSample from training dataset:")
        time_series, other_features = train_ds[0]
        print(f"Time series shape: {time_series.shape}")
        print(f"Other features: {other_features}")
        print(f"Time series dtype: {time_series.dtype}")
        print(f"Other features dtype: {other_features.dtype}")

    if len(test_ds) > 0:
        print("\nSample from test dataset:")
        time_series_test = test_ds[0]
        print(f"Test time series shape: {time_series_test.shape}")
        print(f"Test time series dtype: {time_series_test.dtype}")