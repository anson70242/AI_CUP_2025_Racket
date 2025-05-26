import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_time_data(folder_path: str = "data/39_Training_Dataset/train_data/",
                   file_name: str = "1.txt") -> pd.DataFrame:
    """Reads time series data from a specified file."""
    data_path = folder_path + file_name
    column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    try:
        time_data = pd.read_csv(data_path, sep=' ', names=column_names)
        return time_data
    except FileNotFoundError:
        # This exception will be caught by the caller in the __main__ block
        raise FileNotFoundError(f"Error: The file {data_path} was not found.")
    except Exception as e:
        # General exception for other read errors (e.g., malformed file)
        print(f"An error occurred while reading {data_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on other errors

def _calculate_sensor_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds accelerometer, gyroscope, and combined magnitudes to the DataFrame."""
    df['Amag'] = df[['Ax', 'Ay', 'Az']].abs().sum(axis=1)
    df['Gmag'] = df[['Gx', 'Gy', 'Gz']].abs().sum(axis=1)
    df['CombinedMag'] = df['Amag'] + df['Gmag']
    return df

def _find_activity_start_index(
    combined_mag_series: pd.Series,
    baseline_window_duration: int,
    swing_threshold_factor: float,
    activity_consistency_check_len: int
) -> int:
    """
    Identifies the start index of significant activity based on baseline noise.

    Args:
        combined_mag_series (pd.Series): Series of combined sensor magnitudes.
        baseline_window_duration (int): Number of initial samples to calculate baseline statistics.
        swing_threshold_factor (float): Factor to multiply baseline std deviation for threshold.
        activity_consistency_check_len (int): Min number of subsequent samples to confirm sustained activity.

    Returns:
        int: The index where significant activity is deemed to start. 0 if no clear start found.
    """
    if len(combined_mag_series) < baseline_window_duration:
        return 0 # Not enough data to determine a baseline

    baseline_mean = combined_mag_series.iloc[:baseline_window_duration].mean()
    baseline_std = combined_mag_series.iloc[:baseline_window_duration].std()
    # Handle cases where std might be zero (e.g. flat signal) by adding a small epsilon or ensuring it's not zero
    if pd.isna(baseline_std) or baseline_std == 0:
        baseline_std = combined_mag_series.mean() * 0.01 # A small fraction of mean as fallback
        if baseline_std == 0: # If mean is also 0
             baseline_std = 1e-6


    noise_activity_threshold = baseline_mean + swing_threshold_factor * baseline_std

    activity_start_index = 0
    for i in range(len(combined_mag_series)):
        if combined_mag_series.iloc[i] > noise_activity_threshold:
            # Check if activity is sustained
            look_ahead_end = min(len(combined_mag_series), i + activity_consistency_check_len)
            if activity_consistency_check_len > 0 and look_ahead_end > i:
                if combined_mag_series.iloc[i : look_ahead_end].mean() > noise_activity_threshold:
                    activity_start_index = i
                    break
            elif activity_consistency_check_len == 0: # If no consistency check needed, first point is enough
                 activity_start_index = i
                 break
    return activity_start_index

def de_noise_time_series(
    df: pd.DataFrame,
    swing_threshold_factor: float = 1.5,
    window_size: int = 85,
    min_quiet_duration: int = 85
) -> pd.DataFrame:
    """
    Removes initial noise from time series data of racket swings.

    Args:
        df (pd.DataFrame): DataFrame with sensor data.
        swing_threshold_factor (float): Factor for threshold calculation.
        window_size (int): Window size for rolling statistics and activity check.
        min_quiet_duration (int): Minimum initial samples to consider for baseline.

    Returns:
        pd.DataFrame: DataFrame with initial noise removed, or original if no noise detected.
    """
    if df.empty or len(df) < min_quiet_duration:
        return df.copy() # Return a copy for consistency

    # Work on a temporary DataFrame to avoid modifying the original df with helper columns
    temp_df = df.copy()
    temp_df = _calculate_sensor_magnitudes(temp_df)

    # Determine the actual window to use for baseline calculation
    baseline_calc_window = min(len(temp_df), max(window_size, min_quiet_duration))
    # Determine consistency check length, e.g., a quarter of the window size
    consistency_check_len = max(1, window_size // 4)

    trim_index = _find_activity_start_index(
        temp_df['CombinedMag'],
        baseline_calc_window,
        swing_threshold_factor,
        consistency_check_len
    )

    denoised_df = df.copy() # Default to returning a copy of the original

    # Heuristic: noise shouldn't be too long (e.g., > 50% of data) and trim_index must be > 0
    if 0 < trim_index < len(temp_df) * 0.5:
        avg_mag_before_trim = temp_df['CombinedMag'].iloc[:trim_index].mean()
        # Check a window after the potential trim point (e.g., 2*window_size)
        avg_mag_after_trim_window_end = min(len(temp_df), trim_index + window_size * 2)
        avg_mag_after_trim = temp_df['CombinedMag'].iloc[trim_index:avg_mag_after_trim_window_end].mean()

        # Only trim if there's a significant increase in activity
        # Use a slightly adjusted factor for this comparison
        if avg_mag_after_trim > avg_mag_before_trim * (1 + swing_threshold_factor / 2.0):
            denoised_df = df.iloc[trim_index:].reset_index(drop=True)
            # print(f"Initial noise detected. Trimming first {trim_index} samples.")
        # else:
            # print("No significant difference in activity found. No trimming.")
    # else:
        # print("No initial noise period identified for trimming or trim_index out of heuristic bounds.")

    # Helper columns ('Amag', 'Gmag', 'CombinedMag') were on temp_df,
    # denoised_df is created from the original df or its slice, so it's already clean.
    return denoised_df

def plot_sensor_data(df: pd.DataFrame, title_suffix: str = "Sensor Data"):
    """
    Plots accelerometer and gyroscope data from the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame. Cannot plot.")
        return
    if df.empty:
        print("DataFrame is empty. Nothing to plot.")
        return

    accel_cols = ['Ax', 'Ay', 'Az']
    gyro_cols = ['Gx', 'Gy', 'Gz']

    # Check for column existence
    if not any(col in df.columns for col in accel_cols + gyro_cols):
        print(f"Error: None of the expected sensor columns ({accel_cols + gyro_cols}) found.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Sensor Data Analysis - {title_suffix}', fontsize=16)

    # Plot Accelerometer Data
    ax1 = axes[0]
    for col in accel_cols:
        if col in df.columns:
            ax1.plot(df.index, df[col], label=col)
        else:
            print(f"Warning: Accelerometer column '{col}' not found.")
    ax1.set_title('Accelerometer Data')
    ax1.set_ylabel('Acceleration (units)')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot Gyroscope Data
    ax2 = axes[1]
    for col in gyro_cols:
        if col in df.columns:
            ax2.plot(df.index, df[col], label=col)
        else:
            print(f"Warning: Gyroscope column '{col}' not found.")
    ax2.set_title('Gyroscope Data')
    ax2.set_ylabel('Angular Velocity (units)')
    ax2.set_xlabel('Sample Index (Time)')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(f"{title_suffix.replace(' ', '_').lower()}_plot.png") # Save with a descriptive name
    plt.show()


if __name__ == '__main__':
    # Define file path components here for easier reference in error messages
    data_folder = "data/39_Training_Dataset/train_data/"
    data_file = "384.txt" # As specified in the original read_time function
    full_data_path = data_folder + data_file

    try:
        print("--- Testing with actual data file ---")
        original_time_data = read_time_data(folder_path=data_folder, file_name=data_file)

        if not original_time_data.empty:
            print(f"Original data length: {len(original_time_data)}")

            # Plot original data for comparison (optional, but good for seeing 'before')
            # plot_sensor_data(original_time_data, title_suffix="Original Raw Data")

            denoised_time_data = de_noise_time_series(
                original_time_data, # No .copy() needed here as de_noise_time_series handles it
                swing_threshold_factor=2.0,
                window_size=85,
                min_quiet_duration=40
            )
            print(f"Denoised data length: {len(denoised_time_data)}")

            # Plot denoised data
            plot_sensor_data(denoised_time_data, title_suffix="Denoised Data")

            if len(denoised_time_data) < len(original_time_data):
                trimmed_samples = len(original_time_data) - len(denoised_time_data)
                print(f"Trimmed {trimmed_samples} samples from the beginning.")
            else:
                print("No noise trimmed, or swing started immediately/no significant quiet period detected.")
        else:
            print(f"Data loaded from {full_data_path} is empty. Skipping processing.")

    except FileNotFoundError:
        # The FileNotFoundError from read_time_data will be caught here
        print(f"Critical Error: The data file '{full_data_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the main process: {e}")