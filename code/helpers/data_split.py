import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# --- Configuration ---
TARGET_COLUMN_NAME = 'Ax' # The column to analyze for peaks
EXPECTED_NUM_SWINGS = 27    # The number of swings expected to find

def load_and_prepare_signal(dataframe: pd.DataFrame, column_name: str) -> tuple[np.array, np.array]:
    """
    Loads the specified column from the DataFrame and prepares it for peak detection.

    Args:
        dataframe (pd.DataFrame): The input DataFrame (e.g., scaled_data).
        column_name (str): The name of the column containing the signal.

    Returns:
        tuple: (signal_array, time_index) or (None, None) if column is not found.
               signal_array is a NumPy array of the signal.
               time_index is a NumPy array representing the sample indices.
    """
    if column_name not in dataframe.columns:
        print(f"Error: Column '{column_name}' not found in the DataFrame.")
        print(f"Available columns are: {dataframe.columns.tolist()}")
        return None, None

    signal_array = dataframe[column_name].to_numpy()
    time_index = np.arange(len(signal_array))
    print(f"Successfully loaded signal from column '{column_name}'. Length: {len(signal_array)} samples.")
    return signal_array, time_index

def detect_swing_peaks(signal_array, num_expected_swings, column_name_for_print="signal"):
    """
    Detects peaks in the signal, attempting to find the specified number of swings.

    Args:
        signal_array (np.ndarray): The input signal.
        num_expected_swings (int): The desired number of peaks (swings).
        column_name_for_print (str): Name of the signal column for printing messages.


    Returns:
        np.ndarray: An array of indices for the detected peaks. Returns empty array if issues.
    """
    if signal_array is None or len(signal_array) == 0:
        print("Error: Signal array is empty or None. Cannot detect peaks.")
        return np.array([])

    # --- Tune these peak detection parameters carefully! ---
    median_val = np.median(signal_array)
    std_val = np.std(signal_array)

    # Params gere!!!!!!!!!!!!!!! Most important
    ###
    ###
    # Initial sensible defaults - these almost ALWAYS need tuning for specific data
    peak_height_param = median_val + std_val * 0.5 # Example: 1 standard deviation above median
    # Ensure distance is at least 1
    peak_distance_param = max(1, len(signal_array) // (num_expected_swings + 85)) # Heuristic
    peak_prominence_param = std_val * 1 # Example: half a standard deviation

    print(f"\nAttempting peak detection on '{column_name_for_print}' with parameters:")
    print(f"  - Min Height: {peak_height_param:.2f}")
    print(f"  - Min Distance: {peak_distance_param}")
    print(f"  - Min Prominence: {peak_prominence_param:.2f}")

    detected_indices, properties = find_peaks(
        signal_array,
        height=peak_height_param,
        distance=peak_distance_param,
        prominence=peak_prominence_param
    )
    print(f"Initially detected {len(detected_indices)} peaks in '{column_name_for_print}'.")

    final_peak_indices = detected_indices

    if len(detected_indices) == 0:
        print(f"No peaks detected in '{column_name_for_print}' with current parameters. Try adjusting them.")
        return np.array([])
    elif len(detected_indices) > num_expected_swings:
        print(f"More than {num_expected_swings} peaks found. Filtering by prominence...")
        if 'prominences' in properties and properties['prominences'] is not None and len(properties['prominences']) > 0:
            prominences = properties['prominences']
            most_prominent_sorted_indices = np.argsort(prominences)[::-1][:num_expected_swings]
            final_peak_indices = np.sort(detected_indices[most_prominent_sorted_indices])
            print(f"Selected {len(final_peak_indices)} most prominent peaks.")
        else:
            print("Prominence data not available for filtering. Using the first N detected peaks as a fallback.")
            final_peak_indices = detected_indices[:num_expected_swings]
    elif len(detected_indices) < num_expected_swings:
        print(f"Fewer than {num_expected_swings} peaks found ({len(detected_indices)}).")
        print("Consider adjusting peak detection parameters (e.g., lower height/prominence, smaller distance).")
        # We'll proceed with the peaks found.
    else:
        print(f"Successfully detected {len(detected_indices)} peaks, matching expected number.")

    return final_peak_indices

def segment_data_around_peaks(full_dataframe, peak_indices, num_expected_segments):
    """
    Segments the full DataFrame around the detected peak indices.

    Args:
        full_dataframe (pd.DataFrame): The original DataFrame containing all data.
        peak_indices (np.ndarray): Indices of the detected peaks.
        num_expected_segments (int): Expected number of segments (swings). Used to estimate window.

    Returns:
        list: A list of DataFrames, where each DataFrame is a segment.
    """
    segmented_dataframes = []
    if peak_indices is None or len(peak_indices) == 0:
        print("No peak indices provided for segmentation.")
        return segmented_dataframes

    # --- Define window for segmentation ---
    # This is a heuristic. You might need a more sophisticated way or fixed values.
    if num_expected_segments > 0 and len(full_dataframe) > 0 :
        points_per_segment_approx = len(full_dataframe) // num_expected_segments
    else:
        points_per_segment_approx = 50 # Default if no segments or data to estimate

    # Adjust these based on your swing characteristics (how much data before/after peak)
    points_before_peak = points_per_segment_approx // 3
    points_after_peak = (points_per_segment_approx * 2) // 3

    print(f"\nSegmenting data with window: {points_before_peak} points before peak, {points_after_peak} points after peak.")

    for peak_idx in peak_indices:
        start_idx = max(0, peak_idx - points_before_peak)
        # +1 to make slicing inclusive of the end_idx if pandas.iloc is used
        end_idx = min(len(full_dataframe), peak_idx + points_after_peak + 1)

        segment_df = full_dataframe.iloc[start_idx:end_idx].copy()
        segmented_dataframes.append(segment_df)

    print(f"Successfully created {len(segmented_dataframes)} segments.")
    if segmented_dataframes:
        print(f"Example: First segment shape: {segmented_dataframes[0].shape}")
    return segmented_dataframes

def plot_segmentation_results(signal_array, time_index, peak_indices, segmented_dataframes, column_name):
    """
    Plots the original signal, detected peaks, and segmented regions.

    Args:
        signal_array (np.ndarray): The signal that was analyzed.
        time_index (np.ndarray): The time/sample index for the signal.
        peak_indices (np.ndarray): Indices of the detected peaks.
        segmented_dataframes (list): List of DataFrames, each being a segment.
        column_name (str): Name of the signal column for plot title.
    """
    if signal_array is None or time_index is None:
        print("Cannot plot: Signal or time index is missing.")
        return

    plt.figure(figsize=(17, 8))
    plt.plot(time_index, signal_array, label=f'Signal: {column_name}', alpha=0.8, color='dodgerblue')

    if peak_indices is not None and len(peak_indices) > 0:
        plt.plot(time_index[peak_indices], signal_array[peak_indices], "x", color='red', markersize=10, label=f'Detected Peaks ({len(peak_indices)})')

        # Shade segmented regions
        # Need to reconstruct window parameters or pass them if they differ from segmentation
        # For simplicity, let's assume segments in `segmented_dataframes` directly map to plotted regions
        for i, segment_df in enumerate(segmented_dataframes):
            if not segment_df.empty:
                start_plot_idx = segment_df.index[0]
                end_plot_idx = segment_df.index[-1]
                plt.axvspan(start_plot_idx, end_plot_idx,
                            color='gold' if i % 2 == 0 else 'lightsalmon',
                            alpha=0.3, label='Segment' if i == 0 else None)

    plt.title(f'Time Series Segmentation based on Peaks in "{column_name}"')
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('test.png')


def split_data(df: pd.DataFrame) -> list[pd.DataFrame]:
    TARGET_COLUMN_NAME = 'Ax'
    EXPECTED_NUM_SWINGS = 27 
    # fs = 85 # Sampling frequency
    # duration_per_swing = 1.0 # seconds

    ax_signal, R_time_index = load_and_prepare_signal(df, TARGET_COLUMN_NAME)
    if ax_signal is not None:
        # 2. Detect Peaks
        R_peak_indices = detect_swing_peaks(ax_signal, EXPECTED_NUM_SWINGS, TARGET_COLUMN_NAME)

        # 3. Segment Data
        R_list_of_swing_dataframes = segment_data_around_peaks(df, R_peak_indices, EXPECTED_NUM_SWINGS)

        # 4. Plot Overall Segmentation Results
        plot_segmentation_results(ax_signal, R_time_index, R_peak_indices, R_list_of_swing_dataframes, TARGET_COLUMN_NAME)

        if R_list_of_swing_dataframes:
            print(f"\nFinished processing. Found {len(R_list_of_swing_dataframes)} swing segments.")
            return R_list_of_swing_dataframes
        else:
            print("\nNo swing segments were created.")
    else:
        print("Could not proceed due to issues loading the signal.")
        return []

