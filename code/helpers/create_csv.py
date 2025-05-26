import pandas as pd
import os

def make_csv(img_path: str, src_csv_path: str):
    """
    Generates a new CSV file by combining data from a source CSV and directory structure.

    The function iterates through subdirectories (representing 'unique_id') in an 'images' folder,
    and then further iterates through sub-subdirectories (representing 'swing').
    It matches 'unique_id' with entries in the source CSV and creates new rows
    that include 'unique_id', 'swing', and other data from the source CSV.
    """
    csv_outpath = "/".join(img_path.split('/')[:2]) + "/" + "labels.csv" 

    # print("Src csv path:", src_csv_path)
    src_df = pd.read_csv(src_csv_path) 
    # Ensure 'unique_id' column is treated as string for consistent matching
    if 'unique_id' in src_df.columns:
        src_df['unique_id'] = src_df['unique_id'].astype(str)

    # Define the desired order of columns in the output CSV
    desired_cols_order = ['unique_id', 'swing'] + [col for col in src_df.columns if col != 'unique_id']
    # Get a list of directory names (unique_ids) from the image path
    src_data_ids = [f for f in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, f))]

    rows_to_append = []  # Initialize a list to store new rows for the output DataFrame
    fails_id = []

    # Iterate over each unique ID found in the image directory
    for id_str in src_data_ids:
        folder_path = os.path.join(img_path, id_str)  # Path to the specific unique_id folder
        # Get a list of subdirectory names (swing_ids) within the unique_id folder
        swing_ids_raw = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        swing_ids = sorted(swing_ids_raw, key=int) # Sort swing_ids numerically

        # Find the corresponding row in the source DataFrame for the current unique_id
        src_row_series = src_df[src_df['unique_id'] == id_str]

        # If no matching row is found in the source CSV, skip this unique_id
        if src_row_series.empty:
            fails_id.append(id_str)
            continue

        src_data = src_row_series.iloc[0].to_dict()  # Convert the found row to a dictionary

        # Iterate over each swing ID for the current unique ID
        for swing_id_str in swing_ids:
            new_row = {}  # Initialize a dictionary for the new row
            new_row['unique_id'] = id_str  # Set the unique_id
            new_row['swing'] = swing_id_str  # Set the swing_id

            # Populate the rest of the columns with data from the source CSV
            for col_name in desired_cols_order:
                if col_name not in ['unique_id', 'swing']:
                    new_row[col_name] = src_data.get(col_name)

            rows_to_append.append(new_row)  # Add the newly created row to the list

    # Create the output DataFrame
    out_df = pd.DataFrame(columns=desired_cols_order) # Initialize with columns
    if rows_to_append: # If there are rows to append, create DataFrame from the list
        out_df = pd.DataFrame(rows_to_append, columns=desired_cols_order)

    out_df.to_csv(csv_outpath, index=False)  # Save the output DataFrame to a CSV file
    
    if len(fails_id) > 0: print("Fails:", fails_id)

def main():
    data_path = "data/train/images" 
    src_csv_path = "data/39_Training_Dataset/train_info.csv"
    make_csv(data_path, src_csv_path) 

if __name__ == "__main__":
    main()