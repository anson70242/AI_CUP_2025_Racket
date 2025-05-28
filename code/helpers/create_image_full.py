import pandas as pd
import matplotlib.pyplot as plt
import os

def _create_full_imgs(
    data_path: str,
    files: list,
    outpath: str,
    target_width_px: int = 224 * 10,
    target_height_px: int = 224 * 2,
    dpi: int = 100):

    column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

    # Calculate figsize in inches (can be done once)
    figsize_width_inches = target_width_px / dpi
    figsize_height_inches = target_height_px / dpi

    for file_name_without_extension in files:
        if file_name_without_extension == 3030: continue

        file_path = f'{data_path}{file_name_without_extension}.txt'
        
        df = pd.read_csv(file_path, sep=' ', names=column_names)
        print(f"Processing file: {file_path}")
        print(df.head())

        for column in df.columns:
            print(f"Plotting column: {column} for file: {file_name_without_extension}")
            plt.figure(figsize=(figsize_width_inches, figsize_height_inches), dpi=dpi, facecolor='black')
            plt.plot(df.index, df[column], color='white', linestyle='-')
            plt.axis('off')

            out_path = f'{outpath}/{file_name_without_extension}'
            # Create the output directory if it doesn't exist
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            output_filename = f'{out_path}/{column}.png'
            plt.savefig(output_filename)
            print(f"Saved plot to {output_filename}")
            plt.close()

