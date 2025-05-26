import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing

def _create_single_image_task(args_tuple):
    """
    Worker function to create a single image.
    This function is designed to be called by multiprocessing.Pool.map().

    Args:
        args_tuple (tuple): A tuple containing (df, df_original_index, column_name, base_outpath, target_width_px, target_height_px, dpi).
                            - df (pd.DataFrame): The DataFrame (or relevant slice) to plot from.
                            - df_original_index (int): The original index of the DataFrame in the input list, used for naming.
                            - column_name (str): The name of the column to plot.
                            - base_outpath (str): The base output directory for all images.
                            - target_width_px (int): Desired width of the output image in pixels.
                            - target_height_px (int): Desired height of the output image in pixels.
                            - dpi (int): Dots per inch for the output image.
    """
    df, df_original_index, col_name, base_outpath, target_width_px, target_height_px, dpi = args_tuple
    
    try:
        time_steps = df.index

        # Calculate figsize in inches
        figsize_width_inches = target_width_px / dpi
        figsize_height_inches = target_height_px / dpi

        # Create plot
        # plt.figure(figsize=(12, 6), facecolor='black') # Original line
        plt.figure(figsize=(figsize_width_inches, figsize_height_inches), dpi=dpi, facecolor='black')
        plt.plot(time_steps, df[col_name], color='white')
        plt.axis('off')
        plt.tight_layout(pad=0) # Reduce padding

        # Prepare output path using df_original_index as the folder name
        index_folder_name = str(df_original_index + 1) # 1-based index for folder
        index_outpath = os.path.join(base_outpath, index_folder_name)
        
        os.makedirs(index_outpath, exist_ok=True)
        
        filename = f'{col_name}.png'
        filepath = os.path.join(index_outpath, filename)

        # Save the plot with the specified DPI
        # plt.savefig(filepath, facecolor='black', edgecolor='none') # Original line
        plt.savefig(filepath, facecolor='black', edgecolor='none', dpi=dpi)
        plt.close() # Important: close the plot to free memory

        # Open the saved image, convert to 1-bit, and save again
        # This part does not change the dimensions, only the color depth.
        img = Image.open(filepath).convert("1")
        img.save(filepath)
        
        return None # Indicate success
    except Exception as e:
        print(f"Error processing DataFrame index {df_original_index}, column '{col_name}': {e}")
        return e

def create_images_parallel(dfs: list[pd.DataFrame], outpath: str, 
                           target_width_px: int = 448, 
                           target_height_px: int = 224, 
                           dpi: int = 100, 
                           num_processes: int = None):
    """
    Creates time series plots from a list of DataFrames in parallel using multiprocessing.

    Args:
        dfs (list[pd.DataFrame]): A list of pandas DataFrames.
        outpath (str): The base directory where images will be saved.
        target_width_px (int): Desired width of the output image in pixels.
        target_height_px (int): Desired height of the output image in pixels.
        dpi (int): Dots per inch for the output image.
        num_processes (int, optional): The number of worker processes to use.
                                       If None, defaults to os.cpu_count().
    """
    if not dfs:
        print("No DataFrames provided to process.")
        return

    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    tasks = []
    for i, df in enumerate(dfs):
        for col in df.columns:
            # Pass the new dimension and DPI arguments
            tasks.append((df.copy(), i, col, outpath, target_width_px, target_height_px, dpi))

    if not tasks:
        print("No image generation tasks to perform.")
        return

    effective_num_processes = num_processes or os.cpu_count()
    print(f"Starting parallel image generation for {len(tasks)} images using up to {effective_num_processes} processes...")
    print(f"Target dimensions: {target_width_px}x{target_height_px} pixels at {dpi} DPI.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_create_single_image_task, tasks)

    successful_tasks = sum(1 for res in results if res is None)
    failed_tasks = len(results) - successful_tasks

    if failed_tasks > 0:
        print(f"\nImage generation completed with {failed_tasks} errors out of {len(tasks)} tasks.")
    else:
        print(f"\nSuccessfully generated {successful_tasks} minimal time series plots in parallel.")
    print(f"Output directory: {outpath}")


# Example of how to use the parallel function (for testing this script directly)
if __name__ == '__main__':
    # Create some dummy data for testing
    num_dfs = 10
    num_cols = 6
    num_rows = 200 
    
    sample_dfs = []
    column_names_example = [f'Sensor_{chr(65+j)}' for j in range(num_cols)]

    for i in range(num_dfs):
        data = {}
        for j_idx, col_name in enumerate(column_names_example):
            data[col_name] = [k + (i*0.2) + (j_idx*0.05) + 5 * (1 if j_idx % 2 == 0 else -1) * (k/50.0) for k in range(num_rows)]
        df_temp = pd.DataFrame(data)
        df_temp.index.name = "time_step"
        sample_dfs.append(df_temp)

    output_directory_parallel = "data_test/parallel_output_images_custom_size"

    print(f"Preparing to generate images for {len(sample_dfs)} DataFrames, each with {num_cols} columns.")

    # Test parallel version with custom dimensions
    print("\nTesting parallel image generation with custom size (448x224 pixels)...")
    # Specify target dimensions and DPI
    create_images_parallel(sample_dfs, output_directory_parallel, 
                           target_width_px=448, 
                           target_height_px=224, 
                           dpi=100, # You can experiment with this
                           num_processes=None)
    print(f"Parallel image generation finished. Check the '{output_directory_parallel}' directory.")

    # Example with different DPI, leading to different figsize in inches but same pixel output
    # output_directory_parallel_dpi_200 = "data_test/parallel_output_images_custom_size_dpi200"
    # print("\nTesting parallel image generation with custom size (448x224 pixels) at 200 DPI...")
    # create_images_parallel(sample_dfs, output_directory_parallel_dpi_200, 
    #                        target_width_px=448, 
    #                        target_height_px=224, 
    #                        dpi=200, 
    #                        num_processes=None)
    # print(f"Parallel image generation finished. Check the '{output_directory_parallel_dpi_200}' directory.")