import torch
import torch.nn as nn
import torch.optim as optim
from helpers import DataBuilder, ImgTimeSeriesDataset, train_val_model
from torch.utils.data import random_split, DataLoader
from models import MyResNet50

CONFIG = {
    'batch_size': 64,
    'lr': 1e-5,
    'epochs': 5,
    'task': 'level',
}

def main(make_data=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    if make_data:
        data_folder = "data/39_Training_Dataset/train_data/"
        out_path = "data/train/full_images"
        csv_path = "data/39_Training_Dataset/train_info.csv"
        column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        train_builder = DataBuilder(data_folder, out_path, column_names, csv_path)
        train_builder.build_data_full()
        # good, bad = train_builder.check_data(out_path)
        # # print("Good:", good)
        # print("Bad:", bad)

        data_folder = "data/39_Test_Dataset/test_data/"
        out_path = "data/test/full_images/"
        csv_path = "data/39_Test_Dataset/test_info.csv"
        column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        test_builder = DataBuilder(data_folder, out_path, column_names, csv_path)
        test_builder.build_data_full()
        # good, bad = test_builder.check_data(out_path)
        # # print("Good:", good)
        # print("Bad:", bad)


    """
    csv_file = "data/train/labels.csv"
    img_dir = "data/train/images"
    split = "train"

    full_dataset = ImgTimeSeriesDataset(
        csv_file=csv_file,
        root_dir=img_dir,
        split=split, 
    )

    dataset_size = len(full_dataset)
    val_split = 0.2
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Split the dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Original dataset size: {len(full_dataset)}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    task = CONFIG['task']
    if task == 'gender':
        num_classes = 2
    elif task == 'hand':
        num_classes = 2
    elif task == 'year':
        num_classes = 3
    elif task == 'level':
        num_classes = 4
    else:
        print(f"Main: Task {task} not supported")
        exit()

    model = MyResNet50(num_input_channels=6, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss() # take logit as input
    optimizer = optim.Adam(model.parameters(), CONFIG['lr'])
    # print(model)

    t_losses, v_losses = train_val_model(
        train_loader = DataLoader(dataset = train_dataset, batch_size=CONFIG['batch_size'], shuffle=True),
        val_loader = DataLoader(dataset = train_dataset, batch_size=CONFIG['batch_size'], shuffle=False),
        model=model,
        epochs=CONFIG['epochs'],
        loss_fn=loss_fn,
        optimizer=optimizer,
        task=task,
        device=device
    )

    print(t_losses, v_losses)
    """
if __name__ == '__main__':
    main(make_data=True)