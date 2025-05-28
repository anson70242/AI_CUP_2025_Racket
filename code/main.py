import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from helpers import DataBuilder, ImgTimeSeriesDataset, train_val_model, plot_loss, predict, predict_logits
from torch.utils.data import random_split, DataLoader
from models import MyResNet
import pandas as pd

CONFIG = {
    'batch_size': 64,
    'lr': 1e-5,
    'epochs': 50,
    'task': 'gender',
}

def main(make_data=False, train=False, pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    task = CONFIG['task']
    if task == 'gender':
        num_classes = 2
        columns=['label_1', 'label_2']
    elif task == 'hand':
        num_classes = 2
        columns=['hand_1', 'hand_2']
    elif task == 'year':
        num_classes = 3
        columns=['year_0', 'year_1', 'year_2']
    elif task == 'level':
        num_classes = 4
        columns=['level_2', 'level_3', 'level_4', 'level_5']
    else:
        print(f"Main: Task {task} not supported")
        exit()

    if make_data:
        data_folder = "data/39_Training_Dataset/train_data/"
        out_path = "data/train/images"
        csv_path = "data/39_Training_Dataset/train_info.csv"
        column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        train_builder = DataBuilder(data_folder, out_path, column_names, csv_path)
        train_builder.build_data()
        # good, bad = train_builder.check_data(out_path)
        # # print("Good:", good)
        # print("Bad:", bad)

        data_folder = "data/39_Test_Dataset/test_data/"
        out_path = "data/test/images/"
        csv_path = "data/39_Test_Dataset/test_info.csv"
        column_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        test_builder = DataBuilder(data_folder, out_path, column_names, csv_path)
        test_builder.build_data()
        # good, bad = test_builder.check_data(out_path)
        # # print("Good:", good)
        # print("Bad:", bad)

    if train:
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

        model = MyResNet(num_input_channels=6, num_classes=num_classes)
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
        plot_loss(t_losses, v_losses, task=task)
    
    if pred:
        # load model
        model = MyResNet(num_input_channels=6, num_classes=num_classes)
        state_dict = torch.load('trained_models/level_best_resnet152_epx50.pth')
        model.load_state_dict(state_dict)
        model.to(device)
        # print(model)

        # load Data
        csv_file = "data/test/labels.csv"
        img_dir = "data/test/images"
        split = "test"

        test_dataset = ImgTimeSeriesDataset(
            csv_file=csv_file,
            root_dir=img_dir,
            split=split, 
        )

        # Subset for Testing
        # num_items_to_take = min(5, len(test_dataset)) 
        # indices = range(num_items_to_take)
        # test_dataset_subset = Subset(test_dataset, indices)

        # test_dataset = test_dataset_subset 

        test_loader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=False)

        predictions = predict_logits(
            data_loader=test_loader,
            model=model,
            device=device,
            task=task,
        )

        # Saving result
        print(predictions)
        df = pd.DataFrame(predictions, columns=columns)
        print(df.head(), "\n", df.shape)
        df.to_csv(f'predictions/{task}_pred.csv', index=False)

if __name__ == '__main__':
    main(make_data=False, train=True, pred=False)