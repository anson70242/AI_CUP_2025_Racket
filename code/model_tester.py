import model
import torch

def TCN_test():

    # Example Usage
    batch_size = 10
    seq_length = 5000  # Example sequence length
    n_features = 6   # Ax, Ay, Az, Gx, Gy, Gz
    hidden_size = 128
    

    tcn = model.TCN(
        time_step=n_features,
        num_channels=[16, 32, 64, 128, 256],
        kernel_size=7,
        mlp_hidden_size=hidden_size,
        dropout=0.2,
        num_classes=[1, 1, 3, 4],
    )

    # Create some dummy input data
    # Shape: (batch_size, seq_length, n_features)
    dummy_input = torch.randn(batch_size, seq_length, n_features)
    dummy_input = dummy_input.transpose(-2, -1)

    # Forward pass
    outputs = tcn(dummy_input)

    # Unpack the outputs based on num_classes
    gender_pred, hand_pred, years_pred, level_pred = outputs

    print("Model:", tcn)
    print("\nInput shape:", dummy_input.shape)
    print("Gender output shape:", gender_pred.shape)  # Expected: (batch_size, 1)
    print("Hand output shape:", hand_pred.shape)      # Expected: (batch_size, 1)
    print("Years output shape:", years_pred.shape)    # Expected: (batch_size, 3)
    print("Level output shape:", level_pred.shape)    # Expected: (batch_size, 4)

    # Example of getting probabilities (optional, usually done by loss function)
    # For binary classification (gender, hand), use sigmoid
    gender_probs = torch.sigmoid(gender_pred)
    hand_probs = torch.sigmoid(hand_pred)

    # For multi-class classification (years, level), use softmax
    years_probs = torch.softmax(years_pred, dim=1)
    level_probs = torch.softmax(level_pred, dim=1)

    print("\nGender probabilities (first 2 samples):\n", gender_probs[:2])
    print("\nHand probabilities (first 2 samples):\n", hand_probs[:2])
    print("\nYears probabilities (first 2 samples):\n", years_probs[:2])
    print("\nLevel probabilities (first 2 samples):\n", level_probs[:2])

if __name__ == '__main__':
    # Call the test function
    TCN_test()