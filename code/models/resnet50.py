import torch
import torch.nn as nn
import torchvision.models as models

class MyResNet50(nn.Module):
    def __init__(self, num_input_channels: int, num_classes: int):
        super(MyResNet50, self).__init__()

        self.base_model = models.resnet152(weights=None)

        original_conv1 = self.base_model.conv1
        new_conv1 = nn.Conv2d(in_channels=num_input_channels,
                              out_channels=original_conv1.out_channels,
                              kernel_size=original_conv1.kernel_size,
                              stride=original_conv1.stride,
                              padding=original_conv1.padding,
                              bias=(original_conv1.bias is not None)) # Check if bias was present

        # Replace the original conv1 layer with the new one
        self.base_model.conv1 = new_conv1

        # Modify the final fully connected layer (classifier)
        # Get the number of input features to the original fully connected layer
        num_ftrs = self.base_model.fc.in_features
        
        # Replace the original fully connected layer with a new one
        # that has 'num_classes' output units.
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

if __name__ == '__main__':

    # --- Configuration ---
    num_img_channels = 6  # For your 6-channel input
    image_height = 224
    image_width = 448
    num_output_classes = 4
    batch_size = 16

    # --- Create an instance of the model ---
    print(f"Creating CustomResNet50 model with {num_img_channels} input channels and {num_output_classes} output classes.")
    my_custom_model = MyResNet50(num_input_channels=num_img_channels, num_classes=num_output_classes)
    

    # --- Example: Create a dummy input tensor ---
    # Shape: (batch_size, num_input_channels, height, width)
    dummy_input = torch.randn(batch_size, num_img_channels, image_height, image_width)
    print(f"\nCreated a dummy input tensor of shape: {dummy_input.shape}")

    # --- Perform a forward pass ---
    try:
        print("Performing a forward pass with the dummy input...")
        output = my_custom_model(dummy_input)
        print(f"Model successfully processed input.")
        print(f"Output tensor shape: {output.shape}") # Expected: (batch_size, num_output_classes)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")