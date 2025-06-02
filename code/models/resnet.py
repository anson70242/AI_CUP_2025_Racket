import torch
import torch.nn as nn
import torchvision.models as models

class MyResNet(nn.Module):
    def __init__(self, num_input_channels: int, num_classes: int, sw_mode_feature_length: int = 10): # , multi_task = False
        """
        Initializes the MyResNet model.

        Args:
            num_input_channels (int): Number of input channels for the first convolutional layer.
            num_classes (int): Number of output classes for the final classifier.
            sw_mode_feature_length (int): The length of the sw_mode tensor that will be concatenated.
                                          Defaults to 4 as per your description.
        """
        super(MyResNet, self).__init__()

        # Load the ResNet152 base model without pretrained weights
        self.base_model = models.resnet152(weights=None)

        # 1. Modify the first convolutional layer (conv1)
        #    to accept the specified number of input channels.
        original_conv1 = self.base_model.conv1
        new_conv1 = nn.Conv2d(in_channels=num_input_channels,
                              out_channels=original_conv1.out_channels,
                              kernel_size=original_conv1.kernel_size,
                              stride=original_conv1.stride,
                              padding=original_conv1.padding,
                              bias=(original_conv1.bias is not None))
        self.base_model.conv1 = new_conv1 # Replace the original conv1

        # 2. Prepare the feature extractor.
        #    This will include all layers of the base_model *except* the original fc layer.
        #    It uses the modified conv1.
        modules = list(self.base_model.children())[:-1] # Exclude the last layer (fc)
        self.feature_extractor = nn.Sequential(*modules)

        # 3. Modify the final fully connected layer (classifier).
        #    Get the number of output features from the ResNet's pooling layer.
        num_ftrs_from_resnet = self.base_model.fc.in_features
        
        self.fc = nn.Linear(num_ftrs_from_resnet + sw_mode_feature_length, num_classes)
        
        #    Replace the original fc layer with a new one that accepts the
        #    concatenated features (ResNet features + sw_mode features).
        #    This new fc layer is stored separately as self.fc
        # if multi_task:
        #     self.log_var_gender = nn.Parameter(torch.tensor(0.0))
        #     self.log_var_hand = nn.Parameter(torch.tensor(0.0))
        #     self.log_var_year = nn.Parameter(torch.tensor(0.0))
        #     self.log_var_level = nn.Parameter(torch.tensor(0.0))
            
        #     self.fc = nn.Linear(num_ftrs_from_resnet + sw_mode_feature_length, 1024)
        #     self.gender_head = nn.Linear(1024, 2)
        #     self.hand_head = nn.Linear(1024, 2)
        #     self.year_head = nn.Linear(1024, 3)
        #     self.level_head = nn.Linear(1024, 4)
        # else:
        #     self.fc = nn.Linear(num_ftrs_from_resnet + sw_mode_feature_length, num_classes)

    def forward(self, x: torch.Tensor, sw_mode: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MyResNet model.

        Args:
            x (torch.Tensor): The input tensor (e.g., images).
            sw_mode (torch.Tensor): The supplementary tensor of length `sw_mode_feature_length`.
                                    Expected shape: [batch_size, sw_mode_feature_length] or [sw_mode_feature_length].
                                    If [sw_mode_feature_length], it will be expanded to match the batch size.

        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        # 1. Extract features using the feature_extractor part of the ResNet.
        #    This uses the modified conv1.
        features = self.feature_extractor(x) # Output shape: [batch_size, num_ftrs_from_resnet, 1, 1]
        
        # 2. Flatten the features from the ResNet.
        features_flattened = torch.flatten(features, 1) # Shape: [batch_size, num_ftrs_from_resnet]
            
        # 4. Concatenate the ResNet features with the sw_mode tensor.
        combined_features = torch.cat((features_flattened, sw_mode), dim=1)
        
        # 5. Pass the combined features through the new fully connected layer.
        # if self.multi_task:
        #     hidden = self.fc(combined_features)
        #     gender_out = self.gender_head(hidden)
        #     hand_out = self.hand_head(hidden)
        #     year_out = self.year_head(hidden)
        #     level_out = self.level_head(hidden)
        #     return gender_out, hand_out, year_out, level_out
        # else:
        #     output = self.fc(combined_features)
        #     return output
        
        output = self.fc(combined_features)
        return output

if __name__ == '__main__':

    # --- Configuration ---
    num_img_channels = 6  # For your 6-channel input
    image_height = 224
    image_width = 448
    num_output_classes = 4
    batch_size = 16

    # --- Create an instance of the model ---
    print(f"Creating CustomResNet model with {num_img_channels} input channels and {num_output_classes} output classes.")
    my_custom_model = MyResNet(num_input_channels=num_img_channels, num_classes=num_output_classes)
    

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