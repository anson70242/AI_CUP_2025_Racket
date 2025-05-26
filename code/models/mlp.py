import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes: list = [1, 1, 3, 4], hidden_dim: int = 128, dropout=0.2):
        super(MLP, self).__init__()
        shared_out_dim = input_dim // 2

        # Shared Layer
        self.shared_fc = nn.Linear(input_dim, shared_out_dim)
        self.shared_relu = nn.ReLU()
        self.shared_dropout = nn.Dropout(dropout)

        # Task-specific Heads using the helper method
        self.gender_head = self._create_head(shared_out_dim, num_classes[0], hidden_dim, dropout)
        self.hand_head = self._create_head(shared_out_dim, num_classes[1], hidden_dim, dropout) 
        self.year_head = self._create_head(shared_out_dim, num_classes[2], hidden_dim, dropout)
        self.level_head = self._create_head(shared_out_dim, num_classes[3], hidden_dim, dropout)
        
    def _create_head(self, in_features, out_features, hidden_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Use the passed dropout_rate
            nn.Linear(hidden_dim, out_features)
        )
    
    def forward(self, x):
        # Shared part
        x = self.shared_fc(x)
        x = self.shared_relu(x)   # Apply activation
        x = self.shared_dropout(x) # Apply dropout

        # Pass the shared features through each head
        gender_out = self.gender_head(x)
        hand_out = self.hand_head(x)
        year_out = self.year_head(x)
        level_out = self.level_head(x)

        return gender_out, hand_out, year_out, level_out