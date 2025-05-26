import torch
import torch.nn as nn

# Input to Feacture Extraction Layer: [Ax, Ay, Az, Gx, Gy, Gz]
# Input to Classify Heads: assumed to be the features from LSTM
# Model Output: [gender, hold racket handed, play years, level]
# gender(binary): 1:Male / 2:Female  (Output will be a logit, apply sigmoid for probability)
# hold racket handed(binary): 1:right / 2:left (Output will be a logit, apply sigmoid for probability)
# play years(3 classes): 0:Low, 1:Medium, 2:High (Outputs will be logits, apply softmax for probabilities)
# level(4 classes) (Outputs will be logits, apply softmax for probabilities)

# num_classes[1, 1, 3, 4]
class BasicLSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim_lstm, hidden_dim_fc, num_classes: list = [1, 1, 3, 4], dropout=0.2): # Added hidden_dim_fc as required
        super(BasicLSTMClassifier, self).__init__()
        self.n_features = n_features
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_classes = num_classes

        if hidden_dim_fc is None:
            raise ValueError("hidden_dim_fc cannot be None. Please provide a dimension for the fully connected layers.")

        # LSTM layer
        # batch_first=True means input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(n_features, hidden_dim_lstm, num_layers=2, batch_first=True, bidirectional=False)

        self.dropout_layer = nn.Dropout(dropout) # Renamed for clarity from self.dropout to self.dropout_layer

        # Gender head
        self.gender_fc1 = nn.Linear(hidden_dim_lstm, hidden_dim_fc)
        self.gender_relu = nn.ReLU()
        self.gender_fc2 = nn.Linear(hidden_dim_fc, num_classes[0])

        # Hand head
        self.hand_fc1 = nn.Linear(hidden_dim_lstm, hidden_dim_fc)
        self.hand_relu = nn.ReLU()
        self.hand_fc2 = nn.Linear(hidden_dim_fc, num_classes[1])

        # Years head
        self.year_fc1 = nn.Linear(hidden_dim_lstm, hidden_dim_fc)
        self.year_relu = nn.ReLU()
        self.year_fc2 = nn.Linear(hidden_dim_fc, num_classes[2])

        # Level head
        self.level_fc1 = nn.Linear(hidden_dim_lstm, hidden_dim_fc)
        self.level_relu = nn.ReLU()
        self.level_fc2 = nn.Linear(hidden_dim_fc, num_classes[3])

    def forward(self, x):
        # x shape: (batch_size, seq_length, n_features)

        # LSTM layer
        # lstm_out shape: (batch_size, seq_length, hidden_dim_lstm)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim_lstm) -> (1, batch_size, hidden_dim_lstm)
        # cell shape: (num_layers * num_directions, batch_size, hidden_dim_lstm) -> (1, batch_size, hidden_dim_lstm)
        lstm_out, (hidden, cell) = self.lstm(x)

        # We'll use the hidden state from the last time step.
        # hidden is (1, batch_size, hidden_dim_lstm), so we squeeze the first dimension.
        features = hidden.squeeze(0) # shape: (batch_size, hidden_dim_lstm)
        # Alternatively, you could take the output of the last time step from lstm_out:
        # features = lstm_out[:, -1, :] # shape: (batch_size, hidden_dim_lstm)
        # Both are common choices. Using hidden state is slightly more conventional for classification.

        # Apply dropout
        features_dropout = self.dropout_layer(features)

        # Gender head
        gender_x = self.gender_fc1(features_dropout)
        gender_x = self.gender_relu(gender_x)
        gender_x = self.dropout_layer(gender_x) # Optional: dropout after ReLU
        gender_output = self.gender_fc2(gender_x)
        # For binary classification with num_classes=1, output is logit.
        # Apply nn.Sigmoid() later or use nn.BCEWithLogitsLoss()

        # Hand head
        hand_x = self.hand_fc1(features_dropout)
        hand_x = self.hand_relu(hand_x)
        hand_x = self.dropout_layer(hand_x) # Optional: dropout after ReLU
        hand_output = self.hand_fc2(hand_x)
        # For binary classification with num_classes=1, output is logit.

        # Years head
        year_x = self.year_fc1(features_dropout)
        year_x = self.year_relu(year_x)
        year_x = self.dropout_layer(year_x) # Optional: dropout after ReLU
        year_output = self.year_fc2(year_x)
        # For multi-class classification, outputs are logits.
        # Apply nn.Softmax(dim=1) later or use nn.CrossEntropyLoss()

        # Level head
        level_x = self.level_fc1(features_dropout)
        level_x = self.level_relu(level_x)
        level_x = self.dropout_layer(level_x) # Optional: dropout after ReLU
        level_output = self.level_fc2(level_x)
        # For multi-class classification, outputs are logits.

        # The comment "Input to Classify Heads: [features, mode]" mentioned 'mode'.
        # If 'mode' was intended to select a specific head or change behavior,
        # that logic would need to be implemented here. For now, all heads are processed.
        # Example: if mode == 'gender': return gender_output
        # For now, returning all outputs as a list, matching the comment structure.
        return [gender_output, hand_output, year_output, level_output]