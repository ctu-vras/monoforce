import torch
import torch.nn as nn


class TrajLSTM(nn.Module):
    def __init__(self, state_features, control_features, heightmap_shape, lstm_hidden_size=256):
        super(TrajLSTM, self).__init__()

        # CNN for processing the entire heightmap
        self.heightmap_cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Compute CNN output size
        heightmap_height, heightmap_width = heightmap_shape
        pooled_height = heightmap_height // (2 * 2)  # Two pooling layers
        pooled_width = heightmap_width // (2 * 2)
        cnn_output_size = pooled_height * pooled_width * 16

        # Dense layer for robot state + control input processing
        self.robot_state_nn = nn.Sequential(
            nn.Linear(state_features + control_features, 16),
            nn.ReLU()
        )

        # LSTM for sequential processing
        self.lstm_input_size = 16 + cnn_output_size
        self.lstm = nn.LSTM(self.lstm_input_size, lstm_hidden_size, batch_first=True)

        # Output layer to predict the next state in the sequence
        self.output_fc = nn.Linear(lstm_hidden_size, state_features)

    def forward(self, initial_state, control_inputs, heightmap):
        # Process heightmap through CNN (global terrain context)
        batch_size, _, heightmap_h, heightmap_w = heightmap.size()
        heightmap_features = self.heightmap_cnn(heightmap)  # Shape: (batch_size, cnn_output_size)

        # Repeat heightmap features for each time step in the sequence
        seq_len = control_inputs.size(1)  # Length of the control input sequence
        heightmap_features = heightmap_features.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate initial robot state and control inputs for each time step
        # Initial state is repeated across the sequence
        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, seq_len, 1)
        combined_inputs = torch.cat((initial_state_repeated, control_inputs), dim=-1)  # Shape: (batch_size, seq_len, state_features + control_features)

        # Process combined inputs through a dense layer
        robot_state_features = self.robot_state_nn(combined_inputs)

        # Concatenate robot state/control features with heightmap features
        combined_features = torch.cat((robot_state_features, heightmap_features), dim=-1)

        # Pass through LSTM to capture temporal dependencies
        lstm_out, _ = self.lstm(combined_features)

        # Predict next states in the sequence
        output = self.output_fc(lstm_out)

        return output

    def from_pretrained(self, modelf):
        if not modelf:
            return self
        print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        model_dict = self.state_dict()
        pretrained_model = torch.load(modelf, weights_only=True)
        model_dict.update(pretrained_model)
        self.load_state_dict(model_dict)
        return self