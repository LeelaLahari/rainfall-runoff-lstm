"""
LSTM rainfall-runoff model. MC-dropout wired in (consistent with your
flood-emulator's uncertainty approach — same technique, different model).
"""
import torch
import torch.nn as nn


class RainfallRunoffLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout_p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_p)   # stays active at inference for MC-dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])      # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)