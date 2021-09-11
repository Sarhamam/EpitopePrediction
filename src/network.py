import sys
import torch
import logging

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from utils import config
from network_training import train
from utils.network_utils import collate_fn, amino_acids_vocab

logger = logging.getLogger("RNNNetwork")

# Parameters
EMBED_SIZE = config["NETWORK"].getint("EMBED_SIZE")
NUM_FEATURES = config["NETWORK"].getint("NUM_FEATURES")
DROPOUT = 0.2

# Remove the following?
BATCH_SIZE = config["NETWORK"].getint("BATCH_SIZE")
HIDDEN_DIM = config["NETWORK"].getint("HIDDEN_DIM")
MAX_BATCHES = config["NETWORK"].getint("MAX_BATCHES")
MAX_LENGTH = config["NETWORK"].getint("MAX_LENGTH")


# Create global vocabulary
##########################


class EpitopePredictor(nn.Module):
    # define all the layers used in model
    def __init__(self, device, input_size, embed_size, numeric_feature_dim, hidden_dim, n_layers,
                 bidirectional, dropout, concat_after, rnn_type):
        # Constructor
        super().__init__()
        self.device = device
        self.concat_after = concat_after
        # embedding layer
        self.embedding = nn.Embedding(input_size, embed_size)
        # RNN layer
        if (rnn_type == 'LSTM'):
            f_RNN = nn.LSTM
        elif (rnn_type == 'GRU'):
            f_RNN = nn.GRU
        else:
            sys.exit('Invalid RNN model. Set RNN to LSTM or GRU')

        self.RNN = f_RNN(input_size=embed_size + numeric_feature_dim * (not concat_after),
                         hidden_size=hidden_dim,
                         num_layers=n_layers,
                         bidirectional=bidirectional,
                         dropout=dropout,
                         batch_first=True)

        # dense layer
        self.dropout = nn.Dropout(dropout)
        self.linear_test = nn.Linear(hidden_dim * (1 + bidirectional) + numeric_feature_dim * concat_after, 1)

        # activation function
        self.activation = nn.Sigmoid()

        self.numeric_feature_dim = numeric_feature_dim

    def forward(self, sequences, properties, text_lengths):
        sequences_final = self.embedding(sequences.to(self.device).transpose(1, 0))
        properties_final = properties.to(self.device).transpose(1, 0)
        if (self.numeric_feature_dim > 0) and (not self.concat_after):
            all_final = torch.cat((sequences_final, properties_final), -1)
        else:
            all_final = sequences_final
        packed_sequences = pack_padded_sequence(all_final, text_lengths, batch_first=True, enforce_sorted=False)
        RRN_out, _ = self.RNN(packed_sequences)
        dropout = self.dropout(RRN_out.data)
        if (self.numeric_feature_dim > 0) and self.concat_after:
            packed_properties = pack_padded_sequence(properties_final, text_lengths, batch_first=True,
                                                     enforce_sorted=False)
            dropout = torch.cat((dropout, packed_properties.data), -1)
        return self.activation(self.linear_test(dropout))


def init_model(device, rnn_type, bidirectional, concat_after, hidden_dim, n_layers, lr, numeric_features=True):
    # instantiate the model
    model = EpitopePredictor(input_size=len(amino_acids_vocab),
                             embed_size=EMBED_SIZE,
                             numeric_feature_dim=(NUM_FEATURES if numeric_features else 0),
                             hidden_dim=hidden_dim,
                             n_layers=n_layers,
                             bidirectional=bidirectional,
                             device=device,
                             concat_after=concat_after,
                             rnn_type=rnn_type,
                             dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) #lr=0.0005
    w = torch.as_tensor([1.0, 7.7]) # weight for 0, weight for 1
    loss_fn = nn.CrossEntropyLoss(weight=w).to(device)
    return model, optimizer, loss_fn


def train_model(device, model, optimizer, loss_fn, train_dataset, test_dataset, epochs, batch_size, window_size, window_overlap,
                loss_at_end,max_batches,max_length):
    # Create train dataloader
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=collate_fn)

    train_loss, train_acc, test_loss, test_acc = train(device, model, optimizer, loss_fn, dataloader, test_dataset,
                                                       max_epochs=epochs, max_batches=MAX_BATCHES,
                                                       window_size=window_size, window_overlap=window_overlap,
                                                       loss_at_end=loss_at_end, max_length=max_length)
    return train_loss, train_acc, test_loss, test_acc
