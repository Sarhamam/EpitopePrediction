import sys
import torch
import logging

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from utils import config
from network_training import train
from utils.network_utils import collate_fn, amino_acids_vocab, get_device

logger = logging.getLogger("RNNNetwork")

# Parameters
NUM_FEATURES = config["NETWORK"].getint("NUM_FEATURES")
DROPOUT = 0.15


class EpitopePredictor(nn.Module):
    # define all the layers used in model
    def __init__(self, input_size, embed_size, numeric_feature_dim, hidden_dim, n_layers,
                 bidirectional, dropout, concat_after, rnn_type):
        # Constructor
        super().__init__()
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
        sequences_final = self.embedding(sequences.to(get_device()).transpose(1, 0))
        properties_final = properties.to(get_device()).transpose(1, 0)
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


def init_model(device, rnn_type, bidirectional, concat_after, hidden_dim, n_layers, lr, embed_size,
               numeric_features=True,
               weighted_loss=False, deterministic=False):
    """ Initializes the model with the params given by the user"""
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(1)

    model = EpitopePredictor(input_size=len(amino_acids_vocab),
                             embed_size=embed_size,
                             numeric_feature_dim=(NUM_FEATURES if numeric_features else 0),
                             hidden_dim=hidden_dim,
                             n_layers=n_layers,
                             bidirectional=bidirectional,
                             concat_after=concat_after,
                             rnn_type=rnn_type,
                             dropout=DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if weighted_loss:
        w = torch.as_tensor([1.0, 13.0])  # weight for 0, weight for 1
        loss_fn = nn.CrossEntropyLoss(weight=w).to(device)
    else:
        loss_fn = nn.BCELoss().to(device)
    return model, optimizer, loss_fn


def train_model(device, model, optimizer, loss_fn, train_dataset, test_dataset, epochs, batch_size, window_size,
                window_overlap, loss_at_end, max_batches, max_length, accuracy_report, deterministic=False):
    """
    Trains the model
    """
    # Create train dataloader
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=not (deterministic),
                            num_workers=0,
                            collate_fn=collate_fn)

    train_loss, train_acc, test_loss, test_acc = train(device, model, optimizer, loss_fn, dataloader, test_dataset,
                                                       max_epochs=epochs, max_batches=max_batches,
                                                       window_size=window_size, window_overlap=window_overlap,
                                                       loss_at_end=loss_at_end, max_length=max_length,
                                                       accuracy_report=accuracy_report, deterministic=deterministic)
    return train_loss, train_acc, test_loss, test_acc


def predict(model, dataset):
    """ Use trained model to predict results"""
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)
    results = {}
    with torch.no_grad():
        for test_idx, test_row in enumerate(dataloader):
            X, p, og_size, id = test_row['Sequence'], test_row['Properties'], test_row['Original-Size'], test_row['ID']
            y_pred = model(X, p, og_size)
            results[id] = (y_pred).detach().squeeze().to('cpu').numpy().tolist()

    return results
