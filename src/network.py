import sys
import json
import time
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def get_device():
    if torch.cuda.is_available():
        logger.info(f"Detected CUDA on device #{torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    logger.info('Using device: %s\n', device)
    return device


logger = logging.getLogger("LSTMNetwork")

# Parameters
BATCH_SIZE = 10
MAX_BATCHES = 1
EMBED_SIZE = 26
HIDDEN_DIM = 128
NUM_FEATURES = 5
MAX_LENGTH = 10000


class AminoAcidDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[str(idx)]


amino_acids_vocab = build_vocab_from_iterator(
    ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v', 'b', 'j', 'z',
     'x'], specials=['<unk>', '<pad>'])
amino_acids_vocab.set_default_index(amino_acids_vocab["<unk>"])


def create_dataset(path, train_test_ratio=0.7):
    dataset = AminoAcidDataset(path)
    train_len = int(len(dataset) * train_test_ratio)
    train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset) - train_len])
    return train_dataset, test_dataset


def calculate_accuracy(true_cls, pred_cls):
    """
    Calculates the AVERAGE prediction accuracy of given sequences (with padding)
    :param true_cls - tensor representing the true classification (of the format (0,0,0,1,1,0,0) where 1 means its in the epitope and 0 not)
    :param pred_cls - tensor representing the predicted probable classification, each entry a float in the range [0,1]
    ## Current implementation treats a float under 0.5 as 0 and above as 1. We could think of a different method if needed ##
    """
    n = pred_cls.shape[0]
    pred_cls = pred_cls[:,1] # probability of 1
    diff = torch.abs(torch.subtract(true_cls, pred_cls))
    corrects = torch.sum(diff < 0.5)

    return corrects / n


def recall_precision_fn(pred_cls, true_cls):
    """
    Calculates the AVERAGE recall accuracy of given sequences (with padding)
    :param true_cls - tensor representing the true classification (of the format (0,0,0,1,1,0,0) where 1 means its in the epitope and 0 not)
    :param pred_cls - tensor representing the predicted probable classification, each entry a float in the range [0,1]
    Compares until the end of the shorter vector
    ## Current implementation treats a float under 0.5 as 0 and above as 1. We could think of a different method if needed ##
    """
    pred_cls = pred_cls[:,1] # probability of 1
    true_indices = (true_cls == 1)  # indices where TP should be expected
    TP = sum(pred_cls[true_indices] > 0.5)
    FP = sum(pred_cls > 0.5) - TP  # Total predicted P minus TP
    FN = sum(true_indices) - TP  # Total expected TP minus predicted TP
    recall = torch.nan_to_num(TP / (TP + FN))
    precision = torch.nan_to_num(TP / (TP + FP))
    return 100 * recall.item(), 100 * precision.item()


def collate_fn(batch):
    """ Returns padded vocabulary mapping tensor"""
    # Batch is a list of entries
    # Cartesian product between all of the values in the batch
    sequence_tensors = []
    properties_tensors = []
    tag_tensors = []
    original_sizes = []
    for entry in batch:
        indexed_sequence = amino_acids_vocab.forward(entry['Sequence'])  # Map according to vocabulary
        original_sizes.append(len(entry["Sequence"]))  # Save original size
        properties_tensors.append(torch.tensor(entry['Properties']))
        sequence_tensors.append(torch.tensor(indexed_sequence))
        tag_tensors.append(torch.tensor(entry['Tags']))

    # Pad batch to minimum size possible
    padded_properties = pad_sequence(properties_tensors, padding_value=-1)
    padded_sequences = pad_sequence(sequence_tensors, padding_value=1)  # <pad> token is index 1 in vocabulary
    padded_tags = pad_sequence(tag_tensors, padding_value=-1)

    result = {
        "Sequence": padded_sequences,
        "Properties": padded_properties,
        "Tags": padded_tags,
        "Original-Size": original_sizes
    }
    return result


class EpitopePredictor(nn.Module):
    # define all the layers used in model
    def __init__(self, device, input_size, embed_size, numeric_feature_dim, hidden_dim, n_layers,
                 bidirectional, dropout, concat_after, rnn_type):
        # Constructor
        super().__init__()
        self.device = device
        self.concat_after = concat_after
        ############################################
        # embedding layer  - CHANGE embed size
        self.embedding = nn.Embedding(input_size, embed_size)
        #############################################
        # RNN layer
        if (rnn_type == 'LSTM'):
            f_RNN = nn.LSTM
        elif (rnn_type == 'GRU'):
            f_RNN = nn.GRU
        else:
            sys.exit('Invalid RNN model. Set RNN to LSTM or GRU')

        self.RNN = f_RNN(input_size=embed_size + numeric_feature_dim * (not concat_after),
                         # not sure, perhaps embed_size
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


def run_model_by_slice(device, model, X, p, y, og_size, win_size, win_overlap):
    win_shift = win_size - win_overlap
    y_pred = torch.empty(0, device=device)
    y_expected = torch.empty(0, device=device)
    y = y.to(device)
    num_seq = X.shape[1]
    for i in range(num_seq):  # for each sequence
        seq_len = og_size[i]
        num_slices = int(np.floor((seq_len - win_size) / win_shift) + 1)
        Li = 0
        Ri = win_size
        size_w = torch.as_tensor([win_size])
        for k in range(num_slices - 1):  # without last slice
            ans = model(X[Li:Ri, i:i + 1], p[Li:Ri, i:i + 1, :], size_w)
            y_pred = torch.cat((y_pred, ans))
            y_expected = torch.cat((y_expected, y[Li:Ri, i]))
            Li += win_shift
            Ri += win_shift
        # last slice
        size_w = torch.as_tensor([seq_len - Li])
        ans = model(X[Li:seq_len, i:i + 1], p[Li:seq_len, i:i + 1, :], size_w)
        y_pred = torch.cat((y_pred, ans))
        y_expected = torch.cat((y_expected, y[Li:seq_len, i]))

    y_pred = y_pred.unsqueeze(-1)
    compl = 1 - y_pred
    y_pred = torch.cat((compl, y_pred), 1) # probablity of 0, probablity of 1
    y_expected = y_expected.type(torch.LongTensor)

    return y_expected, y_pred


def train(device, model, optimizer, loss_fn, train_dataloader, test_dataset, batch_size, window_size, window_overlap, loss_at_end,
          max_epochs=100,
          max_batches=200):

    avg_train_loss = []
    avg_train_acc = []

    avg_train_recall = []
    avg_train_precision = []

    avg_test_loss = []
    avg_test_acc = []

    start_time = time.time()
    for epoch_idx in range(max_epochs):
        logger.info("Running epoch %s out of %s", epoch_idx+1, max_epochs)
        train_loss, train_acc, train_recall, train_precision = 0, 0, 0, 0
        j = 0
        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            X, p, y, og_size = batch['Sequence'], batch['Properties'], batch['Tags'], batch['Original-Size']
            og_size = [min(og_size[i], MAX_LENGTH) for i in range(len(og_size))]

            y = y.type(torch.LongTensor)
            y = y.to(device)

            if loss_at_end == True and window_size > 0:
                y_expected, y_pred = run_model_by_slice(device, model, X, p, y, og_size, window_size, window_overlap)
                y_expected = y_expected.unsqueeze(-1)
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_expected)
                loss.backward()

                # Weight updates
                optimizer.step()
                train_loss += loss.item()

                # Accuracy, Recall, Precision
                accuracy = calculate_accuracy(y_expected, y_pred)
                train_acc += accuracy
                recall, precision = recall_precision_fn(y_pred, y_expected)
                train_recall += recall
                train_precision += precision

                # print(y_pred[y_expected == 1, 0])
                # print("min:", min(y_pred[y_expected == 0, 0]))
                # print("max:", max(y_pred[y_expected == 0, 0]))

            elif window_size > 0:
                win_shift = window_size - window_overlap
                n_shifts = int(np.ceil(1.0 * (X.size(0) - window_size) / win_shift))  # need to check
                for i_shift in range(n_shifts + 1):
                    bs = range(len(og_size))
                    Li = i_shift * win_shift  # left index
                    Ri = Li + window_overlap  # right index
                    Ri = min(Ri, X.size(0))
                    size_W = [max(min(og_size[i] - Li, window_overlap), 0) for i in bs]
                    non_empty = torch.as_tensor(size_W) > 0
                    size_W = torch.as_tensor(size_W)
                    size_W = size_W[non_empty]

                    X_W = X[Li:Ri, non_empty]
                    p_W = p[Li:Ri, non_empty]
                    y_W = y[Li:Ri, non_empty]

                    # Forward pass
                    optimizer.zero_grad()
                    y_pred_W = model(X_W, p_W, size_W)
                    y_expected_W = pack_padded_sequence(y_W.T, size_W, batch_first=True,
                                                        enforce_sorted=False).data.unsqueeze(-1)
                    y_pred_W = y_pred_W.unsqueeze(-1)
                    compl = 1 - y_pred_W
                    y_pred_W = torch.cat((compl, y_pred_W), 1) # probablity of 0, probablity of 1
                    y_expected_W = y_expected_W.type(torch.LongTensor)

                    loss = loss_fn(y_pred_W, y_expected_W)

                    # Weight updates
                    loss.backward()
                    optimizer.step()

                    # Contribute to average loss, accuracy etc.
                    train_loss += loss.item() / (n_shifts + 1)
                    accuracy = calculate_accuracy(y_expected_W, y_pred_W)
                    train_acc += accuracy/(n_shifts+1)
                    recall, precision = recall_precision_fn(y_pred_W, y_expected_W)
                    train_recall += recall/(n_shifts+1)
                    train_precision += precision/(n_shifts+1)

            else:
                y_pred = model(X, p, og_size)
                y_expected = pack_padded_sequence(y.T, og_size, batch_first=True,
                                                        enforce_sorted=False).data.unsqueeze(-1)
                y_pred = y_pred.unsqueeze(-1)
                compl = 1 - y_pred
                y_pred = torch.cat((compl, y_pred), 1) # probablity of 0, probablity of 1
                y_expected = y_expected.type(torch.LongTensor)

                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_expected)
                loss.backward()

                # Weight updates
                optimizer.step()
                train_loss += loss.item()

                # Accuracy, Recall, Precision
                accuracy = calculate_accuracy(y_expected, y_pred)
                train_acc += accuracy
                recall, precision = recall_precision_fn(y_pred, y_expected)
                train_recall += recall
                train_precision += precision


            j += 1

            # For debugging
            if batch_idx == max_batches - 1:
                break

        # avg batch metrics after each epoch (j total batches):
        avg_train_loss.append(train_loss / j)
        avg_train_acc.append(train_acc / j)
        avg_train_recall.append(train_recall / j)
        avg_train_precision.append(train_precision / j)

        logger.info(
            f"Epoch #{epoch_idx}, train loss = {train_loss / j:.3f}, train accuracy = {train_acc / j:.3f}, train recall % = {train_recall / j:.1f}, train precision % = {train_precision / j:.1f}, epoch_time={time.time() - epoch_start_time:.1f} sec, total_time={time.time() - start_time:.1f} sec")

        # test network on the test dataset
        test_loss, test_acc = test_model(device, model, loss_fn, test_dataset)
        avg_test_loss.append(test_loss)
        avg_test_acc.append(test_acc)

        logger.info(
            f"\t  test loss = {test_loss:.3f}, test accuracy = {test_acc:.3f}")

    np.savetxt("dbg_loss.csv", np.asarray(avg_train_loss), delimiter=",")

    return avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc


def test(device, model, loss_fn, dataloader):
    " run on test data and get loss and accuracy "
    test_loss = 0
    test_acc = 0
    j = 0
    for test_idx, test_row in enumerate(dataloader):
        X, p, y, og_size = test_row['Sequence'], test_row['Properties'], test_row['Tags'], test_row['Original-Size']
        y = y.type(torch.FloatTensor).to(device)

        # predict
        y_pred = model(X, p, og_size)
        y_pred = y_pred.unsqueeze(-1)
        compl = 1 - y_pred
        y_pred = torch.cat((compl, y_pred), 1)  # probablity of 0, probablity of 1

        y_expected = pack_padded_sequence(y.T,og_size, batch_first=True,enforce_sorted=False).data.unsqueeze(-1)
        y_expected = y_expected.type(torch.LongTensor)

        # loss
        loss = loss_fn(y_pred, y_expected)
        loss = loss.item()
        test_loss += loss

        # Accuracy
        accuracy = calculate_accuracy(y_expected, y_pred)
        test_acc += accuracy

        j += 1
        # For debugging:
        if j == 4:
            break

    return test_loss / j, test_acc / j


def init_model(device, rnn_type, bidirectional, concat_after):
    # instantiate the model
    model = EpitopePredictor(input_size=len(amino_acids_vocab),
                             embed_size=EMBED_SIZE,
                             numeric_feature_dim=NUM_FEATURES,
                             hidden_dim=HIDDEN_DIM,
                             n_layers=2,
                             bidirectional=bidirectional,
                             device=device,
                             concat_after=concat_after,
                             rnn_type=rnn_type,
                             dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    w = torch.as_tensor([1.0, 7.7]) # weight for 0, weight for 1
    loss_fn = nn.CrossEntropyLoss(weight=w).to(device)
    # loss_fn = nn.BCELoss().to(device)
    return model, optimizer, loss_fn


def train_model(device, model, optimizer, loss_fn, train_dataset, test_dataset, epochs, batch_size, window_size, window_overlap,
                loss_at_end):
    # Create train dataloader
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=collate_fn)

    train_loss, train_acc, test_loss, test_acc = train(device, model, optimizer, loss_fn, dataloader, test_dataset,
                                                        batch_size=batch_size, max_epochs=epochs, max_batches=MAX_BATCHES,
                                                        window_size=window_size, window_overlap=window_overlap, loss_at_end=loss_at_end)
    return train_loss, train_acc, test_loss, test_acc


def test_model(device, model, loss_fn, test_dataset):
    # Create test dataloader
    dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=collate_fn)

    test_loss, test_acc = test(device, model, loss_fn, dataloader)
    return test_loss, test_acc
