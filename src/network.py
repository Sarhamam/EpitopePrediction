import json
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torchtext.data
import torch.optim as optim
import os, sys
import time
import numpy as np


if torch.cuda.is_available():
    print(f"Detected CUDA on device #{torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
else:
    device = 'cpu'

print('Using device:\n', device)


# Parameters
BATCH_SIZE = 10
MAX_BATCHES = 1
MAX_EPOCH = 100
EMBED_SIZE = 26
HIDDEN_DIM = 128
NUM_FEATURES = 5
BI_DIRECT = True        # Biderctional RNN
CONCAT_AFTER = False    # Concatenate physical properties after RNN. If true, must use 'BI_DIRECT = True'
WINDOW_SIZE = 256       # Slicing Window size, -1 for no slice
WINDOW_OVERLAP = 32     # Slicing Window overlap
MAX_LENGTH = 10000      # Max truncated sequences length 
SHUFFLE = False         # Shuffle of training dataset
RNN = 'LSTM'            # LSTM/GRU

LOSS_AT_END = False     # True for testing Michal's method

torch.manual_seed(1)    # Should be removed! Used to avoid randomness when debbuging


class AminoAcidDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file) as h:
            self.data = json.load(h)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[str(idx)]

dataset = AminoAcidDataset(os.path.dirname(__file__)+"/../resources/iedb_linear_epitopes.tsv")
# dataset = AminoAcidDataset("/content/drive/My Drive/Colab Notebooks/full.json")
# dataset = AminoAcidDataset("C:\\Users\\eladm\\Documents\\Uni\\Year C\\Sem B\\Courses\\Bio Project\\Final Project\\EpitopePrediction-main\\EpitopePrediction-main\\resources\\full.json")

train_test_ratio = 0.7
train_len = int(len(dataset) * train_test_ratio)
train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset) - train_len])


amino_acids_vocab = build_vocab_from_iterator(
    ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v', 'b', 'j', 'z',
     'x'], specials=['<unk>', '<pad>'])
amino_acids_vocab.set_default_index(amino_acids_vocab["<unk>"])

def calculate_accuracy(true_cls, pred_cls):
    """
    Calculates the AVERAGE prediction accuracy of given sequences (with padding)
    :param true_cls - tensor representing the true classification (of the format (0,0,0,1,1,0,0) where 1 means its in the epitope and 0 not)
    :param pred_cls - tensor representing the predicted probable classification, each entry a float in the range [0,1]
    ## Current implementation treats a float under 0.5 as 0 and above as 1. We could think of a different method if needed ##
    """
    # print(pred_cls.shape, true_cls.shape)
    n = pred_cls.shape[0]
    diff = torch.abs(torch.subtract(true_cls, pred_cls))
    corrects = torch.sum(diff < 0.5)

    return corrects / n


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

traindataloader = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=SHUFFLE,
                        num_workers=0,
                        collate_fn=collate_fn)

testdataloader = DataLoader(test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_fn)


class EpitopePredictor(nn.Module):
    # define all the layers used in model
    def __init__(self, input_size, embed_size, numeric_feature_dim, hidden_dim, n_layers,
                 bidirectional, dropout):
        # Constructor
        super().__init__()
        ############################################
        # embedding layer  - CHANGE embed size
        self.embedding = nn.Embedding(input_size, embed_size)
        #############################################
        # RNN layer
        if (RNN == 'LSTM'):
            f_RNN = nn.LSTM
        elif (RNN == 'GRU'):
            f_RNN = nn.GRU
        else:
            sys.exit('Invalid RNN model. Set RNN to LSTM or GRU')
        
        self.RNN = f_RNN(   input_size=embed_size+numeric_feature_dim*(not CONCAT_AFTER), #not sure, perhaps embed_size
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)       

        # dense layer
        self.dropout = nn.Dropout(0.2)
        self.linear_test = nn.Linear(hidden_dim*(1+bidirectional)+numeric_feature_dim*CONCAT_AFTER, 1)

        # activation function
        self.activation = nn.Sigmoid()

        self.numeric_feature_dim = numeric_feature_dim


    def forward(self, sequences, properties, text_lengths):
        sequences_final = self.embedding(sequences.to(device).transpose(1,0))
        properties_final = properties.to(device).transpose(1,0)
        if (self.numeric_feature_dim > 0) and (not CONCAT_AFTER):
            all_final = torch.cat((sequences_final,properties_final),-1)
        else:
            all_final = sequences_final
        packed_sequences = pack_padded_sequence(all_final, text_lengths, batch_first=True, enforce_sorted=False)
        RRN_out, _ = self.RNN(packed_sequences)
        dropout = self.dropout(RRN_out.data)
        if (self.numeric_feature_dim > 0) and CONCAT_AFTER:
            packed_properties = pack_padded_sequence(properties_final, text_lengths, batch_first=True, enforce_sorted=False)
            dropout = torch.cat((dropout,packed_properties.data),-1)
        return self.activation((self.linear_test(dropout)))

def recall_precision_fn(pred_cls,true_cls):
    """
    Calculates the AVERAGE recall accuracy of given sequences (with padding)
    :param true_cls - tensor representing the true classification (of the format (0,0,0,1,1,0,0) where 1 means its in the epitope and 0 not)
    :param pred_cls - tensor representing the predicted probable classification, each entry a float in the range [0,1]
    Compares until the end of the shorter vector
    ## Current implementation treats a float under 0.5 as 0 and above as 1. We could think of a different method if needed ##
    """
    true_indices = (true_cls==1) #indices where TP should be expected
    TP = sum(pred_cls[true_indices] > 0.5)
    FP = sum(pred_cls > 0.5) - TP # Total predicted P minus TP
    FN = sum(true_indices) - TP  # Total expected TP minus predicted TP
    recall = torch.nan_to_num(TP/(TP+FN))
    precision = torch.nan_to_num(TP/(TP+FP))
    return 100*recall.item(), 100*precision.item()

def run_model_by_slice(model, X, p, y, og_size, win_size, win_overlap):
    win_shift = win_size-win_overlap
    y_pred = torch.empty(0,device=device)
    y_expected = torch.empty(0,device=device)
    y = y.to(device)
    num_seq = X.shape[1]
    for i in range(num_seq): # for each sequence
        seq_len = og_size[i]
        num_slices = int(np.floor((seq_len-win_size) / win_shift)+1)
        Li = 0
        Ri = win_size
        size_w = torch.as_tensor([win_size])
        for k in range(num_slices-1): # without last slice
            ans = model(X[Li:Ri, i:i+1], p[Li:Ri, i:i+1, :], size_w)
            y_pred = torch.cat((y_pred, ans))
            y_expected = torch.cat((y_expected, y[Li:Ri, i]))
            Li += win_shift
            Ri += win_shift
        # last slice
        size_w = torch.as_tensor([seq_len-Li])
        ans = model(X[Li:seq_len, i:i+1], p[Li:seq_len, i:i+1, :], size_w)
        y_pred = torch.cat((y_pred,ans))
        y_expected = torch.cat((y_expected, y[Li:seq_len, i]))

    return y_expected, y_pred


def train(model, optimizer, loss_fn, dataloader, max_epochs=100, max_batches=200):

    avg_train_loss = []
    avg_train_recall = []
    avg_train_precision = []

    start_time = time.time()

    for epoch_idx in range(max_epochs):
        train_loss, train_acc, train_recall, train_precision = 0, 0, 0, 0
        j = 0
        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            X, p, y, og_size = batch['Sequence'], batch['Properties'], batch['Tags'], batch['Original-Size']
            og_size= [min(og_size[i],MAX_LENGTH) for i in range(len(og_size))]

            y = y.type(torch.FloatTensor)
            y = y.to(device)

            if LOSS_AT_END == True:
                # Forward pass - SLICED
                y_expected, y_pred = run_model_by_slice(model, X, p, y, og_size, WINDOW_SIZE, WINDOW_OVERLAP)            
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_expected.unsqueeze(-1))
                loss.backward()
                # Weight updates
                optimizer.step()
                train_loss += loss.item()                
            else:
                win_shift = WINDOW_SIZE-WINDOW_OVERLAP
                n_shifts = int(np.ceil(1.0*(X.size(0)-WINDOW_SIZE)/win_shift)) # need to check
                for i_shift in range(n_shifts+1):
                    bs=range(len(og_size))
                    Li = i_shift*win_shift #left index
                    Ri = Li+WINDOW_SIZE #right index
                    Ri = min(Ri,X.size(0))
                    size_W = [max(min(og_size[i]-Li,WINDOW_SIZE),0) for i in bs]
                    non_empty = torch.as_tensor(size_W) > 0
                    size_W = torch.as_tensor(size_W)
                    size_W = size_W[non_empty]

                    X_W=X[Li:Ri,non_empty]
                    p_W=p[Li:Ri,non_empty] 
                    y_W=y[Li:Ri,non_empty]                                                            
                    
                    # Forward pass
                    optimizer.zero_grad()
                    y_pred_W = model(X_W,p_W,size_W)
                    y_expected_W = pack_padded_sequence(y_W.T, size_W, batch_first=True, enforce_sorted=False).data.unsqueeze(-1)
                    loss = loss_fn(y_pred_W,y_expected_W)

                    # Weight updates
                    loss.backward()
                    optimizer.step()                       

                    # Contribute to average loss
                    train_loss += loss.item()/(n_shifts+1)                                                                                                

            # Accuracy, Recall, Precision
            y_expected = pack_padded_sequence(y.T,og_size, batch_first=True,enforce_sorted=False).data.unsqueeze(-1)
            y_pred = model(X,p,og_size)
            accuracy = calculate_accuracy(y_expected, y_pred)
            train_acc += accuracy
            recall,precision = recall_precision_fn(y_pred, y_expected)
            train_recall += recall
            train_precision += precision

            j += 1

            if batch_idx == max_batches-1:
                # np.savetxt("dbg_X.csv", np.asarray(X[:og_size[0]].cpu().data), delimiter=",")
                # np.savetxt("dbg_y_pred.csv", np.asarray(y_pred.cpu().data), delimiter=",")
                # np.savetxt("dbg_y_expected.csv", np.asarray(y_expected.cpu().data.T), delimiter=",")
                break

        # avg batch metrics after each epoch (j total batches):
        avg_train_loss.append(train_loss / j)
        avg_train_recall.append(train_recall / j)
        avg_train_precision.append(train_precision / j)

        print(f"Epoch #{epoch_idx}, loss = {train_loss / j:.3f}, accuracy = {train_acc / j:.3f}, recall % = {train_recall / j:.1f}, precision % = {train_precision / j:.1f}, epoch_time={time.time()-epoch_start_time:.1f} sec, total_time={time.time()-start_time:.1f} sec")

    np.savetxt("dbg_loss.csv", np.asarray(avg_train_loss), delimiter=",")

    return avg_train_loss


# instantiate the model
model = EpitopePredictor(input_size=len(amino_acids_vocab),
                         embed_size=EMBED_SIZE,
                         numeric_feature_dim=NUM_FEATURES,
                         hidden_dim=HIDDEN_DIM,
                         n_layers=2,
                         bidirectional=BI_DIRECT, dropout=0.2).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCELoss().to(device)
train_loss = train(model, optimizer, loss_fn, traindataloader, max_epochs=MAX_EPOCH, max_batches=MAX_BATCHES)

# run on test data
i = 0
for test_idx, test_row in enumerate(testdataloader):
    i += 1
    X, p, y, og_size = test_row['Sequence'], test_row['Properties'], test_row['Tags'], test_row['Original-Size']
    y = y.type(torch.FloatTensor).to(device)

    # predict
    # Accuracy, Recall, Precision
    y_expected = pack_padded_sequence(y.T,og_size, batch_first=True,enforce_sorted=False).data.unsqueeze(-1)
    y_pred = model(X,p,og_size)

    # loss
    loss = loss_fn(y_pred, y_expected)
    loss = loss.item()

    # Recall & Precision
    accuracy = calculate_accuracy(y_expected, y_pred)
    recall,precision = recall_precision_fn(y_pred, y_expected)

    print(f"Test #{i}, test loss = {loss:.3f}, test accuracy = {accuracy:.3f}, test recall = {recall:.3f}, test precision = {precision:.3f}")
    print(y_pred[y_expected==1])
    print(f"min value of non-epitope: {min(y_pred[y_expected==0]).item():.3f}")
    print(f"max value of non-epitope: {max(y_pred[y_expected==0]).item():.3f}")
    break