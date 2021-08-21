import json
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torchtext.data

# These should be changed, we use this seed to avoid randomness when debbuging
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True


class AminoAcidDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file) as h:
            self.data = json.load(h)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[str(idx)]


dataset = AminoAcidDataset("/Users/sarhamam/git/sadna/EpitopePrediction/resources/test.tsv")
train_test_ratio = 0.7
train_len = int(len(dataset) * train_test_ratio)
train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset) - train_len])

PAD_LEN = 2000  # Arbitrary value

amino_acids_vocab = build_vocab_from_iterator(
    ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v', 'b', 'j', 'z',
     'x'], specials=['<unk>', '<pad>'])
amino_acids_vocab.set_default_index(amino_acids_vocab["<unk>"])


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


dataloader = DataLoader(dataset,
                        batch_size=2,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=collate_fn)


class EpitopePredictor(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, numeric_feature_dim, hidden_dim, linear_dim, output_dim, n_layers,
                 bidirectional, dropout):
        # Constructor
        super().__init__()
        # lstm layer
        self.lstm = nn.LSTM(vocab_size,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.dropout = nn.Dropout(0.2)
        self.linear_1 = nn.Linear(hidden_dim, linear_dim)
        self.linear_2 = nn.Linear(linear_dim + numeric_feature_dim, output_dim)

        # activation function
        self.activation = nn.Sigmoid()

    def forward(self, sequences, properties, text_lengths):
        packed_sequences = pack_padded_sequence(sequences, text_lengths, batch_first=False, enforce_sorted=False)
        lstm_out, hidden = self.lstm(packed_sequences)
        lstm_out = lstm_out[-1, :, :]
        dropout = self.dropout(lstm_out)
        linear_out = self.linear_1(dropout)
        # Concate with numeric features:
        concat_layer = torch.cat((linear_out, properties, 1))
        linear_2_out = self.linear_2(concat_layer)
        # Final activation function
        outputs = self.activation(linear_2_out)

        return outputs


size_of_vocab = len(amino_acids_vocab)
num_hidden_nodes = 32
num_output_nodes = 2000
num_layers = 2
bidirection = True
dropout = 0.2

# instantiate the model
model = EpitopePredictor(vocab_size=size_of_vocab,
                         numeric_feature_dim=1,
                         linear_dim=1,
                         hidden_dim=num_hidden_nodes,
                         output_dim=num_output_nodes,
                         n_layers=num_layers,
                         bidirectional=True, dropout=dropout)

for i_batch, sample_batched in enumerate(dataloader):
    model(sample_batched['Sequence'], sample_batched['Properties'], sample_batched['Original-Size'])
