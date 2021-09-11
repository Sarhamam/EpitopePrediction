import json
import torch
import logging

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from torchtext.vocab import build_vocab_from_iterator

logger = logging.getLogger("NetworkUtils")

# Constants
##########
amino_acids_vocab = build_vocab_from_iterator(
    ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v', 'b', 'j',
     'z',
     'x'], specials=['<unk>', '<pad>'])
amino_acids_vocab.set_default_index(amino_acids_vocab["<unk>"])


# Dataset Definition
#####################


class AminoAcidDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[str(idx)]


# Dataset functions
##################

def create_dataset(path, train_test_ratio=0.7):
    dataset = AminoAcidDataset(path)
    train_len = int(len(dataset) * train_test_ratio)
    train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset) - train_len])
    return train_dataset, test_dataset


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


#  Getters
##########

def get_device():
    if torch.cuda.is_available():
        logger.info(f"Detected CUDA on device #{torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    logger.info('Using device: %s\n', device)
    return device


# Accuracy functions
####################


def calculate_accuracy(true_cls, pred_cls):
    """
    Calculates the AVERAGE prediction accuracy of given sequences (with padding)
    :param true_cls - tensor representing the true classification (of the format (0,0,0,1,1,0,0) where 1 means its in the epitope and 0 not)
    :param pred_cls - tensor representing the predicted probable classification, each entry a float in the range [0,1]
    ## Current implementation treats a float under 0.5 as 0 and above as 1. We could think of a different method if needed ##
    """
    n = pred_cls.shape[0]
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
    true_indices = (true_cls == 1)  # indices where TP should be expected
    TP = sum(pred_cls[true_indices] > 0.5)
    FP = sum(pred_cls > 0.5) - TP  # Total predicted P minus TP
    FN = sum(true_indices) - TP  # Total expected TP minus predicted TP
    recall = torch.nan_to_num(TP / (TP + FN))
    precision = torch.nan_to_num(TP / (TP + FP))
    return 100 * recall.item(), 100 * precision.item()
