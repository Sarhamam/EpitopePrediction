import json
import torch
import logging
import matplotlib
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from torchtext.vocab import build_vocab_from_iterator

logger = logging.getLogger("NetworkUtils")
matplotlib.use("Agg")
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

def create_dataset(path, train_test_ratio=0.7, predict=False, deterministic=False):
    if deterministic:
        torch.manual_seed(1)
    dataset = AminoAcidDataset(path)
    if predict:
        return dataset
    train_len = int(len(dataset) * train_test_ratio)
    if train_len > 0:
        train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset) - train_len])
    else:
        logger.warning("Dataset is too short to split. Will use the same data for training and testing.")
        test_dataset = dataset
        train_dataset = dataset

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
    if len(batch) == 1:  # Predict
        result['ID'] = batch[0]['ID']
    return result


#  Getters
##########


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
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
    if pred_cls.size(1) > 1:
        pred_cls = pred_cls[:, 1]  # probability of 1
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
    if pred_cls.size(1) > 1:
        pred_cls = pred_cls[:, 1]  # probability of 1
    true_indices = (true_cls == 1)  # indices where TP should be expected
    TP = sum(pred_cls[true_indices] > 0.5)
    FP = sum(pred_cls > 0.5) - TP  # Total predicted P minus TP
    FN = sum(true_indices) - TP  # Total expected TP minus predicted TP
    recall = torch.nan_to_num(TP / (TP + FN))
    precision = torch.nan_to_num(TP / (TP + FP))
    return 100 * recall.item(), 100 * precision.item()


# Auxiliary functions
#####################

def prepare_for_crossentropy_loss(tensor):
    """ Turns 1D Tensor of probabilities to 2D Tensor of probabilites and complements"""
    tensor = tensor.unsqueeze(-1)
    compl = 1 - tensor
    return torch.cat((compl, tensor), 1)  # probablity of 0, probablity of 1


def print_results(parsed_data, results):
    """ Colors the amino acid according to the probabilities in the results"""
    fg = lambda text, color: "\33[38;5;" + str(color) + "m" + text + "\33[0m"
    print("Probabilities:")
    colors = [118, 112, 106, 100, 94, 88]
    probs = ""
    for i in range(6):
        color = colors[i]
        probs += " " + fg("{:.2f}".format(i / 6.0), color)

    print(probs)

    for idx, d in parsed_data.items():
        print(d["ID"])
        probabilities = results[d["ID"]]
        colored_result = ""
        for i in range(len(probabilities)):
            idx = int(probabilities[i] * 6)
            if idx > 5:
                idx = 5
            color = colors[idx]
            colored_result += fg(d["Sequence"][i], color)

        print(colored_result)


def plot_results(parsed_data, results):
    for idx, d in parsed_data.items():
        print(d["ID"])
        probabilities = results[d["ID"]]
        x = list(range(len(probabilities)))

        fig, ax = plt.subplots(1)
        ax.scatter(x, probabilities)
        ax.plot([0, x[-1]], [0.5, 0.5], 'g--')  # plot 50% line
        ax.legend(['50% probability', 'Prediction'], loc='best')

        ax.set_title(f'Epitope prediction - ID {d["ID"]}')
        ax.set_xlabel('Amino acid number')
        ax.set_ylabel('Probability')

        ax.set_xlim(0, x[-1])
        ax.set_ylim(0, 1)

        plt.savefig(f'./{d["ID"]}.png')
