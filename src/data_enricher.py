"""
This script is used to preprocess the fasta file into a JSON file with the following schema:

NUM
    ID
    Sequence
    Properties
    Tags

(Example:
0
    7
    [m, ]
    [hydrophobicity, polarity, volume, rsa, q3]
    [0, ]  <--- 0 or 1, if in epitope or not

"""
import re
import json
import logging
import numpy as np

from utils import config
from utils.protein_properties import read_fasta, calculate_properties

logger = logging.getLogger("DataEnricher")


def merge_seq(input):
    res = dict()
    for key in input.keys():
        name, seq = input[key]
        low_seq = seq.lower()
        new_seq = np.array(list(seq))
        try:
            curr_seq = res[low_seq][0]
            idx = np.where(new_seq != curr_seq)
            curr_seq[idx] = np.char.upper(curr_seq[idx])
        except:
            res[low_seq] = (new_seq, name)
    ret_dict = {name: [name, ''.join(seq)] for seq, name in res.values()}

    return ret_dict


def data_enricher(input_file):
    parsed_input = read_fasta(input_file)
    parsed_input = merge_seq(parsed_input)
    sequence_properties = calculate_properties(sequences=parsed_input, window=1)
    d = {}
    num = 0
    for ident, sequence in sequence_properties.items():
        full_sequence = []
        full_properties = []
        tags = []
        for i in range(len(sequence)):  # Iterate with i because we also need to access parsed_input
            type, properties = sequence[i].encode()
            full_sequence.append(type)
            full_properties.append(properties)
            tags.append(1 if parsed_input[ident][1][i].isupper() else 0)

        d[num] = {"ID": ident, "Sequence": full_sequence, "Properties": full_properties, "Tags": tags}
        num += 1
    return d


if __name__ == "__main__":
    output_file = open(config["TRAIN"]["JSON"], 'w')
    data = data_enricher(input_file=config["TRAIN"]["FASTA"])
    json.dump(data, output_file)
    output_file.close()
