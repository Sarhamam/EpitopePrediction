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
    [hydrophobicity, polarity, volume, rsa,q8]
    [0, ]  <--- 0 or 1, if in epitope or not

"""
import json
from utils import config
from utils.protein_properties import read_fasta, init_netsurf_model, calculate_properties
import numpy as np

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


def data_enricher(input_file, output):
    parsed_input = read_fasta(input_file)
    netsurf_searcher, netsurf_model = init_netsurf_model()
    uniques_input = merge_seq(parsed_input)
    print(len(uniques_input))
    sequence_properties = calculate_properties(sequences=uniques_input, window=1, netsurf_searcher=netsurf_searcher,
                                               netsurf_model=netsurf_model)
    d = {}
    output_file = open(output, 'w')
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

    json.dump(d, output_file)
    output_file.close()


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("DataEnricher")
    data_enricher(input_file='/Users/sarhamam/git/sadna/EpitopePrediction/resources/test.fasta',
                  output='/Users/sarhamam/git/sadna/EpitopePrediction/resources/test.tsv')
    # for i in range(1, 307):
    #     logger.debug("Starting to calculate %s iteration", i)
    #     input_file = config["TRAIN"]["FASTA"].format(i=i)
    #     data_enricher(input_file=input_file, output=config["TRAIN"]["TSV"].format(i=i))
