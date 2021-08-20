"""
This script is used to preprocess the fasta file into a TSV file with the following headers:

ID    Sequence    Properties    Tags

(Example:
7    [m, ]      [hydrophobicity, polarity, volume, rsa,q8]    [0, ]  <--- 0 or 1, if in epitope or not

Note: We're using TSV Format to be able to work with torchtext.data.TabularData (more convenient than dictionary.)
"""

import csv
from utils import config
from utils.protein_properties import read_fasta, init_netsurf_model, calculate_properties

def data_enricher(output):
    parsed_input = read_fasta(config["TRAIN"]["FASTA"])
    netsurf_searcher, netsurf_model = init_netsurf_model()
    sequence_properties = calculate_properties(sequences=parsed_input, window=1, netsurf_searcher=netsurf_searcher,
                                               netsurf_model=netsurf_model)
    csvfile = open(output, "w")
    writer = csv.writer(csvfile, dialect=csv.excel_tab)  # Delimiter is tab, we have commas in our data.

    # Write headers
    writer.writerow(["ID", "Sequence", "Properties", "Tags"])

    for ident, sequence in sequence_properties.items():
        full_sequence = []
        full_properties = []
        tags = []
        for i in range(len(sequence)):  # Iterate with i because we also need to access parsed_input
            type, properties = sequence[i].encode()
            full_sequence.append(type)
            full_properties.append(properties)
            tags.append(1 if parsed_input[ident][1][i].isupper() else 0)

        writer.writerow([ident, full_sequence, full_properties, tags])


if __name__ == "__main__":
    data_enricher(output=config["TRAIN"]["TSV"])
