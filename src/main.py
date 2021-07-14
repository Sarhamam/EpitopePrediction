import click
from Bio import SeqIO
from boltons import iterutils
from Bio.SeqUtils.ProtParam import ProteinAnalysis


@click.command()
@click.option('--input', help="Input file (Sequence))")
def cli_main(input_file):
    main(input_file)


def main(input_file):
    parsed_input = read_fasta(input_file)
    sequence_properties = calculate_properties(seq=parsed_input[2], window=3)
    


def read_fasta(input_file):
    """Reads fasta file.
    :return format is a list of SeqRecords"""
    with open(input_file) as handle:
        parsed_file = SeqIO.parse(handle, "fasta")
        records = [r for r in parsed_file.records]

    return records


def _calculate_scale(seq, window, scale):
    analyzed_protein = ProteinAnalysis(str(seq.seq).lower())
    res = analyzed_protein.protein_scale(window=window, param_dict=scale)
    return res


def calculate_polarity(seq, window):
    """ Calculates sequence polarity with respect to the given window.
    :param seq SeqRecord representing the protein sequence
    :param window to calculate the polarity
    """
    # Article: Comparing the polarities of the amino acids: Side-chain distribution coefficients between the vapor phase, cyclohexane, 1-octanol, and neutral aqueous solution
    POLARITY = {'A': -0.06, 'R': -0.84, 'N': -0.48, 'D': -0.80, 'C': 1.36, 'Q': -0.73,
                'E': -0.77, 'G': -0.41, 'H': 0.49, 'I': 1.31, 'L': 1.21, 'K': -1.18,
                'M': 1.27, 'F': 1.27, 'P': 0.0, 'S': -0.50, 'T': -0.27, 'W': 0.88,
                'Y': 0.33, 'V': 1.0}
    return _calculate_scale(seq, window, POLARITY)


def calculate_hydrophobicity(seq, window):
    """ Calculates sequence hydrophobicity with respect to the given window.
    :param seq SeqRecord representing the protein sequence
    :param window to calculate the polarity
    """
    # Article: Correlation of sequence hydrophobicities measures similarity in three-dimensional protein structure
    HYDROPHOBICITY = {'A': -0.40, 'R': -0.59, 'N': -0.92, 'D': -1.31, 'C': 0.17, 'Q': -0.91,
                      'E': -1.22, 'G': -0.67, 'H': -0.64, 'I': 1.25, 'L': 1.22, 'K': -0.67,
                      'M': 1.02, 'F': 1.92, 'P': -0.49, 'S': -0.55, 'T': -0.28, 'W': 0.50,
                      'Y': 1.67, 'V': 0.91}
    return _calculate_scale(seq, window, HYDROPHOBICITY)


def calculate_volume(seq, window):
    """ Calculates sequence volume with respect to the given window.
    :param seq SeqRecord representing the protein sequence
    :param window to calculate the polarity
    """
    # Article: On the average hydrophobicity of proteins and the relation between it and protein structure
    VOLUME = {'A': 52.6, 'R': 109.1, 'N': 75.7, 'D': 68.4, 'C': 68.3, 'Q': 89.7,
              'E': 84.7, 'G': 36.3, 'H': 91.9, 'I': 102.0, 'L': 102.0, 'K': 105.1,
              'M': 97.7, 'F': 113.9, 'P': 73.6, 'S': 54.9, 'T': 71.2, 'W': 135.4,
              'Y': 116.2, 'V': 85.1, 'X': 52.6, 'Z': 52.6, 'B': 52.6, 'J': 102.0}
    return _calculate_scale(seq, window, VOLUME)


def calculate_properties(seq, window):
    """

    :param seq: Protein sequence (SeqRecord)
    :param window: Window to calculate properties on
    :return: A list of preprocessed sequence, each element is a PreprocessedAminoAcidWindow,
    containing the volume, hydrophobicity, polarity, and type.
    """
    preprocessed_sequence = []
    volume = calculate_volume(seq, window)
    polarity = calculate_polarity(seq, window)
    hydrophobicity = calculate_hydrophobicity(seq, window)
    window_sequence = iterutils.windowed(str(seq.seq), window)
    # Iterate with a sliding window
    for i in range(len(window_sequence)):  # Chunk the sequence into groups of size "window"
        aa_group = PreprocessedAminoAcidWindow(type=window_sequence[i],
                                               volume=volume[i],
                                               polarity=polarity[i],
                                               hydrophobicity=hydrophobicity[i])
        preprocessed_sequence.append(aa_group)

    # DEBUG
    assert i + 1 == len(volume) == len(polarity) == len(hydrophobicity)
    return preprocessed_sequence


class PreprocessedAminoAcidWindow():
    def __init__(self, type, volume, polarity, hydrophobicity):
        self.type = type
        self.volume = volume
        self.polarity = polarity
        self.hydrophobicity = hydrophobicity
        self.size = len(type)


if __name__ == '__main__':
    main("../resources/iedb_linear_epitopes.fasta")
