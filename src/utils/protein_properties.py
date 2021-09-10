import logging
from Bio import SeqIO
from utils import config
from boltons import iterutils
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


##################
# Public functions
##################
def read_fasta(input_file):
    """Reads fasta file.
    :return format is a list of dictionaries with key - sequence name and value - a list of [description, sequence]"""
    with open(input_file) as handle:
        parsed_file = SeqIO.parse(handle, "fasta")
        records = {record.name: [record.description, str(record.seq)] for record in parsed_file.records}

    return records


def calculate_properties(sequences, window):
    """
    :param sequences: Protein sequences (dictionary with key name and value [description, sequence] )
    :param window: Window to calculate properties on
    :return: A list of preprocessed sequence, each element is a PreprocessedAminoAcidWindow,
    containing the volume, hydrophobicity, polarity, and type.
    """
    result = {}
    logger.debug("Starting to calculate scale properties")
    STRUCT_WINDOW = 7
    for seq_name, seq_details in sequences.items():
        preprocessed_sequence = []
        _, seq = seq_details
        seq = seq.lower()
        volume = _calculate_volume(seq, window)
        polarity = _calculate_polarity(seq, window)
        hydrophobicity = _calculate_hydrophobicity(seq, window)
        rsa = _calculate_rsa(seq, window)
        q3 = _calculate_secondary_structure(seq, STRUCT_WINDOW)  # secondary structure
        window_sequence = iterutils.windowed(seq, window)
        # Iterate with a sliding window
        for i in range(len(window_sequence)):  # Chunk the sequence into groups of size "window"
            aa_group = PreprocessedAminoAcidWindow(type=window_sequence[i],
                                                   volume=volume[i],
                                                   polarity=polarity[i],
                                                   hydrophobicity=hydrophobicity[i],
                                                   rsa=rsa[i],
                                                   q3=q3[i])  # it doesn't work with windows larger than 1 !!
            preprocessed_sequence.append(aa_group)

        result[seq_name] = preprocessed_sequence
        # Important Note : this can be SMALLER than the length of sequence. for example:
        # we can have a sequence of 171 AAs and the result will only contain 169 windows.
        # We might need to consider padding.
    logger.debug("Scale properties calculated.")

    return result


###################
# Private functions
###################

def _calculate_scale(seq, window, scale):
    X = sum(scale.values()) // len(scale)  # X is any
    B = scale['D'] + scale['N'] / 2  # B is D or N
    J = scale['L'] + scale['I'] / 2  # J is L or I
    Z = scale['Q'] + scale['E'] / 2  # Z is Q or E
    scale['B'] = scale.get('B') or B
    scale['J'] = scale.get('J') or J
    scale['Z'] = scale.get('Z') or Z
    scale['X'] = scale.get('X') or X
    if window == 1:
        return [scale[aa.upper()] for aa in seq]

    analyzed_protein = ProteinAnalysis(seq.lower())
    res = analyzed_protein.protein_scale(window=window, param_dict=scale)
    return res


def _nomalized_data(dic):
    """ Calculates z-score for every value in a dictionary:  (data - data.mean()) / sqrt(data.var())"""
    keys, vals = zip(*dic.items())
    z = stats.zscore(vals)
    newmap = dict(zip(keys, z))
    dic_new = {k: round(v, 2) for k, v in newmap.items()}
    return dic_new


def _calculate_polarity(seq, window):
    """ Calculates sequence polarity with respect to the given window.
    :param seq string representing the protein sequence
    :param window to calculate the polarity
    """
    # Article: Comparing the polarities of the amino acids: Side-chain distribution coefficients between the vapor phase, cyclohexane, 1-octanol, and neutral aqueous solution
    POLARITY = {'A': -0.06, 'R': -0.84, 'N': -0.48, 'D': -0.80, 'C': 1.36, 'Q': -0.73,
                'E': -0.77, 'G': -0.41, 'H': 0.49, 'I': 1.31, 'L': 1.21, 'K': -1.18,
                'M': 1.27, 'F': 1.27, 'P': 0.0, 'S': -0.50, 'T': -0.27, 'W': 0.88,
                'Y': 0.33, 'V': 1.0}
    POLARITY_N = _nomalized_data(POLARITY)
    return _calculate_scale(seq, window, POLARITY_N)


def _calculate_hydrophobicity(seq, window):
    """ Calculates sequence hydrophobicity with respect to the given window.
    :param seq string representing the protein sequence
    :param window to calculate the hydrophobicity
    """
    # Article: Correlation of sequence hydrophobicities measures similarity in three-dimensional protein structure
    HYDROPHOBICITY = {'A': -0.40, 'R': -0.59, 'N': -0.92, 'D': -1.31, 'C': 0.17, 'Q': -0.91,
                      'E': -1.22, 'G': -0.67, 'H': -0.64, 'I': 1.25, 'L': 1.22, 'K': -0.67,
                      'M': 1.02, 'F': 1.92, 'P': -0.49, 'S': -0.55, 'T': -0.28, 'W': 0.50,
                      'Y': 1.67, 'V': 0.91}

    HYDROPHOBICITY_N = _nomalized_data(HYDROPHOBICITY)
    return _calculate_scale(seq, window, HYDROPHOBICITY_N)


def _calculate_volume(seq, window):
    """ Calculates sequence volume with respect to the given window.
    :param seq string representing the protein sequence
    :param window to calculate the volume
    """
    # Article: On the average hydrophobicity of proteins and the relation between it and protein structure
    VOLUME = {'A': 52.6, 'R': 109.1, 'N': 75.7, 'D': 68.4, 'C': 68.3, 'Q': 89.7,
              'E': 84.7, 'G': 36.3, 'H': 91.9, 'I': 102.0, 'L': 102.0, 'K': 105.1,
              'M': 97.7, 'F': 113.9, 'P': 73.6, 'S': 54.9, 'T': 71.2, 'W': 135.4,
              'Y': 116.2, 'V': 85.1}

    VOLUME_N = _nomalized_data(VOLUME)
    return _calculate_scale(seq, window, VOLUME_N)


def _calculate_rsa(seq, window):
    """ Calculates sequence RSA with respect to the given window.
    :param seq string representing the protein sequence
    :param window to calculate the RSA
    """
    # Article: Maximum Allowed Solvent Accessibilites of Residues in Proteins (used theoretical values)
    RSA = {'A': 0.796, 'R': 0.651, 'N': 0.672, 'D': 0.646, 'C': 0.911, 'Q': 0.654,
           'E': 0.605, 'G': 0.749, 'H': 0.741, 'I': 0.876, 'L': 0.861, 'K': 0.565,
           'M': 0.856, 'F': 0.87, 'P': 0.669, 'S': 0.744, 'T': 0.742, 'W': 0.849,
           'Y': 0.818, 'V': 0.864}

    RSA_N = _nomalized_data(RSA)
    return _calculate_scale(seq, window, RSA_N)


def _calculate_secondary_structure(seq, window):
    """ Calculates sequence secondary structure with respect to the given window.
    Each residue gets a secondary structure estimate from the window in which it is in the middle of
    :param seq string representing the protein sequence
    :param window to calculate the secondary structure - WINDOW MUST BE ODD AT THE MOMENT
    """
    # STRUCTS = {0: "Helix", 1: "Turn", 2: "Sheet"}
    window_sequence = iterutils.windowed(seq, window)  # sliding windows
    residue_num = len(seq)
    struct = [0 for _ in range(residue_num)]
    tail = (window - 1) // 2
    for i in range(residue_num):
        if i < tail:  # first few residues don't have a window surrounding them
            sub_seq = ''.join(window_sequence[0])  # instead take the result of the first window for them
        elif residue_num - i <= tail:  # last few residues don't have a window surrounding them
            sub_seq = ''.join(window_sequence[-1])  # instead take the result of the last window for them
        else:
            sub_seq = ''.join(window_sequence[i - tail])
        prob = ProteinAnalysis(sub_seq.upper()).secondary_structure_fraction()
        # here we choose argmax but use randomness in tie situations
        isMax = np.array(list(enumerate(prob == np.max(prob))))  # rows of (index, is max)
        maxIdx = isMax[np.where(isMax[:, 1] == 1)]  # rows of (index, 1) because only max were left
        argMax = np.random.choice(maxIdx[:, 0])  # choose random index
        # struct[i] = STRUCTS[argMax]
        struct[i] = float(argMax)
    return struct


##########
# Classes
##########

class PreprocessedAminoAcidWindow(object):
    def __init__(self, type, volume, polarity, hydrophobicity, rsa, q3):
        self.q3 = q3
        self.rsa = rsa
        self.type = type  # Tuple
        self.volume = volume
        self.polarity = polarity
        self.size = len(self.type)
        self.hydrophobicity = hydrophobicity

    def encode(self):
        """
        Returns a string of "type|hydrophobicity|polarity|volume|rsa|q8|
        """
        # Will we want to truncate the numbers so we will have a constant word length?
        type_as_str = "".join(self.type)
        # q3_as_str = "".join(self.q3)
        properties_to_encode = [self.hydrophobicity, self.polarity, self.volume, self.rsa, self.q3]

        return type_as_str, tuple(p for p in properties_to_encode)
