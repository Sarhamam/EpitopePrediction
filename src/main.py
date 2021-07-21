import click

try:
    import netsurfp2 as nsp
    import netsurfp2.model as nsp_model
    import netsurfp2.preprocess as nsp_preprocess

    NETSURF_AVAILABLE = True
except ImportError:
    NETSURF_AVAILABLE = False

from Bio import SeqIO
from boltons import iterutils
from Bio.SeqUtils.ProtParam import ProteinAnalysis

UNICLUST_PATH_ON_UNI_SERVER = "/home/iscb/wolfson/sequence_database/uniclust30_2018_08_copy2/uniclust30_2018_08"
HHSUIT_PATH_ON_UNI_SERVER = "/specific/a/home/cc/students/csguests/sarhamam/sadna2021/models/hhsuite.pb"


@click.command()
@click.option('--input', help="Input file (Sequence))")
def cli_main(input_file):
    main(input_file)


def main(input_file):
    parsed_input = read_fasta(input_file)
    netsurf_searcher, netsurf_model = init_netsurf_model(path_to_hhblits=UNICLUST_PATH_ON_UNI_SERVER)
    sequence_properties = calculate_properties(sequences=parsed_input, window=3, netsurf_searcher=netsurf_searcher,
                                               netsurf_model=netsurf_model)


def init_netsurf_model(path_to_hhblits):
    if not NETSURF_AVAILABLE:
        # Do not crash if NSP is unavailable
        return None, None

    searcher = nsp_preprocess.HHblits(path_to_hhblits)
    netsurf_model = nsp_model.TfGraphModel.load_graph(
        HHSUIT_PATH_ON_UNI_SERVER)  # Unloads the HHblits NetSurfP-2.0 model
    return searcher, netsurf_model


def read_fasta(input_file):
    """Reads fasta file.
    :return format is a list of dictionaries with key - sequence name and value - a list of [description, sequence]"""
    with open(input_file) as handle:
        parsed_file = SeqIO.parse(handle, "fasta")
        records = {record.name: [record.description, str(record.seq)] for record in parsed_file.records}

    return records


def _calculate_scale(seq, window, scale):
    analyzed_protein = ProteinAnalysis(seq.lower())
    res = analyzed_protein.protein_scale(window=window, param_dict=scale)
    return res


def calculate_polarity(seq, window):
    """ Calculates sequence polarity with respect to the given window.
    :param seq string representing the protein sequence
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
    :param seq string representing the protein sequence
    :param window to calculate the hydrophobicity
    """
    # Article: Correlation of sequence hydrophobicities measures similarity in three-dimensional protein structure
    HYDROPHOBICITY = {'A': -0.40, 'R': -0.59, 'N': -0.92, 'D': -1.31, 'C': 0.17, 'Q': -0.91,
                      'E': -1.22, 'G': -0.67, 'H': -0.64, 'I': 1.25, 'L': 1.22, 'K': -0.67,
                      'M': 1.02, 'F': 1.92, 'P': -0.49, 'S': -0.55, 'T': -0.28, 'W': 0.50,
                      'Y': 1.67, 'V': 0.91}
    return _calculate_scale(seq, window, HYDROPHOBICITY)


def calculate_volume(seq, window):
    """ Calculates sequence volume with respect to the given window.
    :param seq string representing the protein sequence
    :param window to calculate the volume
    """
    # Article: On the average hydrophobicity of proteins and the relation between it and protein structure
    VOLUME = {'A': 52.6, 'R': 109.1, 'N': 75.7, 'D': 68.4, 'C': 68.3, 'Q': 89.7,
              'E': 84.7, 'G': 36.3, 'H': 91.9, 'I': 102.0, 'L': 102.0, 'K': 105.1,
              'M': 97.7, 'F': 113.9, 'P': 73.6, 'S': 54.9, 'T': 71.2, 'W': 135.4,
              'Y': 116.2, 'V': 85.1, 'X': 52.6, 'Z': 52.6, 'B': 52.6, 'J': 102.0}
    return _calculate_scale(seq, window, VOLUME)


def calculate_properties(sequences, window, netsurf_searcher, netsurf_model):
    """
    :param sequences: Protein sequences (dictionary with key name and value [description, sequence] )
    :param window: Window to calculate properties on
    :return: A list of preprocessed sequence, each element is a PreprocessedAminoAcidWindow,
    containing the volume, hydrophobicity, polarity, and type.
    """
    netsurf_predictions = calculate_netsurf_properties(sequences, window, netsurf_searcher=netsurf_searcher,
                                                       netsurf_model=netsurf_model)
    result = {}
    for seq_name, seq_details in sequences.items():
        preprocessed_sequence = []
        _, seq = seq_details
        seq = seq.lower()
        volume = calculate_volume(seq, window)
        polarity = calculate_polarity(seq, window)
        hydrophobicity = calculate_hydrophobicity(seq, window)
        window_sequence = iterutils.windowed(seq, window)
        # Iterate with a sliding window
        for i in range(len(window_sequence)):  # Chunk the sequence into groups of size "window"
            aa_group = PreprocessedAminoAcidWindow(type=window_sequence[i],
                                                   volume=volume[i],
                                                   polarity=polarity[i],
                                                   hydrophobicity=hydrophobicity[i],
                                                   rsa=netsurf_predictions[seq_name]['rsa'][i:i + 3],
                                                   q8=netsurf_predictions[seq_name]['q8'][i:i + 3])
            preprocessed_sequence.append(aa_group)

        # DEBUG
        assert i + 1 == len(volume) == len(polarity) == len(hydrophobicity)
        assert i + 3 == len(netsurf_predictions[seq_name]['rsa']) == len(netsurf_predictions[seq_name]['q8'])
        result[seq_name] = preprocessed_sequence
        # Important Note : this can be SMALLER than the length of sequence. for example:
        # we can have a sequence of 171 AAs and the result will only contain 169 windows.
        # We might need to consider padding.

    return result


class PreprocessedAminoAcidWindow(object):
    def __init__(self, type, volume, polarity, hydrophobicity, rsa, q8):
        self.q8 = q8
        self.rsa = rsa
        self.type = type
        self.volume = volume
        self.polarity = polarity
        self.size = len(self.type)
        self.hydrophobicity = hydrophobicity


def calculate_netsurf_properties(sequences, window, netsurf_searcher, netsurf_model, outdir=None):
    """
    Calculates RSA and SS of the sequence with respect to the given window.
    :param sequences SeqRecord respresenting the protein sequence
    :param window to calculate properties with
    :param netsurf_searcher searcher object from netsurfP2 package
    :param netsurf_model model object from netsurfP2 package
    :param outdir output dir for netsurfp2
    """
    if not NETSURF_AVAILABLE:
        return None

    if not outdir:
        outdir = "./outdir_netsurf"
    for name in sequences:
        sequences[name][1] = sequences[name][1].upper()

    profiles = netsurf_searcher(sequences, outdir)
    results = netsurf_model.predict(profiles, outdir, batch_size=window)
    # Results in a list of dictionaries, keys: id,desc,seq,n,rsa,asa,phi,psi,disorder, interface, q3, q8
    results = {res['id']: res for res in results}  # This dictionary will have rsa, q3, q8, and q3_prob, q8_prob,phi,psi
    return results


if __name__ == '__main__':
    main("../resources/test.fasta")
