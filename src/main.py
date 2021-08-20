import click

from network import init_network, init_model
from utils.protein_properties import init_netsurf_model, calculate_properties, read_fasta


@click.command()
@click.option('--input', help="Input file (Sequence))")
def cli_main(input_file):
    main(input_file)


def main(input_file):
    parsed_input = read_fasta(input_file)
    netsurf_searcher, netsurf_model = init_netsurf_model()
    sequence_properties = calculate_properties(sequences=parsed_input, window=1, netsurf_searcher=netsurf_searcher,
                                               netsurf_model=netsurf_model)

    training_data = init_network(sequence_properties, parsed_input)  # just to make sure shit works
    init_model(training_data)


if __name__ == '__main__':
    main("../resources/test.fasta")
