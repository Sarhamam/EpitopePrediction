import sys
import json
import click
import torch
import logging

from data_enricher import data_enricher
from network import init_model, train_model
from utils.network_utils import get_device, create_dataset

logger = logging.getLogger("EpitopePrediction")


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('mode', type=click.Choice(['train', 'predict']))
@click.option('--weights', type=click.Path(exists=True), default="../resources")
@click.option('--rnn_type', type=click.Choice(['LSTM', 'GRU']), help="Type of network to run", default='LSTM')
@click.option('--bidirectional', type=bool, help="Bidirectional RNN", default=True)
@click.option('--batch_size', type=int, help="Batch size", default=10)
@click.option('--concat_after', type=bool, help="Concat numerical properties with RNN output", default=False)
@click.option('--window_size', type=int, help="Window size", default=-1)
@click.option('--window_overlap', type=int, help="Window overlap", default=0)
@click.option('--loss_at_end', type=bool, help="Calculates loss after batch (instead of after window)", default=True)
@click.option('--epochs', type=int, help="Number of epochs to train", default=1)
@click.option('--max_batches', type=int, help="Number of maximum batches (-1 is unlimited)", default=-1)
@click.option('--max_length', type=int, help="Max truncated sequences length", default=10000)
@click.option('--hidden_dim', type=int, help="RNN hidden dimensions", default=128)
@click.option('--n_layers', type=int, help="RNN number of layers", default=2)
@click.option('--lr', type=click.FloatRange(1e-6, 1e-1, clamp=True), help="Learning rate", default=5e-4)
@click.option('--numeric_features', type=bool, help="Include numeric features", default=True)

def cli_main(input_file, output_file, mode, weights, rnn_type, bidirectional, batch_size, concat_after, window_size,
             window_overlap, loss_at_end, epochs, max_batches, max_length, hidden_dim, n_layers, lr, numeric_features):
    try:
        parsed_data = data_enricher(input_file)
        with open("./in.parsed", "w") as f:
            json.dump(parsed_data, f)

    except Exception as e:
        message = "Failed parsing input file. Please make sure the file is a proper FASTA file."
        logger.exception(message)
        sys.exit(message)

    device = get_device()
    model, optimizer, loss_fn = init_model(device, rnn_type, bidirectional, concat_after, hidden_dim, n_layers, lr, numeric_features)
    if mode == 'train':
        model.train()
        train_data, test_data = create_dataset("./in.parsed")
        train_loss, train_acc, test_loss, test_acc = train_model(device, model, optimizer, loss_fn, train_data, test_data,
                                                                epochs, batch_size, window_size, window_overlap, loss_at_end, 
                                                                max_batches, max_length)
        logger.info("Training complete. Average training loss is %s", train_loss[-1])


if __name__ == '__main__':
    cli_main()