import io
import json
import sys
import click
import logging
from data_enricher import data_enricher
from network import init_model, get_device, create_dataset, train_model

logger = logging.getLogger("EpitopePrediction")


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('mode', type=click.Choice(['train', 'predict']))
@click.option('--rnn_type', type=click.Choice(['LSTM', 'GRU']), help="Type of network to run", default='LSTM')
@click.option('--bidirectional', type=bool, help="Bidirectional LSTM", default=True)
@click.option('--batch_size', type=int, help="Batch size", default=10)
@click.option('--concat_after', type=bool, help="Concat numerical properties with LSTM output", default=False)
@click.option('--window_size', type=int, help="Window size", default=256)
@click.option('--window_overlap', type=int, help="Window overlap", default=32)
@click.option('--loss_at_end', type=bool, help="Calculates loss after batch (instead of after window)", default=False)
@click.option('--epochs', type=int, help="Number of epochs to train", default=1)
def cli_main(input_file, output_file, mode, rnn_type, bidirectional, batch_size, concat_after, window_size,
             window_overlap, loss_at_end, epochs):
    try:
        parsed_data = data_enricher(input_file)
        with open("./in.parsed", "w") as f:
            json.dump(parsed_data, f)

    except Exception as e:
        message = "Failed parsing input file. Please make sure the file is a proper FASTA file."
        logger.exception(message)
        sys.exit(message)

    device = get_device()
    model, optimizer, loss_fn = init_model(device, rnn_type, bidirectional, concat_after)
    if mode == 'train':
        model.train()
        train_data, test_data = create_dataset("./in.parsed")
        train_loss, train_acc, test_loss, test_acc = train_model(device, model, optimizer, loss_fn, train_data, test_data,
                                                                epochs, batch_size, window_size, window_overlap, loss_at_end)
        logger.info("Training complete. Average training loss is %s", train_loss[-1])


if __name__ == '__main__':
    cli_main()
