import os
import sys
import json
import click
import torch
import logging
from shutil import copyfile

from data_enricher import data_enricher
from network import init_model, train_model, predict
from utils.network_utils import get_device, create_dataset, print_results

logger = logging.getLogger("EpitopePrediction")


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('mode', type=click.Choice(['train', 'predict']))
@click.option('--weights', type=click.Path(exists=True),
              help="Path of saved weights created by train mode with the same parameters",
              default="./resources/weights.pytw")
@click.option('--rnn_type', type=click.Choice(['LSTM', 'GRU']), help="Type of network to run", default='GRU')
@click.option('--bidirectional', type=bool, help="Bidirectional RNN", default=True)
@click.option('--batch_size', type=int, help="Batch size", default=10)
@click.option('--embed_size', type=int, help="Embedding size", default=30)
@click.option('--concat_after', type=bool, help="Concat numerical properties with RNN output", default=False)
@click.option('--window_size', type=int, help="Window size", default=0)
@click.option('--window_overlap', type=int, help="Window overlap", default=0)
@click.option('--loss_at_end', type=bool, help="Calculates loss after batch (instead of after window)", default=False)
@click.option('--epochs', type=int, help="Number of epochs to train", default=15)
@click.option('--max_batches', type=int, help="Number of maximum batches (-1 is unlimited)", default=-1)
@click.option('--max_length', type=int, help="Max truncated sequences length", default=10000)
@click.option('--hidden_dim', type=int, help="RNN hidden dimensions", default=128)
@click.option('--n_layers', type=int, help="RNN number of layers", default=2)
@click.option('--lr', type=click.FloatRange(1e-6, 1e-1, clamp=True), help="Learning rate", default=5e-4)
@click.option('--numeric_features', type=bool, help="Include numeric features", default=True)
@click.option('--dont_print', is_flag=True, help="Dont print results to stdout")
@click.option('--accuracy_report', type=click.Path(exists=False),
              help="CSV report containing loss and accuracy per epoch", default="report.csv")
@click.option('--weighted_loss', type=bool, help="Use weighted loss function instead of BCE", default=False)
@click.option('--deterministic', type=bool, help="Deterministic with no shuffle of training data set (for debugging)", default=False)
def cli_main(input_file, output_file, mode, weights, rnn_type, bidirectional, batch_size, embed_size, concat_after, window_size,
             window_overlap, loss_at_end, epochs, max_batches, max_length, hidden_dim, n_layers, lr, numeric_features,
             dont_print, accuracy_report,weighted_loss,deterministic):
    try:
        _, file_type = os.path.splitext(input_file)
        if file_type == '.fasta':
            parsed_data = data_enricher(input_file)
            with open("./in.parsed", "w") as f:
                json.dump(parsed_data, f)
        if file_type == '.tsv':
            copyfile(input_file,"./in.parsed")

    except Exception:
        message = "Failed parsing input file. Please make sure the file is a proper FASTA file."
        logger.exception(message)
        sys.exit(message)

    device = get_device()
    if device == 'cuda':
        logger.info(f"Detected CUDA on device #{torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
    else:
        logger.info('Using device: %s\n', device)

    model, optimizer, loss_fn = init_model(device, rnn_type, bidirectional, concat_after, hidden_dim, n_layers, lr,
                                           numeric_features, weighted_loss, deterministic)
    loss_fn.to(device)
    if window_size == 0:
        window_size = -1
    if mode == 'train':
        if deterministic:
            # make reproducible
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(1)
        model.train()
        model.to(device)
        train_data, test_data = create_dataset("./in.parsed", deterministic=deterministic)
        train_loss, train_acc, test_loss, test_acc = train_model(device, model, optimizer, loss_fn, train_data,
                                                                 test_data,
                                                                 epochs, batch_size, window_size, window_overlap,
                                                                 loss_at_end,
                                                                 max_batches, max_length, accuracy_report, deterministic)
        logger.info("Training complete. Average training loss is %s", train_loss[-1])
        logger.info("Saving weights to %s", output_file)
        torch.save(model.state_dict(), output_file)
    else:  # Predict mode
        model.load_state_dict(torch.load(weights))
        model.to(device)
        model.eval()
        test_data = create_dataset("./in.parsed", predict=True)
        results = predict(model, test_data)
        with open(output_file, "w") as f:
            json.dump(results, f)

        if not dont_print:
            print_results(parsed_data, results)
            plot_results(parsed_data, results)


if __name__ == '__main__':
    cli_main()
