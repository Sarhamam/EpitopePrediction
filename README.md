# Epitope Prediction

EpitopePrediction is a python module that uses LSTM to predict epitopes in linear amino acids sequences.

## Installation

### TAU Nova Server

```bash
# Go to the submission directory
cd /home/iscb/wolfson/bio3d_ws_2021b/sarhamam/
# Alias conda (If you dont have it installed in your account already!
source /home/iscb/wolfson/bio3d_ws_2021b/sarhamam/venv/etc/profile.d/conda.csh
# Activate the conda environment with the requirements installed, or create your own.
conda activate /home/iscb/wolfson/bio3d_ws_2021b/sarhamam/venv/envs
# Add the source directory to your PYTHONPATH
set PYTHONPATH=/home/iscb/wolfson/bio3d_ws_2021b/sarhamam/project/src
```
In case of a permission error, you will have to install conda and the requirements as explained in the Portable section.
### Google Collab

To run easily on Google Collab, copy `Epitope_project_template.ipynb` to your collab notebook
Make sure to fill the appropriate parameters according to the examples in the notebook.

### Portable

Using python version `3.7.10` run

```bash
pip install -r $PROJECT_FOLDER/src/requirements.txt
set PYTHONPATH=$PROJECT_FOLDER/src:$PYTHONPATH
```

Where the project folder is the location the submitted zip file was extracted to.

## Usage

The module has two mods of operations - `predict` and `train`.

```bash
Usage: main.py [OPTIONS] INPUT_FILE OUTPUT_FILE {train|predict}

Options:
  --weights PATH              Path of saved weights created by train mode with
                              the same parameters
  --rnn_type [LSTM|GRU]       Type of network to run
  --bidirectional BOOLEAN     Bidirectional RNN
  --batch_size INTEGER        Batch size
  --embed_size INTEGER        Embedding size
  --concat_after BOOLEAN      Concat numerical properties with RNN output
  --window_size INTEGER       Window size
  --window_overlap INTEGER    Window overlap
  --loss_at_end BOOLEAN       Calculates loss after batch (instead of after
                              window)
  --epochs INTEGER            Number of epochs to train
  --max_batches INTEGER       Number of maximum batches (-1 is unlimited)
  --max_length INTEGER        Max truncated sequences length
  --hidden_dim INTEGER        RNN hidden dimensions
  --n_layers INTEGER          RNN number of layers
  --lr FLOAT RANGE            Learning rate  [1e-06<=x<=0.1]
  --numeric_features BOOLEAN  Include numeric features
  --dont_print                Dont print results to stdout
  --accuracy_report PATH      CSV report containing loss and accuracy per
                              epoch
  --weighted_loss BOOLEAN     Use weighted loss function instead of BCE
  --deterministic BOOLEAN     Deterministic with no shuffle of training data
                              set (for debugging)
  --help                      Show this message and exit.
```

Note that changing any of the optionals will require you to train and predict the network with the same flags.
#### Additional output
When running the network on predict mode, additional output file of a graph of the probabilities will be created for each sequence in the fasta file.

The file name will be `<sequence_id>.png`.

When running the network on training mode, a folder called `train_results` will be created, and will contain the states of the model and optimizer for each epoch. (`optimizer_{epoch_index}, model_{epoch_index}`), This allows choosing the best epoch in the training process.
In addition, a file named `report.csv` will contain the accuracy, loss, precision and recall of the test and train data for each epoch. (Unless `--accuracy-report` parameter given with a different path.)

#### Examples

Run with default weights

```bash
python src/main.py resources/example.fasta output.json predict
````

Train network with different parameters and then predict

```bash
# Train
python src/main.py resources/example.fasta resources/lstm_weights.pytw train --bidirectional False --rnn_type LSTM --epochs 15
# Predict, while making sure to pass the same parameters.
python src/main.py resources/example.fasta output.json predict --weights resources/lstm_weights.pytw --bidirectional False --rnn_type LSTM
````

## Logs

Runtime logs are located at `$PROJECT_FOLDER/logs/` and are printed to stdout and saved to `./logs/epitope_prediction.log`.
