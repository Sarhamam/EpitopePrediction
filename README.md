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
PYTHONPATH=/home/iscb/wolfson/bio3d_ws_2021b/sarhamam/project/src:$PYTHONPATH
```

### Outside of TAU Nova Server

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
  --help                      Show this message and exit.
```

Note that changing any of the optionals will require you to train and predict the network with the same flags.

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

Runtime logs are located at `$PROJECT_FOLDER/logs/` and are printed to stdout.
