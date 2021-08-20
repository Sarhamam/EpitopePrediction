import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


word_to_ix = {}
tag_to_ix = {"Y": 1, "N": 0}  # Assign each tag with a unique index


def init_network(sequences, train_seq):
    training_data = []
    for key in sequences:
        sent = [amino_acid.type[0] for amino_acid in sequences[key]]
        tags = ["Y" if c.isupper() else "N" for c in train_seq[key][1]]
        sent += ['<PAD>'] * (3000 - len(sent))
        tags += ['N'] * (3000 - len(tags))
        training_data.append([sent, tags])

    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

    return training_data


def init_model(training_data):
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    count = 0
    for epoch in range(25):  # again, normally you would NOT do 300 epochs, it is toy data
        count +=1
        print(f"epoch begin {count}")
        avg_accuracy = 0
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)
            if count == 49:
                accuracy = 0
                c = 0
                for i in range(len(tag_scores)):
                    if sentence[i] == "<PAD>":
                        continue

                    accuracy += bool(torch.argmax(tag_scores[i]) == targets[i])
                    c +=1

                accuracy = accuracy / c
                avg_accuracy += accuracy
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        if count == 49:
            print(f"avg accuracy is {avg_accuracy/len(training_data)}\n")

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)
        for idx in range(len(tag_scores)):
            if torch.argmax(tag_scores[idx]) == 1:
                print(training_data[0][0][idx])
