import csv
import os
import time
import torch
import logging

import numpy as np

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from utils import config
from utils.network_utils import calculate_accuracy, recall_precision_fn, collate_fn, prepare_for_crossentropy_loss

logger = logging.getLogger("NetworkTraining")

MAX_LENGTH = config["NETWORK"].getint("MAX_LENGTH")


# Train function
#################


def train(device, model, optimizer, loss_fn, train_dataloader, test_dataset, window_size, window_overlap, loss_at_end,
          accuracy_report, max_epochs=100, max_batches=200, max_length=10000):
    avg_train_loss = []
    avg_train_acc = []

    avg_train_recall = []
    avg_train_precision = []

    avg_test_loss = []
    avg_test_acc = []

    start_time = time.time()
    csvfile = open(accuracy_report, 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter=' ')
    csvwriter.writerow(["Epoch #", "Train loss", "Train accuracy", "Train recall", "Train precision", "Epoch time",
                        "Total time", "Test loss", "Test accuracy", "Test recall", "Test precision"])
    for epoch_idx in range(max_epochs):
        logger.info("Running epoch %s out of %s", epoch_idx + 1, max_epochs)
        train_loss, train_acc, train_recall, train_precision = 0, 0, 0, 0
        j = 0
        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            X, p, y, og_size = batch['Sequence'], batch['Properties'], batch['Tags'], batch['Original-Size']
            y = y.type(torch.FloatTensor).to(device)

            og_size = [min(og_size[i], max_length) for i in range(len(og_size))]
            if (window_size == -1):
                window_size = max_length
                window_overlap = 0

            if loss_at_end == True:
                y_expected, y_pred = run_model_by_slice(device, model, X, p, y, og_size, window_size, window_overlap, not(loss_fn.weight is None) )
                y_expected = y_expected.unsqueeze(-1)
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_expected)
                loss.backward()

                # Weight updates
                optimizer.step()
                train_loss += loss.item()

                # Accuracy, Recall, Precision
                accuracy = calculate_accuracy(y_expected, y_pred)
                train_acc += accuracy
                recall, precision = recall_precision_fn(y_pred, y_expected)
                train_recall += recall
                train_precision += precision

                # print(y_pred[y_expected == 1, 0])
                # print("min:", min(y_pred[y_expected == 0, 0]))
                # print("max:", max(y_pred[y_expected == 0, 0]))

            else:
                win_shift = window_size - window_overlap
                n_shifts = int(np.ceil(1.0 * (max(og_size) - window_size) / win_shift))  # need to check
                for i_shift in range(n_shifts + 1):
                    bs = range(len(og_size))
                    Li = i_shift * win_shift  # left index
                    Ri = Li + window_size  # right index
                    Ri = min(Ri, max(og_size))
                    size_W = [max(min(og_size[i] - Li, window_size), 0) for i in bs]
                    non_empty = torch.as_tensor(size_W) > 0
                    size_W = torch.as_tensor(size_W)
                    size_W = size_W[non_empty]

                    X_W = X[Li:Ri, non_empty]
                    p_W = p[Li:Ri, non_empty]
                    y_W = y[Li:Ri, non_empty]

                    # Forward pass
                    optimizer.zero_grad()
                    y_pred_W = model(X_W, p_W, size_W)
                    y_expected_W = pack_padded_sequence(y_W.T, size_W, batch_first=True,
                                                        enforce_sorted=False).data.unsqueeze(-1)
                    if not(loss_fn.weight is None):
                        y_pred_W = prepare_for_crossentropy_loss(y_pred_W)
                        y_expected_W = y_expected_W.type(torch.LongTensor).to(device)

                    loss = loss_fn(y_pred_W, y_expected_W)

                    # Weight updates
                    loss.backward()
                    optimizer.step()

                    # Contribute to average loss, accuracy etc.
                    train_loss += loss.item() / (n_shifts + 1)
                    accuracy = calculate_accuracy(y_expected_W, y_pred_W)
                    train_acc += accuracy / (n_shifts + 1)
                    recall, precision = recall_precision_fn(y_pred_W, y_expected_W)
                    train_recall += recall / (n_shifts + 1)
                    train_precision += precision / (n_shifts + 1)

            j += 1

            if batch_idx == max_batches - 1:
                break

        # avg batch metrics after each epoch (j total batches):
        avg_train_loss.append(train_loss / j)
        avg_train_acc.append(train_acc / j)
        avg_train_recall.append(train_recall / j)
        avg_train_precision.append(train_precision / j)
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        logger.info(f"Epoch #{epoch_idx+1}, train loss = {train_loss / j:.3f}, train accuracy = {train_acc / j:.3f},"
                    f" train recall % = {train_recall / j:.1f}, train precision % = {train_precision / j:.1f},"
                    f" epoch_time={epoch_time:.1f} sec, total_time={total_time:.1f} sec")

        # test network on the test dataset
        test_start_time = time.time()
        test_loss, test_acc, test_recall, test_precision = test(device, model, loss_fn, test_dataset)
        avg_test_loss.append(test_loss)
        avg_test_acc.append(test_acc)
        # Save each epoch's weights, accuracy loss and precision
        if not os.path.exists("train_results"):
            os.mkdir("train_results")
        torch.save(model.state_dict(), f"./train_results/weights_{epoch_idx+1}")
        csvwriter.writerow(
            [epoch_idx+1, train_loss / j, train_acc.item() / j, train_recall / j, train_precision / j, epoch_time,
             total_time, test_loss, test_acc, test_recall, test_precision])
        logger.info(
            f"\t  test loss = {test_loss:.3f}, test accuracy = {test_acc:.3f}, test_time={time.time()-test_start_time:.1f} sec")

    np.savetxt("total_loss.csv", np.asarray(avg_train_loss), delimiter=",")

    return avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc


def test(device, model, loss_fn, dataset):
    " run on test data and get loss and accuracy "
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=collate_fn)
    test_loss = 0
    test_acc = 0
    test_recall = 0
    test_precision = 0
    j = 0
    for test_idx, test_row in enumerate(dataloader):
        X, p, y, og_size = test_row['Sequence'], test_row['Properties'], test_row['Tags'], test_row['Original-Size']
        y = y.type(torch.FloatTensor).to(device)

        # predict
        y_pred = model(X, p, og_size)
        y_expected = pack_padded_sequence(y.T, og_size, batch_first=True, enforce_sorted=False).data.unsqueeze(-1)

        if not(loss_fn.weight is None):
            y_pred = prepare_for_crossentropy_loss(y_pred)
            y_expected = y_expected.type(torch.LongTensor).to(device)

        # loss
        loss = loss_fn(y_pred, y_expected)
        loss = loss.item()
        test_loss += loss

        # Accuracy
        accuracy = calculate_accuracy(y_expected, y_pred)
        test_acc += accuracy
        recall, precision = recall_precision_fn(y_pred, y_expected)
        test_recall += recall
        test_precision += precision

        j += 1

    return test_loss / j, test_acc / j, test_recall / j, test_precision / j


# Auxiliary functions
#####################

def run_model_by_slice(device, model, X, p, y, og_size, win_size, win_overlap,weighted_loss):
    win_shift = win_size - win_overlap
    y_pred = torch.empty(0, device=device)
    y_expected = torch.empty(0, device=device)
    y = y.to(device)
    num_seq = X.shape[1]
    for i in range(num_seq):  # for each sequence
        seq_len = og_size[i]
        num_slices = int(np.floor((seq_len - win_size) / win_shift) + 1)
        Li = 0
        Ri = win_size
        size_w = torch.as_tensor([win_size])
        for k in range(num_slices - 1):  # without last slice
            ans = model(X[Li:Ri, i:i + 1], p[Li:Ri, i:i + 1, :], size_w)
            y_pred = torch.cat((y_pred, ans))
            y_expected = torch.cat((y_expected, y[Li:Ri, i]))
            Li += win_shift
            Ri += win_shift
        # last slice
        size_w = torch.as_tensor([seq_len - Li])
        ans = model(X[Li:seq_len, i:i + 1], p[Li:seq_len, i:i + 1, :], size_w)
        y_pred = torch.cat((y_pred, ans))
        y_expected = torch.cat((y_expected, y[Li:seq_len, i]))
    
    if weighted_loss:
        y_pred = prepare_for_crossentropy_loss(y_pred)
        y_expected = y_expected.type(torch.LongTensor).to(device)

    return y_expected, y_pred
