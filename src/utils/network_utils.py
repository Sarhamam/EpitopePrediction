import numpy as np

def calculate_accuracy(true_cls, pred_cls):
    """
    Calculates the prediction accuracy of a single sequence
    :param true_cls - np.array representing the true classification (of the format (0,0,0,1,1,0,0) where 1 means its in the epitope and 0 not)
    :param pred_cls - np.array representing the predicted probable classification, each entry a float in the range [0,1]
    Both vectors are expected to have the same length
    ## Current implementation treats a float under 0.5 as 0 and above as 1. We could think of a different method if needed ##
    """
    assert len(true_cls) == len(pred_cls)
    diff = np.abs(true_cls - pred_cls)
    corrects = sum(diff < 0.5)

    return corrects / len(diff)