import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.cpu().numpy()
    y_true = encode_onehot(y_true)
    y_pred = y_preds.cpu().detach().numpy()
    # return roc_auc_score(y_true, y_pred, multi_class="ovo")
    return roc_auc_score(y_true, y_pred)

def f1_score(output, labels):
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    pred_labels = output.max(1)[1].cpu().numpy()
    labels_2 = labels.cpu().numpy()
    return f1_score(labels_2, pred_labels, average='macro')


def prec_recall_n(output, labels, topn):
    preds = output.detach().numpy()[-1]
    pass
