import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import torch
import time
from scipy import signal



def episode_metrics(model, x, rhythm, outputs=None, out_message=False):
    """ arg

    """
    # Rhythmn predictions

    if outputs == None:
        predictions_r, predictions_qa = model(x)

    else:
        predictions_r, predictions_qa = outputs
    predictions_r = predictions_r.cpu().detach().numpy()
    y_predictions = np.argmax(predictions_r, axis=1)
    y_truth = np.argmax(rhythm.cpu().detach().numpy(), axis=1)
    # Confusion_matrix
    cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
    TN, FP, FN, TP = cf.ravel()
    support = TN + FP + FN + TP
    # Sensitivity, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # F1 score
    f1 = metrics.f1_score(y_truth, y_predictions, average=None)
    # selecting f1score for positive case if present
    f1_pos = f1[1] if len(f1) > 1 else f1[0]

    if out_message:
        print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1: " Predicted AF"},
                                      index={0: "True Non-AF", 1: "True AF"}))
        print("Sensitivity/Recall: %0.4f" % TPR)
        print("Specificity: %0.4f" % TNR)
        print("Precision/PPV: %0.4f" % PPV)
        print("Negative predictive value/NPV: %0.2f" % NPV)
        print("False positive rate: %0.4f" % FPR)
        print("False negative rate: %0.4f" % FNR)
        print("F1 score: %0.4f" % f1_pos)
        print('support: ', support)
        print('\n\n\n')
    episode_metrics = [TPR, TNR, PPV, NPV, FPR, FNR, f1_pos, support]
    return episode_metrics


def batch_stat_deepbeat(rhythm_true: torch.Tensor, rhythm_pred: torch.Tensor, prefix='', suffix='') -> dict:
    y_predictions = np.argmax(rhythm_pred.cpu().detach().numpy(), axis=1)
    y_truth = np.argmax(rhythm_true.cpu().detach().numpy(), axis=1)
    cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
    tn, fp, fn, tp = cf.ravel()
    assert ((tn + fp + fn + tp) == len(y_truth))
    return {
        prefix + 'tp' + suffix: tp,
        prefix + 'tn' + suffix: tn,
        prefix + 'fp' + suffix: fp,
        prefix + 'fn' + suffix: fn,
    }


def metrics_from_stat(tp, tn, fp, fn, prefix='', suffix='', output_map_all = None) -> dict:
    # Sensitivity, recall, or true positive rate
    tpr = tp / (tp + fn)
    # Specificity or true negative rate
    tnr = tn / (tn + fp)
    # Precision or positive predictive value
    ppv = tp / (tp + fp)
    # Negative predictive value
    npv = tn / (tn + fn)
    # Fall out or false positive rate
    fpr = fp / (fp + tn)
    # False negative rate
    fnr = fn / (tp + fn)
    f1_score = 2 * tpr * ppv / (tpr + ppv)

    mcc = -9999
    auroc = -9999

    if output_map_all != None:
        output_map_all['true'] = output_map_all['true'].cpu().detach().numpy()
        output_map_all['pred'] = output_map_all['pred'].cpu().detach().numpy()

        mcc = matthews_corrcoef(np.argmax(output_map_all['true'], axis = 1), np.argmax(output_map_all['pred'], axis=1))
        auroc = roc_auc_score(np.argmax(output_map_all['true'], axis = 1), output_map_all['pred'][:,1])


    return {
        prefix + 'TPR' + suffix: tpr,
        prefix + 'TNR' + suffix: tnr,
        prefix + 'FPR' + suffix: fpr,
        prefix + 'FNR' + suffix: fnr,
        prefix + 'PPV' + suffix: ppv,
        prefix + 'NPV' + suffix: npv,
        prefix + 'F1' + suffix: f1_score,
        prefix + 'MCC' + suffix: mcc,
        prefix + 'AUROC' + suffix: auroc
    }


def accumulate_stat(stat_map: dict, tp: int, tn: int, fp: int, fn: int):
    stat_map['tp'] += tp
    stat_map['tn'] += tn
    stat_map['fp'] += fp
    stat_map['fn'] += fn
    return stat_map

def accumulate_responses(output_map_all, y_true, y_pred):
    output_map_all['true'] = torch.cat((output_map_all['true'], y_true))
    y_pred = torch.nn.Softmax(dim = 1)(y_pred)
    output_map_all['pred'] = torch.cat((output_map_all['pred'], y_pred))
    return output_map_all



def print_stat_map(stat_map: dict):
    for key, val in stat_map.items():
        print('%s: %.3f, ' % (key, val), end='')
    print()
