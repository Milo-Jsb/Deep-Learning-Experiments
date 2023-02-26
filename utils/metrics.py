import torch
import numpy as np

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import utils.plots

def print_metrics_full_lc(y_test, y_pred, y_pred_prob, loss ,dict_labels ,one_hot):
    
    # Pasamos a formato one-hot si es necesario
    if one_hot:
        y_true_onehot = y_test
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true_onehot = torch.nn.functional.one_hot(y_test)
        y_true = y_test
    

    # Reemplazamos por strs para visualizar más tarde
    y_true, y_pred = utils.plots.replace_for_visualization(y_true, y_pred, dict_labels)

    # Calculamos las métricas seleccionadas
    #auc          = roc_auc_score(y_true_onehot, y_pred_prob, average='macro')
    acc          = accuracy_score(y_true, y_pred)
    acc_balanced = balanced_accuracy_score(y_true, y_pred)
    prec_macro   = precision_score(y_true, y_pred, average='macro')
    rec_macro    = recall_score(y_true, y_pred, average='macro')
    f1           = f1_score(y_true, y_pred, average='macro')
    report       = classification_report(y_true, y_pred, digits=3)

    # Imprimimos
    print('Loss:', loss)
    print("Accuracy:", "%0.5f" % acc)
    print("Accuracy balanced:","%0.5f" %  acc_balanced)   
    #print('AUC:', auc)
    print("macro precision: ","%0.5f" %  prec_macro)
    print("macro recall: ","%0.5f" %  rec_macro)
    print("macro F1: ","%0.5f" %  f1)

    print(f'\n{report}')

def print_metrics_pbp_lc(y_test, y_pred, loss, one_hot):
    
    if one_hot:
        y_true_onehot = y_test
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true_onehot = torch.nn.functional.one_hot(y_test)
        y_true = y_test

    y_test        = y_test.flatten()
    y_pred        = y_pred.flatten()
    
    acc          = accuracy_score(y_test, y_pred)
    acc_balanced = balanced_accuracy_score(y_test, y_pred)
    prec_macro   = precision_score(y_test, y_pred, average = 'macro')
    rec_macro    = recall_score(y_test, y_pred, average = 'macro')
    f1           = f1_score(y_test, y_pred, average = 'macro')
    report       = classification_report(y_test, y_pred, digits=3)

    print('Loss:', loss)
    print("Accuracy:", "%0.5f" % acc)
    print("Accuracy balanced:","%0.5f" %  acc_balanced)   
    #print('AUC:', auc)
    print("macro precision: ","%0.5f" %  prec_macro)
    print("macro recall: ","%0.5f" %  rec_macro)
    print("macro F1: ","%0.5f" %  f1)

    print(f'\n{report}')

def get_metrics(y_true, y_pred):
    
    acc     = accuracy_score(y_true, y_pred)
    acc_bal = balanced_accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, average = 'macro')
    rec     = recall_score(y_true, y_pred, average = 'macro')
    f1      = f1_score(y_true, y_pred, average = 'macro')
    cm      = confusion_matrix(y_true, y_pred)

    return acc, acc_bal ,prec, rec, f1, cm
    