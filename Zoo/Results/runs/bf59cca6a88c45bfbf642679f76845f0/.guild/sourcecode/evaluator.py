import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import tensorflow.keras as k
import json


def evaluate(test_labels, predicted_labels, mode):
    """
    Evalaute the model

    :param test_labels:
    :param predicted_labels:
    :param mode:
    :param prediction_mode:
    :return:
    """
    if len(test_labels.shape) > 1:
        test_labels = np.argmax(test_labels, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(test_labels,
                                                               predicted_labels, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                                           average='macro')

    metrics = {'accuracy': np.sum(test_labels == predicted_labels) / len(test_labels), 'precision': precision.tolist(),
               'recall': recall.tolist(), 'f1': f1.tolist(), 'precision_avg': precision_avg, 'recall_avg': recall_avg,
               'f1_avg': f1_avg}

    with open(mode + '_metrics.json', 'w') as fp:
        json.dump(metrics, fp, sort_keys=True, indent=4)

    return metrics


def save_plots(predicted_labels, predicted_probabilities, true_labels, mode):
    conf_mx = confusion_matrix(true_labels, predicted_labels)
    fig = plt.figure()
    plt.matshow(conf_mx)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(mode + '_confusion_matrix.png')
    plt.close(fig)

    # Compute ROC curve and ROC area for each class
    fig = plt.figure()
    n_classes = len((np.unique(true_labels)))

    if n_classes > 2:
        true_labels_binary = label_binarize(true_labels, classes=np.unique(true_labels))
    else:
        true_labels_binary = k.utils.to_categorical(true_labels, num_classes=n_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_binary[:, i], predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['c', 'm', 'y', 'k', ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(mode + '_roc-auc_curve.png')
    plt.close(fig)

