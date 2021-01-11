import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from tabulate import tabulate


class Evaluator:

    @staticmethod
    def lob_evaluator(test_labels, predicted_labels):
        print(f"Test labels shape: {test_labels.shape}")
        print(f"Predicted labels shape: {predicted_labels.shape}")

        test_labels = np.argmax(test_labels, axis=1)
        predicted_labels = np.argmax(predicted_labels, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(test_labels,
                                                                   predicted_labels, average=None)
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                                               average='macro')

        metrics = {'accuracy': np.sum(test_labels == predicted_labels) / len(test_labels), 'precision': precision,
                   'recall': recall, 'f1': f1, 'precision_avg': precision_avg, 'recall_avg': recall_avg,
                   'f1_avg': f1_avg}

        return metrics

    @staticmethod
    def print_metrics(metrics):
        df = pd.DataFrame(data=metrics)
        print(tabulate(df, headers='keys', tablefmt='psql'))
