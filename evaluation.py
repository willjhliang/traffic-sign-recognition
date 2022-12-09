
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tabulate import tabulate
import pandas as pd


def generate_confusion_matrix(y_actual, y_pred, labels):
    """Generates a confusion matrix for the given labels and predictions."""
    mat = confusion_matrix(y_actual, y_pred)
    plt.figure(figsize = (30, 30))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = labels['Name'], yticklabels = labels['Name'])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
 
 
def get_classification_report(y_actual, y_pred, labels, print_df=False):
    """Reports evaluation metrics for the given labels and predictions."""
    report = classification_report(y_actual, y_pred, target_names = labels['Name'], output_dict = True)
    mat = confusion_matrix(y_actual, y_pred)
    class_accuracies = mat.diagonal()/(mat.sum(axis = 1))
    for index, class_name in enumerate(labels['Name']):
        report[class_name]['accuracy'] = class_accuracies[index]

    df = pd.DataFrame.from_dict(report).T
    if print_df:
        print(tabulate(df, headers = ['Label', 'Precision', 'Recall', 'F1 Score', 'Support', 'Accuracy'], tablefmt = 'fancy_grid'))

    return report, df


def generate_class_comparison(report, metric, labels):
    """Compares metrics for each class."""
    classes = list(labels['Name'])
    values = [report[cls][metric] for cls in classes]
    plt.xticks([], [])
    plt.bar(classes, values)
    plt.show()