from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np


def get_unique(y):
    y = np.array(y)
    return np.unique(y)


# Display the Confusion Matrix
def display_CM(y_pred, y_true):

    labels = get_unique(y_pred + y_true)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()

    disp.ax_.set(
        title='Confusion Matrix',
        xlabel='Predicted',
        ylabel='Actual')

    plt.show()
