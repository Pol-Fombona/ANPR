from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Display the Confusion Matrix
def display_CM(y_pred, y_true):

    cm = confusion_matrix(y_true, y_pred)
    cm.plot()
    plt.show()
