import matplotlib.pyplot as plt



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    """
    This function prints and plots the confusion matrix.
    """
    import numpy as np
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(f'{save_path}confusion_matrix.png')
    else:
        plt.show()

def plot_metrics(train_loss, val_loss, accuracy, save_path=None):
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot train loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'g-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params('y', colors='b')

    # Create another y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracy, 'r-', label='Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # Adjust top margin
    plt.title('Train Loss, Validation Loss, and Accuracy', fontsize=16)
    fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98))
    plt.grid(axis='y')
    # plt.show()
    if save_path:
        plt.savefig(f'{save_path}loss_accuracy.png')
    else:
        plt.show()
def test_code():
    # Example usage:
    train_loss = [0.5, 0.4, 0.3, 0.2, 0.1]
    val_loss = [0.6, 0.5, 0.4, 0.3, 0.2]
    accuracy = [0.7, 0.8, 0.85, 0.9, 0.95]

    plot_metrics(train_loss, val_loss, accuracy)

if __name__ == '__main__':
    test_code()
