import matplotlib.pyplot as plt


def plot_dropout_vs_acc():
    """
    Looks at the improvement made by different values for Dropout on Adam (0.1 -> 0.5)
    :return:
    """
    data = {"10%":0.9810, "20%":0.9797, "30%":0.9798, "40%":0.9781, "50%":0.9773}
    x = list(data.keys())
    y = list(data.values())

    plt.plot(x, y, color='blue', linewidth=2)
    plt.xlabel("Dropout", labelpad=8)
    plt.ylabel("Accuracy")
    plt.title("Plot of Test Accuracy V/s Dropout Value")
    plt.grid(True)

    for i, j in enumerate(y):
        plt.text(i, j, j, ha='center', va='bottom', color='blue', fontsize=12)

    plt.savefig('plots/Test-Accuracy-Vs-Dropout')
    plt.show()
    return

def plot_test_acc_vs_model():
    """
    Looks at the improvement made by different Architectures. Dropout of 0.30 Considered.
    :return:
    """
    data = {"MNIST_V1": 0.9225, "MNIST_V2": 0.9765, "MNIST_Drop": 0.9775, "MNIST_RMSProp": 0.9774, "MNIST_Adam": 0.9798}
    x = list(data.keys())
    y = list(data.values())

    plt.plot(x, y, color='blue', linewidth=2)
    plt.xlabel("Model", labelpad=8)
    plt.ylabel("Accuracy")
    plt.title("Plot of Test Accuracy V/s Model")
    plt.grid(True)

    for i, j in enumerate(y):
        plt.text(i, j, j, ha='center', va='bottom', color='blue', fontsize=12)

    plt.savefig('plots/Test-Accuracy-Vs-Model')
    plt.show()
    return

