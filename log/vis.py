import csv
import matplotlib.pyplot as plt


def func1():
    X = []
    Y1 = []
    Y2 = []
    with open('mmd_notrain.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X.append(int(row[0]))
            Y1.append(float(row[1]))
            Y2.append(float(row[2])*150)

    plt.plot(X[::10], Y1[::10], label='mmd')
    plt.plot(X[::10], Y2[::10], label='kl*150')
    plt.legend()
    plt.title("Not Train, Only Interact")
    plt.savefig("Not-Train-Only-Interact.png", dpi=300)
    plt.show()


def func2():
    X = []
    Y1 = []
    Y2 = []
    with open('mmd_half.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X.append(int(row[0]))
            Y1.append(float(row[1]))
            Y2.append(float(row[2]))

    plt.plot(X[::2], Y1[::2], label='mmd')
    plt.plot(X[::2], Y2[::2], label='kl')
    plt.title("Stop Interact After 50K Steps")
    plt.legend()
    plt.savefig("Stop-Interact-After-50K-Steps.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    func1()
