import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # f = open("../data/input.txt", "w")
    # for i in range(200):
    #     f.write(str(np.sin(i)) + '\n')

    input = np.loadtxt('../data/input.txt')
    output = np.loadtxt('../data/output.txt')

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(range(len(input)), input)
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('input')

    output = output[13280:13480]
    axs[1].plot(range(len(output)), output)
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('output')

    axs[0].grid(True)
    axs[1].grid(True)
    plt.show()