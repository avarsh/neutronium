from network import Network
import numpy as np

if __name__ == '__main__':
    network = Network([2, 2, 1])
    xor_data = [
        (np.array([[1], [1]]), np.array([[0]])),
        (np.array([[1], [0]]), np.array([[1]])),
        (np.array([[0], [1]]), np.array([[1]])),
        (np.array([[0], [0]]), np.array([[0]]))
    ]
    network.online(xor_data, 10000, 0.5, True)

    print("1 xor 1 =", network.get_output(np.array([[1], [1]])))
    print("0 xor 1 =", network.get_output(np.array([[0], [1]])))
    print("1 xor 0 =", network.get_output(np.array([[1], [0]])))
    print("0 xor 0 =", network.get_output(np.array([[0], [0]])))