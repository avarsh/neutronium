from network import Network
import numpy as np
from math import sin, pi

if __name__ == '__main__':
    network = Network([1, 20, 1])
    np.random.seed(0)
    arr = np.random.rand(1000, 1)
    x_vals = [x * 2 * pi for x in arr]
    sine_data = [(np.array([x]), np.array([[0.5 * (sin(x) + 1)]])) for x in x_vals]

    print("Example input:\n", sine_data[0:10])

    eval_data = [ (np.array([[pi]]), np.array([[0]])),
                  (np.array([[1]]),  np.array([[sin(1) / 2 + 0.5]])), 
                  (np.array([[0]]), np.array([[0]])),
                  (np.array([[pi/2]]), np.array([[sin(pi/2) / 2 + 0.5]])),
                  (np.array([[pi/6]]), np.array([[sin(pi) / 6 + 0.5]])),
                  (np.array([[2 * pi / 3]]), np.array([[sin(2 * pi / 3) / 2 + 0.5]])),
                  (np.array([[pi / 4]]), np.array([[sin(pi / 4) / 2 + 0.5]])) ]

    network.stochastic_grad_desc(sine_data, 4000, 0.04, 20)

    print("sin pi = ", network.get_output(np.array([[pi]]))[0] * 2 - 1, " - expected: ", 0)
    print("sin 1 = ", network.get_output(np.array([[1]]))[0] * 2 - 1, " - expected:", sin(1))
    print("sin 0 = ", network.get_output(np.array([[0]]))[0] * 2 - 1, " - expected:", 0)
    print("sin pi/2 = ", network.get_output(np.array([[pi/2]]))[0] * 2 - 1, " - expected:", sin(pi / 2))
    print("sin pi/6 = ", network.get_output(np.array([[pi/6]]))[0] * 2 - 1, " - expected:", sin(pi / 6))
    print("sin 2pi/3 = ", network.get_output(np.array([[2 * pi/3]]))[0] * 2 - 1, " - expected:", sin(2 * pi / 3))
    print("sin pi/4 = ", network.get_output(np.array([[pi/4]]))[0] * 2 - 1, " - expected:", sin(pi / 4))