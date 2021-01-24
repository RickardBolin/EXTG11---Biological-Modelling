import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load data
    english_data_path = "./Data/salammbo_a_en.tsv"
    french_data_path = './Data/salammbo_a_fr.tsv'

    english_data = np.genfromtxt(fname=english_data_path, delimiter="\t")
    french_data = np.genfromtxt(fname=french_data_path, delimiter="\t")

    # Define a learning rate/step size for our gradient descent method:
    lr = 0.1
    # Specify for how many iterations we should run our algorithm
    iterations = 5000

    # Normalize data and save normalizing constants for later
    norm_english_data, max_x_en, max_y_en = normalize(english_data)
    norm_french_data, max_x_fr, max_y_fr = normalize(french_data)

    english_k, english_m = calc_lin_reg(norm_english_data, iterations, lr, sgd=True)
    french_k, french_m = calc_lin_reg(norm_french_data, iterations, lr, sgd=True)

    # Plotting the english data together with regression line
    plt.title("Plot of the English data together with its regression line.")
    plt.xlabel("Total number of letters")
    plt.ylabel("Number of letter 'a' ")
    plt.scatter(norm_english_data[:, 0] * max_x_en, norm_english_data[:, 1] * max_y_en)
    eng_reg_y = english_k * norm_english_data[:, 0] + english_m
    plt.plot(norm_english_data[:, 0] * max_x_en, eng_reg_y * max_y_en, color="red",
             label="y = " + str(round(english_k[0], 2)) + "x + " + str(round(english_m[0], 2)))
    plt.legend()
    print("Parameters for regression line on english data:   k: " + str(english_k[0])
          + "  m: " + str(english_m[0]))

    # Plotting the french data together with regression line
    plt.figure()
    plt.title("Plot of the French data together with its regression line.")
    plt.xlabel("Total number of letters")
    plt.ylabel("Number of letter 'a' ")

    plt.scatter(norm_french_data[:, 0] * max_x_fr, norm_french_data[:, 1] * max_y_fr)
    fr_reg_y = french_k * norm_french_data[:, 0] + french_m
    plt.plot(norm_french_data[:, 0] * max_x_fr, fr_reg_y * max_y_fr, color="red",
             label="y = " + str(round(french_k[0], 2)) + "x + " + str(round(french_m[0], 2)))
    plt.legend()
    print("Parameters for regression line on french data:   k: " + str(french_k[0]) + "  m: "
          + str(french_m[0]))

    # Plotting everything in the same figure
    plt.figure()
    plt.title("Plot of both English and French data together with their respective regression lines.")
    plt.xlabel("Total number of letters")
    plt.ylabel("Number of letter 'a' ")

    plt.scatter(norm_english_data[:, 0] * max_x_en, norm_english_data[:, 1] * max_y_en)
    eng_reg_y = english_k * norm_english_data[:, 0] + english_m
    plt.plot(norm_english_data[:, 0] * max_x_en, eng_reg_y * max_y_en, label="English regression line")
    plt.scatter(norm_french_data[:, 0] * max_x_fr, norm_french_data[:, 1] * max_y_fr)
    fr_reg_y = french_k * norm_french_data[:, 0] + french_m
    plt.plot(norm_french_data[:, 0] * max_x_fr, fr_reg_y * max_y_fr, label="French regression line")

    plt.legend()
    plt.show()


# Function to normalize data
def normalize(data):
    data = data.copy()
    max_x = np.max(data[:, 0])
    max_y = np.max(data[:, 1])
    data[:, 0] = data[:, 0] / max_x
    data[:, 1] = data[:, 1] / max_y
    return data, max_x, max_y


# Calculate the mean squared error (MSE):
def calc_MSE(data, y_pred):
    return sum((data[:, 1] - y_pred) ** 2) / data.shape[0]


# Derivative of MSE with respect to k:
def calc_deriv_k(data, k, m, sgd=False):
    if sgd is True:
        # If sgd is True, use minibatch SGD
        # Pick a few random indices
        idx = np.random.randint(data.shape[0], size=(3, 1))
        return -2 * sum(data[idx, 0] * (data[idx, 1] - (k * data[idx, 0] + m))) / idx.shape[0]
    else:
        return -2 * sum(data[:, 0] * (data[:, 1] - (k * data[:, 0] + m))) / data.shape[0]


# Derivative of MSE with respect to m:
def calc_deriv_m(data, k, m, sgd=False):
    if sgd is True:
        # If sgd is True, use minibatch SGD
        # Pick a few random indices
        idx = np.random.randint(data.shape[0], size=(3, 1))
        return -2 * sum(data[idx, 1] - (k * data[idx, 0] + m)) / idx.shape[0]
    else:
        # If sgd is False, use batch gradient descent
        return -2 * sum(data[:, 1] - (k * data[:, 0] + m)) / data.shape[0]


def perform_gradient_descent(data, k, m, lr, sgd=False):
    # Calculate the derivatives used in the gradient descent
    dk = calc_deriv_k(data, k, m, sgd)
    dm = calc_deriv_m(data, k, m, sgd)
    return k - lr * dk, m - lr * dm


def calc_lin_reg(data, iterations, lr, sgd=False):
    # y = kx+m
    # Start with k and m both equal to zero
    k = 0
    m = 0
    for i in range(iterations):
        # Check if the line is good enough
        if calc_MSE(data, k * data[:, 0] + m) < 1e-5:
            break
        # Perform the gradient descent
        k, m = perform_gradient_descent(data, k, m, lr, sgd)
    return k, m


if __name__ == '__main__':
    main()
