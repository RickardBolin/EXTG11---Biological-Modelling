import numpy as np
import matplotlib.pyplot as plt
import Perceptron as p


def main():
    # Load data
    english_data_path = "./Data/salammbo_a_en.tsv"
    french_data_path = './Data/salammbo_a_fr.tsv'

    english_data = np.genfromtxt(fname=english_data_path, delimiter="\t")
    french_data = np.genfromtxt(fname=french_data_path, delimiter="\t")

    # Create a new data array with all data that in addition contains the class of the data in each row
    english_data_w_class = np.hstack([english_data, np.ones(shape=(english_data.shape[0], 1))])
    french_data_w_class = np.hstack([french_data, np.zeros(shape=(french_data.shape[0], 1))])
    combined_data_w_class = np.vstack([english_data_w_class, french_data_w_class])

    # Normalize the combined data
    norm_combined_data_w_class, max_x, max_y = p.normalize(combined_data_w_class)

    # Train on all data to check approximate size of the weights
    trained_weights = p.train_perceptron(norm_combined_data_w_class, iterations=15000, lr=0.1, sgd=True)
    print("Weights when trained on all data: " + str(trained_weights))

    # Leave one out cross validation for logistic regression

    # Concatenate a duplicate of the data on top of itself to make cross validation easy
    double_norm_combined_data_w_class = np.vstack([norm_combined_data_w_class, norm_combined_data_w_class])

    # Keep track of how many correct predictions we've made.
    score = 0

    for i in range(30):
        # Pick the row that we'll use as evaluation
        eval_row = double_norm_combined_data_w_class[i, :]
        # Use the rest for training
        data = double_norm_combined_data_w_class[i + 1:i + 30, :]
        trained_weights = p.train_perceptron(data, iterations=15000, lr=3, sgd=True, logistic=True)
        # Predict what class the data in eval_row belongs to
        pred_class = 0
        if p.sigmoid(eval_row[0] * trained_weights[0] + eval_row[1] * trained_weights[1] + trained_weights[2]) > 0.5:
            pred_class = 1

        # If we're right, add one to score
        if pred_class == eval_row[2]:
            score += 1

    print("Score:" + str(score))

    # Plot separation line
    # 1/1+e^-(X • w) = 0.5 ->
    # ->  y = -(w0/w1)x - (w2/w1)
    k = -(trained_weights[0] / trained_weights[1])
    m = - trained_weights[2] / trained_weights[1]

    plt.figure(1)
    plt.title("Plot of separation line calculated by logistic regression")
    plt.xlabel("Total number of letters (Normalized)")
    plt.ylabel("Number of letter 'a' (Normalized)")
    plt.scatter(norm_combined_data_w_class[:15, 0], norm_combined_data_w_class[:15, 1], color="green")
    plt.scatter(norm_combined_data_w_class[16:, 0], norm_combined_data_w_class[16:, 1])

    reg = k * norm_combined_data_w_class[:, 0] + m
    plt.plot(norm_combined_data_w_class[:, 0], reg, color="red",
             label="y = " + str(round(k, 2)) + "x + " + str(round(m, 2)))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
