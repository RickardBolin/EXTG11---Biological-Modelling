import numpy as np
import matplotlib.pyplot as plt


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
    norm_combined_data_w_class, max_x, max_y = normalize(combined_data_w_class)

    # Train on all data to check approximate size of the weights
    trained_weights = train_perceptron(norm_combined_data_w_class, iterations=3000, lr=0.01, sgd=True)
    print("Weights when trained on all data: " + str(trained_weights))

    # Leave one out cross validation for perceptron:
    # Concatenate a duplicate of the data on top of itself to make cross validation easy
    double_norm_combined_data_w_class = np.vstack([norm_combined_data_w_class, norm_combined_data_w_class])

    # Keep track of how many correct predictions we've made.
    score = 0
    for i in range(30):
        # Pick the row that we'll use as evaluation
        eval_row = double_norm_combined_data_w_class[i, :]
        # Use the rest for training
        data = double_norm_combined_data_w_class[i + 1:i + 30, :]
        trained_weights = train_perceptron(data, iterations=5000, lr=0.01, sgd=True)

        # Predict what class the data in eval_row belongs to
        pred_class = 0
        if eval_row[0] * trained_weights[0] + eval_row[1] * trained_weights[1] + trained_weights[2] > 0:
            pred_class = 1

        # If we're right, add one to score
        if pred_class == eval_row[2]:
            score += 1

    print("Score: " + str(score))

    # Plot separation line
    # w1*x1 + w2*x2 + w3 = 0 ->
    # ->  x2 = -(w0/w1)x1 + w2/w1
    k = -(trained_weights[0] / trained_weights[1])
    m = - trained_weights[2] / trained_weights[1]

    plt.figure(1)
    plt.title("Plot of separation line calculated by regular perceptron")
    plt.xlabel("Total number of letters (Normalized)")
    plt.ylabel("Number of letter 'a' (Normalized)")
    plt.scatter(norm_combined_data_w_class[:15, 0], norm_combined_data_w_class[:15, 1], color="green")
    plt.scatter(norm_combined_data_w_class[16:, 0], norm_combined_data_w_class[16:, 1])

    reg = k * norm_combined_data_w_class[:, 0] + m
    plt.plot(norm_combined_data_w_class[:, 0], reg, color="red",
             label="y = " + str(round(k, 2)) + "x + " + str(round(m, 2)))
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


# Classification: English = 1, French = 0
def predict(data, w, logistic=False):
    predictions = np.array([])
    for r in range(data.shape[0]):
        prediction = 0
        # Check if we should predict english
        if logistic is True:
            if sigmoid(data[r, 0] * w[0] + data[r, 1] * w[1] + w[2]) > 0.5:
                prediction = 1
        else:
            # w contains three weights; w1, w2 and bias (w[0], w[1] and w[2] respectively)
            if data[r, 0] * w[0] + data[r, 1] * w[1] + w[2] > 0:
                prediction = 1
        # Append the prediction to the end of the array
        predictions = np.append(predictions, prediction)
    return predictions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_weights(weights, data, predictions, lr, sgd=False, logistic=False):
    if logistic is True:
        return update_weights_log(weights, data, lr, sgd)
    else:
        if sgd is True:
            # If sgd is true, pick one or a few random indicies for your minibatch
            idx = np.random.randint(data[:, 2].shape[0], size=(1, 1))
            # Update weights
            for i in range(2):
                weights[i] += lr * sum(data[idx, i] * (data[idx, 2] - predictions[idx])) / idx.shape[0]
            weights[2] += lr * sum(data[idx, 2] - predictions[idx]) / idx.shape[0]
        else:
            # If sgd is false, use all rows when calculating the weight update
            for i in range(2):
                weights[i] += lr * sum(data[:, i] * (data[:, 2] - predictions)) / data[:, 2].shape[0]
            weights[2] += lr * sum(data[:, 2] - predictions) / data[:, 2].shape[0]
        return weights


def update_weights_log(weights, data, lr, sgd=False):
    if sgd is True:
        # If sgd is true, pick one or a few random indicies for your minibatch
        idx = np.random.randint(data[:, 2].shape[0], size=(4, 1))
        # Calculate the scalar product x•d for the minibatch
        predictions = data[idx, 0] * weights[0] + data[idx, 1] * weights[1] + weights[2]
        # Update weights
        for i in range(2):
            weights[i] += lr * np.matmul(sigmoid(predictions).T, (1 - sigmoid(predictions))) * sum(
                data[idx, i] * (data[idx, 2] - sigmoid(predictions))) / idx.shape[0]
        weights[2] += lr * np.matmul(sigmoid(predictions).T, (1 - sigmoid(predictions))) * sum(
            data[idx, 2] - sigmoid(predictions)) / idx.shape[0]
    else:
        # If sgd is false, follow the same procedure as above, but for all rows
        predictions = data[:, 0] * weights[0] + data[:, 1] * weights[1] + weights[2]
        for i in range(2):
            weights[i] += lr * np.matmul(sigmoid(predictions).T, (1 - sigmoid(predictions))) * sum(
                data[:, i] * (data[:, 2] - sigmoid(predictions))) / data[:, 2].shape[0]
        weights[2] += lr * np.matmul(sigmoid(predictions).T, (1 - sigmoid(predictions))) * sum(
            data[:, 2] - sigmoid(predictions)) / data[:, 2].shape[0]
    return weights


def train_perceptron(data, iterations, lr, sgd=False, logistic=False):
    # Initialize weights
    weights = np.array([0.1, 0.1, 0.1])
    for i in range(iterations):
        predictions = predict(data, weights, logistic)
        # If the number of misclassifications are small, break
        if sum(abs(data[:, 2] - predictions)) == 0:
            break
        # Update weights
        weights = update_weights(weights, data, predictions, lr, sgd, logistic)
    return weights


if __name__ == '__main__':
    main()
