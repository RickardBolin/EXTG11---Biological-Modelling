import numpy as np

# LIBSVM format:

english_data_path = "./Data/salammbo_a_en.tsv"
french_data_path = './Data/salammbo_a_fr.tsv'


def main():
    path = "./Data/dataInLIBSVMformat.txt"
    english_data = np.genfromtxt(fname=english_data_path, delimiter="\t")
    french_data = np.genfromtxt(fname=french_data_path, delimiter="\t")

    # Create a new data array with all data that in addition contains the class of the data in each row
    english_data_w_class = np.hstack([english_data, np.ones(shape=(english_data.shape[0], 1))])
    french_data_w_class = np.hstack([french_data, np.zeros(shape=(french_data.shape[0], 1))])

    save_LIBSVM(path, english_data_w_class)
    save_LIBSVM(path, french_data_w_class, 'a')

    combined_data_w_class = load_LIBSVM(path)
    print(combined_data_w_class)


# Save data to file with LIBSVM format where data holds the class at the
# last index of every row
def save_LIBSVM(path, data, mode='w'):
    with open(path, mode) as file:
        for row in data:
            string = ""
            string += str(int(row[2]))
            for index in range(len(row) - 1):
                string += " " + str(index) + ":" + str(row[index])
            string += "\n"
            file.write(string)


# Load data from file with LIBSVM format
def load_LIBSVM(path):
    with open(path) as file:
        data = []
        idx = 0
        for line in file:
            l = line.split(" ")
            r = np.zeros(len(l))
            # Get class
            r[2] = l.pop(0)

            # Loop over remaining item in list
            for item in l:
                indexValuePair = item.split(":")
                r[int(indexValuePair[0])] = float(indexValuePair[1])

            data.append(r)

    return np.asarray(data)


if __name__ == '__main__':
    main()
