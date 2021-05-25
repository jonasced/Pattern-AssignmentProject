import numpy as np
import pandas as pd


def dataprep(db_name, max_labels=0, max_samples=0, nr_test=5, shuffle=False, useprint=True):
    """ train_data, test_data, data_labels = dataprep(db_name, data_diff)
    Input:
        db_name:
        data_diff: the difference in testing/training samples
    Output:
        train_data: data[k][r] == np.array (ndim, t); K (number of letters) of R samples with Tr individual lengths
        test_data:
    """
    # Read database
    data_features = pd.read_pickle(r'data/' + db_name + '_features.cdb')
    data_labels = pd.read_pickle(r'data/' + db_name + '_labels.cdb')

    if useprint:
        print("\nDatabase read is ", db_name)

    K = len(data_labels)  # number of characters in data set
    R = len(data_features[0])  # number of samples of each character in data set; here uniform values

    # If data-reducing options are used:
    if max_labels != 0:
        data_labels = data_labels[0:max_labels]
        K = max_labels
        if useprint:
            print("OPTION USED: Maximum number of labels used is ", max_labels)
    if max_samples != 0:
        R = max_samples
        if useprint:
            print("OPTION USED: Maximum number of samples used is ", R)

    if useprint:
        print("Labels used are ", data_labels)
        print("Total training samples are ", R - nr_test, " and testing samples are ", nr_test, "\n")

    train_data = []
    test_data = []

    for k in range(K):

        # Shuffles the data before separating into training/testing
        if shuffle:
            np.random.shuffle(data_features[k])

        # Data preprocessing: invert data s.t. it fits the HMM_TA class.
        for r in range(R):

            data_features[k][r] = np.transpose(data_features[k][r])

        # Divide data into training and testing
        train_obs = data_features[k][0:R - nr_test]
        test_obs = data_features[k][R - nr_test:R]
        train_data.append(train_obs)
        test_data.append(test_obs)

    return train_data, test_data, data_labels


def main():
    train_data, test_data, labels = dataprep("database_inc_sampchar", shuffle=True, max_labels=2, max_samples=5, nr_test=2)
    print(train_data[1][1].shape)
    print(len(train_data))
    print(len(train_data[1]))
    print(type(train_data[1][1]))
    print(type(train_data[1][1][1]))


if __name__ == "__main__":
    main()
