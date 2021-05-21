import numpy as np
import pandas as pd
import random


def dataprep(db_name, data_diff=5, shuffle=False):
    """ train_data, test_data = dataprep(db_name, data_diff)
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

    print("Database read is ", db_name)
    print("Labels used are ", data_labels)

    train_data = []
    test_data = []
    K = len(data_labels)
    for k in range(K):

        # Data preprocessing: invert data s.t. it fits the HMM_TA class.
        R = len(data_features[k])
        for r in range(R):
            data_features[k][r] = np.transpose(data_features[k][r])

        # Shuffles the data before separating into training/testing
        if shuffle:
            random.shuffle(data_features[k])

        # Divide data into training and testing
        train_obs = data_features[k][0:R-data_diff]
        test_obs = data_features[k][R-data_diff:R]
        train_data.append(train_obs)
        test_data.append(test_obs)

    return train_data, test_data, data_labels


def main():
    train_data, test_data, labels = dataprep("database_inc_sampchar")
    print(train_data[1][1].shape)
    print(len(train_data))
    print(len(train_data[1]))
    print(type(train_data[1][1]))
    print(type(train_data[1][1][1]))


if __name__ == "__main__":
    main()
