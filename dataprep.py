import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from CharacterFeatureExtractor import featureExtractor
from DrawCharacter import DrawCharacter
from PattRecClasses import HMM_TA
from hmm_gen import hmm_gen


def dataprep(db_name, data_diff):
    """ train_data, test_data = dataprep(db_name, data_diff)
    Input:
        db_name:
        data_diff: the difference in testing/training samples
    Output:
        train_data:
        test_data:

    """
    db_name = "database_inc_sampchar"
    data_features = pd.read_pickle(r'data/' + db_name + '_features.cdb')
    data_labels = pd.read_pickle(r'data/' + db_name + '_labels.cdb')

    # data_features[k][r] == np.array (ndim, t); K (number of letters) of R samples with Tr individual lengths
    # print((data_features[1][1].shape))
    print(data_labels)
    nr_char = len(data_labels)

    train_data = []
    test_data = []
    for char in range(nr_char):
        print("\n")
        print("------------ CHARACTER", data_labels[char], "------------")
        # Data preprocessing
        obs = data_features[char]
        # obsTA = np.array([ hm_learn.rand(100)[0] for _ in range(10) ])
        # print(type(obsTA))
        # print(obsTA[1].shape) == (100,2)
        # Our data has format (2,15) ! Transpose all datapoints
        for i in range(len(obs)):
            obs[i] = np.transpose(obs[i])

        # data_features[char] = obs  # so we do not have to reinvert the data later

        # Divide data into training and testing

        train_obs = obs[0:len(obs) - data_diff]
        test_obs = obs[len(obs) - data_diff:len(obs)]
        train_data += [train_obs]
        test_data += [test_obs]
    return train_data, test_data, data_labels


def main():
    train_data, test_data, labels = dataprep("database_inc_sampchar", 5)
    print(train_data[1][1].shape)


if __name__ == "__main__":
    main()
