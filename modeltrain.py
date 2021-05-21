import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from CharacterFeatureExtractor import featureExtractor
from DrawCharacter import DrawCharacter
from PattRecClasses import HMM_TA
from hmm_gen import hmm_gen
from dataprep import dataprep


def modeltrain(train_data, labels, iter):
    """
    hmm_learn, train_acc = modeltrain(train_data, labels):

    """

    hmm_learn = hmm_gen(train_data, 10)
    nr_char = len(train_data)
    for char in range(nr_char):
        hmm_learn[char].baum_welch(train_data, iter, prin=1, uselog=False)


def main():
    train_data, test_data, labels = dataprep("database_inc_sampchar", 5)
    print(train_data[1][1].shape)

    modeltrain(train_data, labels, 20)


if __name__ == "__main__":
    main()
