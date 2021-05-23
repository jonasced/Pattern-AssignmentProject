import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from CharacterFeatureExtractor import featureExtractor
from dataprep import *
from modeltrain import *
from classifier import *
from hmm_test import *
from createDataSet import *
from DrawCharacter import DrawCharacter
from PattRecClasses import HMM_TA
from hmm_gen import hmm_gen

db_name = "featTest2"
data_features = pd.read_pickle(r'data/' + db_name + '_features.cdb')
data_labels = pd.read_pickle(r'data/' + db_name + '_labels.cdb')
class_state_nums = np.array([2, 5, 5, 5, 5, 5, 5, 6, 5, 5])

train_data, test_data, data_labels = dataprep(db_name, nr_test=5)
hm_learn, train_acc = modeltrain(train_data, data_labels, 12, class_state_nums, longest_sample=True, useprint=False)
accuracies, result_labels_list = hmm_test(hm_learn, test_data, data_labels)

