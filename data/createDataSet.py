import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from DrawCharacter import *
from CharacterFeatureExtractor import *

num_char = 5
num_sample = 5
char_labels = ["A", "C", "K", "P", "X"]
current_path = os.path.dirname(os.path.abspath(__file__)) 
list_of_char_lists = []
list_of_feature_lists = []
list_of_sampchar_lists = []
thr = 9 #Threshold for sampling of characters before feature extraction

for c in range(num_char):
    temp_char_list = []
    temp_feature_list = []
    temp_sampchar_list = []
    for s in range(num_sample):
        char = np.load(os.path.join(current_path, char_labels[c] + str(s + 1) + ".npy") )
        feature,sampled_char = featureExtractor(char,thr, False)
        temp_char_list.append(char)
        temp_feature_list.append(feature)
        temp_sampchar_list.append(sampled_char)

    list_of_char_lists.append(temp_char_list)
    list_of_feature_lists.append(temp_feature_list)
    list_of_sampchar_lists.append(temp_sampchar_list)

decision = input("Do you want to save the database? ('y' for yes, 'n' for no): ")

if decision == "y":
    name = input("What do you want to save the database file as? : ")

    file_path_char = os.path.join(current_path, name + ".cdb")
    file_path_feauture = os.path.join(current_path, name + "_features.cdb")
    file_path_labels = os.path.join(current_path, name + "_labels.cdb")
    file_path_sampchar = os.path.join(current_path, name + "_sampchar.cdb")

    with open(file_path_char, "wb") as fp: #Pickling
        pickle.dump(list_of_char_lists, fp)
    with open(file_path_feauture, "wb") as fp: #Pickling
        pickle.dump(list_of_feature_lists, fp)   
    with open(file_path_labels, "wb") as fp: #Pickling
        pickle.dump(char_labels, fp)
    with open(file_path_sampchar, "wb") as fp: #Pickling
        pickle.dump(list_of_sampchar_lists, fp)



