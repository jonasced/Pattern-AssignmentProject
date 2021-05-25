import os
import pickle
from CharacterFeatureExtractor import *


def createDataSet(act=True, name=None, test=False):
    num_char = 10
    num_sample = 20
    char_labels = ["A", "C", "K", "P", "X", "T", "+", "N", "V", "4"]
    current_path = os.path.dirname(os.path.abspath(__file__))
    list_of_char_lists = []
    list_of_feature_lists = []
    list_of_sampchar_lists = []
    thr = 9 #Threshold for sampling of characters before feature extraction

    print(current_path)
    for c in range(num_char):
        temp_char_list = []
        temp_feature_list = []
        temp_sampchar_list = []
        for s in range(num_sample):
            # charpath = os.path.join(current_path, "/data/" + char_labels[c] + str(s + 1) + ".npy")
            charpath = current_path + "/data/" + char_labels[c] + str(s + 1) + ".npy"
            char = np.load(charpath)
            feature, sampled_char = featureExtractor(char, thr, False, test=test)
            temp_char_list.append(char)
            temp_feature_list.append(feature)
            temp_sampchar_list.append(sampled_char)

        list_of_char_lists.append(temp_char_list)
        list_of_feature_lists.append(temp_feature_list)
        list_of_sampchar_lists.append(temp_sampchar_list)

    # If running in interactive mode:
    if act:
        decision = input("Do you want to save the database? ('y' for yes, 'n' for no): ")
    # Else always save
    else:
        decision = "y"

    if decision == "y":
        if name is None:  # option to provide dbname in input
            name = input("What do you want to save the database file as? : ")

        # Always saves the cdb to the data folder
        if "data" not in current_path:
            current_path += "/data/"

        file_path_char = os.path.join(current_path, name + ".cdb")
        file_path_feauture = os.path.join(current_path, name + "_features.cdb")
        file_path_labels = os.path.join(current_path, name + "_labels.cdb")
        file_path_sampchar = os.path.join(current_path, name + "_sampchar.cdb")

        with open(file_path_char, "wb") as fp: # Pickling
            pickle.dump(list_of_char_lists, fp)
        with open(file_path_feauture, "wb") as fp: # Pickling
            pickle.dump(list_of_feature_lists, fp)
        with open(file_path_labels, "wb") as fp: # Pickling
            pickle.dump(char_labels, fp)
        with open(file_path_sampchar, "wb") as fp: # Pickling
            pickle.dump(list_of_sampchar_lists, fp)

        print("Dataset saved")


def main():
    createDataSet(act=False, name="100xtest", test=True)


if __name__ == "__main__":
    main()


