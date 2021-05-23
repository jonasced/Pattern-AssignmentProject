import numpy as np


def classifier(HMM_Models, labels, sample, useprint=False):
    """
    Inputs:
        1) List of Class Models
        2) list of class labels
        3) 1 sample from a class containing features (d,t)
    Output:
        1)the label of the class that the sample is classified as

    """

    num_class = len(HMM_Models)
    class_probs = np.zeros(num_class)
    for char in range(num_class):
        a, c = HMM_Models[char].alphahat(sample)
        clog = np.log(c)
        lprob = np.sum(np.array(clog))
        # lprob = np.exp(lprob)
        class_probs[char] = lprob

    index_class = np.nanargmax(class_probs)

    if useprint:
        print("\n---START OF SAMPLE---")
        print("Classified as:", labels[index_class])
        for i in range(len(labels)):
            print("P(input) = ", labels[i], ": ", class_probs[i])
        print("\n")

    return labels[index_class]

