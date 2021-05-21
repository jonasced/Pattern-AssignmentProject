import numpy as np
from PattRecClasses import HMM_TA

"""
Inputs: 
    1) List of Class Models
    2) list of class labels
    3) 1 sample from a class containing features (d,t)
Output: 
    1)the label of the class that the sample is classified as

"""
def classifier(HMM_Models, labels, sample):
    num_class = len(HMM_Models)
    class_probs = np.zeros(num_class)
    for char in range(num_class):
         a, c = HMM_Models[char].alphahat(sample)
         clog = np.log(c)
         lprob = np.sum(np.array(clog))
         lprob = np.exp(lprob)
         class_probs[char] = lprob
    
    index_class = np.argmax(class_probs)
    
    return labels[index_class]
        
   