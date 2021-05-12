import numpy as np
import matplotlib.pyplot as plt
from CharacterFeatureExtractor import *

#CASE FOR SHOWING THE EFFECT OF SAMPLING
#Get symbol-1
dc1 = np.load("A_low_left.npy")

#Get symbol-2
dc2 = np.load("A_low_right.npy")

#Prepare data for original symbols
data1_clean = removeZero(dc1) 
data2_clean = removeZero(dc2)

thr = 9 # threshold for sampling and distance normalization

# combined pixel values of all strokes for original symbols
comb_stroke_list1 = data1_clean 
comb_stroke_list2 = data2_clean

#sampled versions of symbols
sampled_symbol1 = sampler(data1_clean, thr)
sampled_symbol2 = sampler(data2_clean, thr)


f, axarr = plt.subplots(2, 2)
f.suptitle('Sampling Effect', fontsize=20)

#Drawing of original symbol-1
axarr[0, 0].scatter(comb_stroke_list1[0], comb_stroke_list1[1])
axarr[0, 0].set_title('Symbol-1')
axarr[0, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[0, 0].set_xlim([0,210])
axarr[0, 0].set_ylim([0,210])

#Drawing of original symbol-2
axarr[0, 1].scatter(comb_stroke_list2[0], comb_stroke_list2[1])
axarr[0, 1].set_title('Symbol-2')
axarr[0, 1].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[0, 1].set_xlim([0,210])
axarr[0, 1].set_ylim([0,210])



#Drawing of sampled symbol-1
axarr[1, 0].scatter(sampled_symbol1[0], sampled_symbol1[1])
axarr[1, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[1, 0].set_xlim([0,210])
axarr[1, 0].set_ylim([0,210])

#Drawing of sampled symbol-2
axarr[1, 1].scatter(sampled_symbol2[0], sampled_symbol2[1])
axarr[1, 1].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[1, 1].set_xlim([0,210])
axarr[1, 1].set_ylim([0,210])


plt.show()