import numpy as np
import matplotlib.pyplot as plt
from CharacterFeatureExtractor import *

# STEP 4- CASE FOR SHOWING FEATURES OF DIFFERENT SYMBOLS


#Get symbol-1 
dc1 = np.load("A_normal.npy")

#Get symbol-2 (different than first one)
dc2 = np.load("B_normal.npy")


thr = 8 # threshold for sampling and distance normalization

# #Feature vectors are returned
feature_symbol1, sampled_symbol1 = featureExtractor(dc1,thr,False)
feature_symbol2, sampled_symbol2 = featureExtractor(dc2,thr,False)



# normalized distance ,slope, and t for symbol-1
f1_symbol1 = feature_symbol1[0]
f2_symbol1 = feature_symbol1[1]
t1 = np.array(range(0,feature_symbol1.shape[1]))


# normalized distance ,slope, and t for symbol-2
f1_symbol2 = feature_symbol2[0]
f2_symbol2 = feature_symbol2[1]
t2 = np.array(range(0,feature_symbol2.shape[1]))



f, axarr = plt.subplots(3, 2)
f.suptitle('Features of Different Symbols', fontsize=20)




#------------- SYMBOL DRAWINGS
#Drawing of sampled symbol-1
axarr[0, 0].scatter(sampled_symbol1[0], sampled_symbol1[1])
axarr[0, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[0, 0].set_title('Symbol-1')
axarr[0, 0].set_xlim([0,210])
axarr[0, 0].set_ylim([0,210])

#Drawing of sampled symbol-2
axarr[0, 1].scatter(sampled_symbol2[0], sampled_symbol2[1])
axarr[0, 1].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[0, 1].set_title('Symbol-2')
axarr[0, 1].set_xlim([0,210])
axarr[0, 1].set_ylim([0,210])


#------------- ABSOLUTE DISTANCE FEATURE

#Absolute distance plot of symbol-1
axarr[1, 0].plot(t1, f1_symbol1)
axarr[1, 0].set(xlabel = "Time", ylabel = "Normalized Distance")
axarr[1, 0].set_ylim([0,np.max(f1_symbol1)])

#Absolute distance plot of symbol-2
axarr[1, 1].plot(t2, f1_symbol2)
axarr[1, 1].set(xlabel = "Time", ylabel = "Normalized Distance")
axarr[1, 1].set_ylim([0,np.max(f1_symbol2)])


#------------- SLOPE FEATURE

#Y-wise distance plot of symbol-1
axarr[2, 0].plot(t1, f2_symbol1)
axarr[2, 0].set(xlabel = "Time", ylabel = "Slope(Degrees)")
axarr[2, 0].set_ylim([-120,120])

#Y-wise distance plot of symbol-2
axarr[2, 1].plot(t2, f2_symbol2)
axarr[2, 1].set(xlabel = "Time", ylabel = "Slope(Degrees)")
axarr[2, 1].set_ylim([-120,120])



plt.show()