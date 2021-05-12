import numpy as np
import matplotlib.pyplot as plt

from DrawCharacter import *
from CharacterFeatureExtractor import *


#Get symbol-1
dc1 = DrawCharacter()
dc1.run()
#Get symbol-2
dc2 = DrawCharacter()
dc2.run()

# #Feature vectors are returned
thr = 9 # threshold for sampling and distance normalization
feature_symbol1, sampled_symbol1 = featureExtractor(dc1,thr)
feature_symbol2, sampled_symbol2 = featureExtractor(dc2,thr)


### PLOTS

#Prepare data for feature plots

# normalized distance ,slope, and t for symbol-1
f1_symbol1 = feature_symbol1[0]
f2_symbol1 = feature_symbol1[1]
t1 = np.array(range(0,feature_symbol1.shape[1]))

# normalized distance ,slope, and t for symbol-2
f1_symbol2 = feature_symbol2[0]
f2_symbol2 = feature_symbol2[1]
t2 = np.array(range(0,feature_symbol2.shape[1]))

#Prepare data for original symbols
get1 = dc1.get_xybpoints()
data1_clean = removeZero(get1) 
get2 = dc2.get_xybpoints()
data2_clean = removeZero(get2)

#Plot setup
f, axarr = plt.subplots(4, 2)
f.suptitle('Character Check', fontsize=20)

#Drawing of original symbol-1
axarr[0, 0].scatter(data1_clean[0], data1_clean[1])
axarr[0, 0].set_title('Symbol-1')
axarr[0, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[0, 0].set_xlim([0,210])
axarr[0, 0].set_ylim([0,210])

#Drawing of original symbol-2
axarr[0, 1].scatter(data2_clean[0], data2_clean[1])
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

#Absolute distance plot of symbol-1
axarr[2, 0].plot(t1, f1_symbol1)
axarr[2, 0].set(xlabel = "Time", ylabel = "Normalized Distance")
axarr[2, 0].set_ylim([0,np.max(f1_symbol1)])

#Absolute distance plot of symbol-2
axarr[2, 1].plot(t2, f1_symbol2)
axarr[2, 1].set(xlabel = "Time", ylabel = "Normalized Distance")
axarr[2, 1].set_ylim([0,np.max(f1_symbol2)])

#Slope(Degree) plot of symbol-1
axarr[3, 0].plot(t1, f2_symbol1)
axarr[3, 0].set(xlabel = "Time", ylabel = "Slope(Degrees)")
axarr[3, 0].set_ylim([-120,120])

#Slope(Degree) plot of symbol-2
axarr[3, 1].plot(t2, f2_symbol2)
axarr[3, 1].set(xlabel = "Time", ylabel = "Slope(Degrees)")
axarr[3, 1].set_ylim([-120,120])

plt.show()












