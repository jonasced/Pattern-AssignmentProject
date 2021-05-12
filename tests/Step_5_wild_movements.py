import numpy as np
import matplotlib.pyplot as plt
from CharacterFeatureExtractor import *

# STEP 5- CASE FOR SHOWING FEATURES OF SAME SYMBOLS WITH WILD MOVEMENTS BETWEEN STROKES


#Get symbol-1 
dc1 = np.load("X_normal.npy")

#Get symbol-2 (The one that has wild movements between strokes)
dc2 = np.load("X_wild.npy")

dc1_xyb = dc1
dc2_xyb = dc2


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

#raw x,y,b values for symbol-1
x1 = dc1_xyb[0,:]
y1 = dc1_xyb[1,:]
b1 = dc1_xyb[2,:]
t1_coordinate = np.array(range(0,x1.size))

#raw x,y,b values for symbol-2
x2 = dc2_xyb[0,:]
y2 = dc2_xyb[1,:]
b2 = dc2_xyb[2,:]
t2_coordinate = np.array(range(0,x2.size))



f, axarr = plt.subplots(3, 2)
f.suptitle('Symbol Without Wild Movements', fontsize=20)


#------- FIRST FIGURE

#Drawing of sampled symbol-1
axarr[0, 0].scatter(sampled_symbol1[0], sampled_symbol1[1])
axarr[0, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr[0, 0].set_title('Symbol-1')
axarr[0, 0].set_xlim([0,210])
axarr[0, 0].set_ylim([0,210])

# X coordinate values of symbol-1
axarr[0, 1].plot(t1_coordinate, x1)
axarr[0, 1].set(xlabel = "Time", ylabel = "X-Coordinate")
axarr[0, 1].set_title('XYB for Symbol-1')

#Absolute distance plot of symbol-1
axarr[1, 0].plot(t1, f1_symbol1)
axarr[1, 0].set(xlabel = "Time", ylabel = "Normalized Distance")
axarr[1, 0].set_ylim([0,np.max(f1_symbol1)])

# Y coordinate values of symbol-1
axarr[1, 1].plot(t1_coordinate, y1)
axarr[1, 1].set(xlabel = "Time", ylabel = "Y-Coordinate")

#Y-wise distance plot of symbol-1
axarr[2, 0].plot(t1, f2_symbol1)
axarr[2, 0].set(xlabel = "Time", ylabel = "Slope(Degrees)")
axarr[2, 0].set_ylim([-120,120])

# B values of symbol-1
axarr[2, 1].plot(t1_coordinate, b1)
axarr[2, 1].set(xlabel = "Time", ylabel = "B Values")


#------- SECOND FIGURE

f2, axarr2 = plt.subplots(3, 2)
f2.suptitle('Symbol With Wild Movements', fontsize=20)

#Drawing of sampled symbol-2
axarr2[0, 0].scatter(sampled_symbol2[0], sampled_symbol2[1])
axarr2[0, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
axarr2[0, 0].set_title('Symbol-2')
axarr2[0, 0].set_xlim([0,210])
axarr2[0, 0].set_ylim([0,210])

# X coordinate values of symbol-2
axarr2[0, 1].plot(t2_coordinate, x2)
axarr2[0, 1].set(xlabel = "Time", ylabel = "X-Coordinate")
axarr2[0, 1].set_title('XYB for Symbol-2')

#Absolute distance plot of symbol-2
axarr2[1, 0].plot(t2, f1_symbol2)
axarr2[1, 0].set(xlabel = "Time", ylabel = "Normalized Distance")
axarr2[1, 0].set_ylim([0,np.max(f1_symbol2)])

# Y coordinate values of symbol-2
axarr2[1, 1].plot(t2_coordinate, y2)
axarr2[1, 1].set(xlabel = "Time", ylabel = "Y-Coordinate")

#Y-wise distance plot of symbol-2
axarr2[2, 0].plot(t2, f2_symbol2)
axarr2[2, 0].set(xlabel = "Time", ylabel = "Slope(Degrees)")
axarr2[2, 0].set_ylim([-120,120])

# B values of symbol-2
axarr2[2, 1].plot(t2_coordinate, b2)
axarr2[2, 1].set(xlabel = "Time", ylabel = "B Values")





plt.show()