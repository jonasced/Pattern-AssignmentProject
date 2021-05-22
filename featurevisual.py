import numpy as np
from matplotlib import pyplot as plt


def featurevisual(data_features, data_labels, data_sampchar, chars=[]):
    if len(chars) == 0:
        chars = range(len(data_labels))
    for char in chars:

        # char = 5 # which character in the label list we are using
        obs = data_features[char]
        # raw_obs = data[char]
        sampchar = data_sampchar[char]

        plt_num = len(data_features)  # number of samples plotted
        f, axarr = plt.subplots(3, plt_num, dpi = 200)
        f.suptitle('Scale & Position Effect of letter '+data_labels[char], fontsize=10)

        for i in range(plt_num):

            feature_symbol1 = obs[i]
            sampled_symbol1 = sampchar[i]

            # normalized distance ,slope, and t for symbol-1
            f1_symbol1 = feature_symbol1[0]
            f2_symbol1 = feature_symbol1[1]
            t1 = np.array(range(0,feature_symbol1.shape[1]))

            # mean and number of states required:
            # killgissa?

            # ------------- SYMBOL DRAWINGS
            # Drawing of sampled symbol-1
            axarr[0, i].scatter(sampled_symbol1[0], sampled_symbol1[1])
            # axarr[0, i].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
            # axarr[0, i].set_title('Symbol-1')
            axarr[0, i].set_xlim([0,210])
            axarr[0, i].set_ylim([0,210])

            # ------------- ABSOLUTE DISTANCE FEATURE
            # Absolute distance plot of symbol-1
            axarr[1, i].plot(t1, f1_symbol1)
            # axarr[1, i].set(xlabel = "Time", ylabel = "Normalized Distance")
            axarr[1, i].set_ylim([0,np.max(f1_symbol1)])

            # ------------- SLOPE FEATURE
            # Y-wise distance plot of symbol-1
            axarr[2, i].plot(t1, f2_symbol1)
            # axarr[2, i].set(xlabel = "Time", ylabel = "Slope(Degrees)")
            axarr[2, i].set_ylim([-120,120])

        plt.show()

        """ Observed results:
        X: 2 states, 
        f1s1mean=10
        f1s2mean=45
        f2s1mean=-60
        f2s2mean=70
        """