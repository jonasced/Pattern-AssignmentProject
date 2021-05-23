import numpy as np
from matplotlib import pyplot as plt


def featurevisual(data_features, data_labels, data_sampchar, chars=[], plt_num=[], model=False):
    if len(chars) == 0:
        chars = range(len(data_labels))
    for char in chars:

        # char = 5 # which character in the label list we are using
        obs = data_features[char]
        # raw_obs = data[char]
        sampchar = data_sampchar[char]

        if len(plt_num) == 0:
            plt_num = range(len(data_features[char]))
            f, axarr = plt.subplots(3, len(data_features[char]), dpi=200)
        elif len(plt_num) == 1:
            print("One number inputed interpreted as number of samples to be shown")
            f, axarr = plt.subplots(3, plt_num[0], dpi=200)
            plt_num = range(plt_num[0])
        else:
            f, axarr = plt.subplots(3, len(plt_num), dpi=200)


        if model:
            f.suptitle('Random sequences from HMM model of ' + data_labels[char], fontsize=10)
        else:
            f.suptitle('Scale & Position Effect of letter '+data_labels[char], fontsize=10)

        axarr[1, 0].set(xlabel="Time", ylabel="Normalized Distance")
        axarr[2, 0].set(xlabel="Time", ylabel="Slope(Degrees)")

        i = 0
        for samp in plt_num:

            feature_symbol1 = obs[samp]
            sampled_symbol1 = sampchar[samp]

            # normalized distance ,slope, and t for symbol-1
            f1_symbol1 = feature_symbol1[0]
            f2_symbol1 = feature_symbol1[1]
            t1 = np.array(range(0, feature_symbol1.shape[1]))

            # mean and number of states required:
            # killgissa?

            # ------------- SYMBOL DRAWINGS
            # Drawing of sampled symbol-1
            if not model:
                axarr[0, i].scatter(sampled_symbol1[0], sampled_symbol1[1])
            else:
                axarr[0, i].plot(t1, sampled_symbol1)
            # axarr[0, i].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
            # axarr[0, i].set_title('Symbol-1')


            # ------------- ABSOLUTE DISTANCE FEATURE
            # Absolute distance plot of symbol-1
            axarr[1, i].plot(t1, f1_symbol1)
            # axarr[1, i].set(xlabel = "Time", ylabel = "Normalized Distance")


            # ------------- SLOPE FEATURE
            # Y-wise distance plot of symbol-1
            axarr[2, i].plot(t1, f2_symbol1)
            # axarr[2, i].set(xlabel = "Time", ylabel = "Slope(Degrees)")

            if not model:
                axarr[0, i].set_xlim([0,210])
                axarr[0, i].set_ylim([0,210])

                axarr[1, i].set_ylim([0,np.max(f1_symbol1)])

                axarr[2, i].set_ylim([-120,120])
            else:
                axarr[0, i].set_ylim([-1, np.max(sampled_symbol1) + 1])
                axarr[1, i].set_ylim([-5, np.max(f1_symbol1)])
                axarr[2, i].set_ylim([-120, 120])
            i += 1

        plt.show()
