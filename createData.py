#!/usr/bin/python3
import os
import matplotlib.pyplot as plt
from DrawCharacter import *
from CharacterFeatureExtractor import *


def createData(plot=True, save=True):
    """ pts, feature_symbol, sampled_symbol = createData(plot=True, save=True)
    Used to create data with cursor, options to plot and save afterwards
    """

    ch = DrawCharacter()
    ch.run()
    pts = ch.get_xybpoints()

    # Feature vectors are returned
    thr = 8  # threshold for sampling and distance normalization
    feature_symbol, sampled_symbol = featureExtractor(pts, thr, input_is_dc=False)

    if plot:
        # normalized distance ,slope, and t for symbol-1
        f1_symbol = feature_symbol[0]
        f2_symbol = feature_symbol[1]
        t = np.array(range(0,feature_symbol.shape[1]))

        f, axarr = plt.subplots(3,1, dpi=200)
        axarr = axarr.reshape(-1,1)
        f.suptitle('Char Feature Check for Char Database', fontsize=20)

        # ------------- SYMBOL DRAWINGS
        # Drawing of sampled symbol
        axarr[0, 0].scatter(sampled_symbol[0], sampled_symbol[1])
        axarr[0, 0].set(xlabel = "X-Coordinate", ylabel = "Y-Coordinate")
        axarr[0, 0].set_title('Symbol')
        axarr[0, 0].set_xlim([0,210])
        axarr[0, 0].set_ylim([0,210])

        # ------------- ABSOLUTE DISTANCE FEATURE

        # Absolute distance plot of symbol
        axarr[1, 0].plot(t, f1_symbol)
        axarr[1, 0].set(xlabel = "Time", ylabel = "Normalized Distance")
        axarr[1, 0].set_ylim([0,np.max(f1_symbol)])

        # ------------- SLOPE FEATURE

        # Y-wise distance plot of symbol
        axarr[2, 0].plot(t, f2_symbol)
        axarr[2, 0].set(xlabel = "Time", ylabel = "Slope(Degrees)")
        axarr[2, 0].set_ylim([-120,120])

        plt.show()
        plt.pause(1)

        # pausing till the figure is closed otherwise figure is not responding
        while True:
            if plt.fignum_exists(f.number):
                plt.pause(0.1)
            else:
                break

    if save:
        # Getting current path of current file to save the data to its location
        current_path = os.path.dirname(os.path.abspath(__file__))

        # Asking if user wants to save the file
        saveit = None
        decision = input("Do you want to save it? ('y' for yes, 'n' for no): ")

        if decision == "y":
            saveit = True
        elif decision == "n":
            saveit = False

        # If user wants to save the file, asking for its name and saving the file to the location of current file
        if saveit:
            name = input("What do you want to save the file as?")
            np.save(os.path.join(current_path, name), pts)

    return pts, feature_symbol, sampled_symbol


def main():
    createData()


if __name__ == "__main__":
    main()
