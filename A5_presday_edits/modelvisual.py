from featurevisual import featurevisual
import numpy as np
import matplotlib.pyplot as plt


def modelvisual(hmm, data_labels, samples, chars=[], overview=False, save=False):
    if not overview:
        mchars = [0,1,2,3,4,5,6,7,8,9]

        model_features = []
        model_states = []
        for mchar in mchars:
            model_samples = []
            model_sample_states = []
            for i in range(samples):
                feature_model, states = (hmm[mchar].rand(30))
                feature_model = np.transpose(feature_model)
                model_samples += [feature_model]
                model_sample_states += [states]

            model_features += [model_samples]
            model_states += [model_sample_states]

        featurevisual(model_features, data_labels, model_states, chars=chars, plt_num=[], model=True, save=save)

    else:
        # Plot some sequences from each model
        mchars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        f, axarr = plt.subplots(3, len(mchars), dpi=200)
        f.suptitle('Generated model feature sequences from all models', fontsize=10)

        axarr[0, 0].set(xlabel="Time", ylabel="States")
        axarr[1, 0].set(xlabel="Time", ylabel="Normalized Distance")
        axarr[2, 0].set(xlabel="Time", ylabel="Slope(Degrees)")

        i = 0
        for mchar in mchars:
            feature_symbol1, states = (hmm[mchar].rand(30))
            feature_symbol1 = np.transpose(feature_symbol1)

            # normalized distance ,slope, and t for symbol-1
            f1_symbol1 = feature_symbol1[0]
            f2_symbol1 = feature_symbol1[1]
            t1 = np.array(range(0, feature_symbol1.shape[1]))
            axarr[0, i].title.set_text(data_labels[mchar])

            # ------------- HIDDEN STATES
            axarr[0, i].plot(t1, states)
            axarr[0, i].set_ylim([-1, np.max(states) + 1])

            # ------------- ABSOLUTE DISTANCE FEATURE
            # Absolute distance plot of symbol-1
            axarr[1, i].plot(t1, f1_symbol1)
            axarr[1, i].set_ylim([-5, np.max(f1_symbol1)])

            # ------------- SLOPE FEATURE
            # Slope plot of symbol-1
            axarr[2, i].plot(t1, f2_symbol1)
            axarr[2, i].set_ylim([-120, 120])

            i += 1

        plt.show()


def main():
    import pandas as pd
    from dataprep import dataprep

    hmm_learn = pd.read_pickle(r'varStateTestSum.model')
    #hmm_learn = pd.read_pickle(r'hmm_demo')
    db_name = "Bigdata"
    train_data, test_data, data_labels = dataprep(db_name, nr_test=5, useprint=False)
    samples = 10
    modelvisual(hmm_learn, data_labels, samples, chars=[3])


if __name__ == "__main__":
    main()