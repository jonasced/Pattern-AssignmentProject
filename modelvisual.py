from featurevisual import featurevisual
import numpy as np


def modelvisual(hmm, data_labels, samples, chars=[]):

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

    featurevisual(model_features, data_labels, model_states, chars=chars, plt_num=0, model=True)


def main():
    modelvisual()


if __name__ == "__main__":
    main()