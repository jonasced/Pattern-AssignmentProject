import numpy as np
from PattRecClasses import HMM_TA
import pandas as pd


def hmm_gen(data_features, thr):
    K = len(data_features)
    hmms = []

    for k in range(K):
        obs_list = data_features[k]

        """ Goals:
        def hmms = hmm_gen(data_features,thr):
        
        Input:
                data_features[k][r] : np.array (featdim, t), K letters, R samples, Tr lengths
                thr : int, threshold of where we implement state transitions
        Output:
                hmms : list of generated HMM models with approximately ok parameters
    
        1. Assign number of states -> when two samples jump over a treshold value
        
        2. Mean of each state, feature
        
        3. Variance of each state, feature
    
        """
        # for r in range(obs_list):
        # maybe only do for one sample?
        # obs = obs_list[r]
        obs = obs_list[0]  # take one arbitrary sample and use it!
        sample_state = []
        samples0 = []
        samples1 = []
        start = 0

        Tr = obs.shape[1]
        for x in range(Tr):
            if x == Tr-1:
                if not start == Tr-1:
                    sample_state += [x]
                    sample0 = np.array(obs[0, start-1:x])
                    sample1 = np.array(obs[1, start-1:x])
                    samples0 += [sample0]
                    samples1 += [sample1]

            elif (obs[0, x+1] - obs[0, x]) > thr or (obs[1, x + 1] - obs[1, x]) > thr:
                sample_state += [x]
                sample0 = np.array(obs[0, start:x+1])
                sample1 = np.array(obs[1, start:x+1])
                samples0 += [sample0]
                samples1 += [sample1]

                start = x+1

        nStates = len(sample_state)
        means = np.empty([nStates, 2])

        for i in range(nStates):
            means[i, 0] = np.mean(samples0[i])
            means[i, 1] = np.mean(samples1[i])

        # Initial prob and transition matrix assignment
        qstar = np.zeros(nStates)
        qstar[0] = 1
        Astar = np.zeros([nStates, nStates])
        for i in range(nStates):
            if i == nStates-1:
                Astar[i,:] = 0.1/nStates
                Astar[i, i] = 1 - 0.1
            else:
                Astar[i,:] = 0.1/nStates
                Astar[i, i+1] = 0.1
                Astar[i, i] = 0.8

        # Covariance and B assignment
        temp = []
        for i in range(nStates):
            temp += [np.ones([2, 2])]
        covsstar = np.array(temp)

        temp = []
        for i in range(nStates):
            temp += [HMM_TA.multigaussD(means[i], covsstar[i])]
        Bstar = np.array(temp)


        print("------------ CHARACTER", k, "------------")
        print("Number of states", nStates)
        print("qstar", qstar)
        print("Astar", Astar)
        print("Bstar", means)
        print("\n")
        hmms += [HMM_TA.HMM(qstar, Astar, Bstar)]
    return hmms


def main():
    db_name = "database_test"
    data_features = pd.read_pickle(r'data/' + db_name + '_features.cdb')
    hmm_learn = hmm_gen(data_features, 5)


if __name__ == "__main__":
    main()

