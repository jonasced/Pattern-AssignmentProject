import numpy as np
from PattRecClasses import HMM_TA
import pandas as pd


def hmm_gen(data_features, num_states, useprint=True, longest_sample = False):
    """ hmms = hmm_gen(data_features,thr): Generates hmm models with feasible starting distributions, to be used in training.
    Input:
        data_features[k][r] : np.array (featdim, t), K letters, R samples, Tr lengths
        num_States : number of states to be included in in hmm model for each class. If it is an array, each position will contain num of states for corresponding class
    Output:
        hmms : list of generated HMM models with approximately ok parameters

    Workflow:
    1. Assign number of states -> when two samples jump over a treshold value
    2. Mean of each state, feature
    3. Variance of each state, feature
    """

    hmms = []

    # Test for correct data structure:
    featdim = data_features[1][1].shape
    if featdim[1] != 2:
        raise Exception("Current dimensions are", featdim, "which are not correct! (15,2) is a correct example.")

    K = len(data_features)
    for k in range(K):
        obs_list = data_features[k]
        
        obs = obs_list[0]  # taking one arbitrary sample and using it!
        #If longest_sample setting is True we are looking for the longest observation for the specific class to have a more reliable solution with high res.
        if longest_sample:
            num_obs = len(obs_list)
            t_lengths = np.zeros(num_obs)
            for o in range(num_obs):
                t_lengths[o] = obs_list[o].shape[0]
            maxt_index = np.argmax(t_lengths)
            obs = obs_list[maxt_index]

       # sample_state = []
        samples0 = []
        samples1 = []
        T = obs.shape[0]
        if  isinstance(num_states, (np.ndarray, np.generic) ):
            if num_states.size == 1:
                nStates = num_states
            else:
                nStates = num_states[k]
        else:
            nStates = num_states
            
        sample_amount = int(T/nStates)
        for i in range(nStates):
            if i+1 == nStates: # if it is last state, include all the remaining samples in observation
                sample0 = np.array(obs[i*sample_amount :, 0 ])
                sample1 = np.array(obs[i*sample_amount :, 1 ])
                samples0 += [sample0]
                samples1 += [sample1]
            else:
                sample0 = np.array(obs[i*sample_amount : (i+1)*sample_amount,0 ])
                sample1 = np.array(obs[i*sample_amount : (i+1)*sample_amount,1 ])
                samples0 += [sample0]
                samples1 += [sample1]
                
                

        
        
        # start = 0
        # count = 0
        # T = obs.shape[0]
        # for x in range(T):
        #     count += 1
        #     if x == T-1:
        #         if not start == T-1:
        #             sample_state += [x]
        #             sample0 = np.array(obs[start-1:x, 0])
        #             sample1 = np.array(obs[start-1:x, 1])
        #             samples0 += [sample0]
        #             samples1 += [sample1]
        #     elif ((obs[x+1, 0] - obs[x, 0]) > thr or (obs[x + 1, 1] - obs[x, 1]) > thr) and count > 2:
        #         sample_state += [x]
        #         sample0 = np.array(obs[start:x+1, 0])
        #         sample1 = np.array(obs[start:x+1, 1])
        #         samples0 += [sample0]
        #         samples1 += [sample1]

        #         start = x+1
        #         count = 0

        # nStates = len(sample_state)

        # Estimate means of each state sequence
        means = np.empty([nStates, 2])
        for i in range(nStates):
            means[i, 0] = np.mean(samples0[i])
            means[i, 1] = np.mean(samples1[i])

        # Estimate covariance
        temp = []
        for i in range(nStates):
            v0, v1 = [1, 1] 
            if len(samples0[i]) > 1:
                v0 = np.var(samples0[i])
            if len(samples1[i]) > 1:
                v1 = np.var(samples1[i])
            temp += [np.array([[v0, 0], [0, v1]])]
        covsstar = np.array(temp)

        # Assign Bstar
        temp = []
        for i in range(nStates):
            temp += [HMM_TA.multigaussD(means[i], covsstar[i])]
        Bstar = np.array(temp)

        # Initial prob and transition matrix assignment, weighted towards sequence probabilities
        qstar = np.zeros(nStates)
        qstar[1:] = 0.1/nStates
        qstar[0] = 0.9
        Astar = np.zeros([nStates, nStates])
        for i in range(nStates):
            if i == nStates-1:
                Astar[i, :] = 0.1/nStates
                Astar[i, i] = 1 - 0.1
            else:
                Astar[i, :] = 0.1/nStates
                Astar[i, i+1] = 0.1
                Astar[i, i] = 0.8
        
        if useprint:
            print("------------ CHARACTER", k, "------------")
            print("Number of states", nStates)
            print("qstar", qstar)
            print("Astar", Astar)
            print("Bstar mean:", means, "covariance:", covsstar)
            print("\n")

        # Create HMM and add to list
        hmms += [HMM_TA.HMM(qstar, Astar, Bstar, finite=True)]

    return hmms


def main():
    db_name = "database_test"
    data_features = pd.read_pickle(r'data/' + db_name + '_features.cdb')
    hmm_learn = hmm_gen(data_features, 5)


if __name__ == "__main__":
    main()

