import numpy as np
from hmm_gen import hmm_gen
from dataprep import dataprep


def modeltrain(train_data, labels, iter, num_states, longest_sample=False, useprint=True, chars=[], test=False):
    """
    hmm_learn, train_acc = modeltrain(train_data, labels): Trains hmms and outputs them in a list

    Input:
        train_data: train_data[k][r] is a np array of features
        labels: length K list of labels

    Output:
        hmm_learn: leangth K list of learned hmms
        train_acc: length K list of training accuracies, being the probability of the entire sequence belonging
            to the specific model
    """
    # Initialize hmms with feasible parameters
    hmm_learn = hmm_gen(train_data, num_states, longest_sample=longest_sample, useprint=useprint, test=test)


    # Training
    train_acc = []
    if len(chars) == 0:
        K = range(len(train_data))
    else:
        print("Only training models", chars, "corresponding to letter", labels[chars[0]])
        K = chars

    for k in K:
        if isinstance(num_states, (np.ndarray, np.generic)):
            if num_states.size == 1:
                nStates = num_states
            else:
                nStates = num_states[k]
        else:
            nStates = num_states
        print("\n ------------ CHARACTER ", labels[k], ", k =",k, "------------")

        # Train the model using baum welch
        if useprint:
            val = 1
        else:
            val = 0
        hmm_learn[k].baum_welch(train_data[k], iter, prin=val, uselog=False)

        # Training accuracy
        lprob_list = []
        for r in range(len(train_data[k])):
            a, c = hmm_learn[k].alphahat(train_data[k][r])
            clog = np.log(c)
            lprob = np.sum(np.array(clog))
            lprob_list += [lprob]

            # Prints probability of each step in sequence
            if useprint:
                print("c is", c)

        # Calculates the average training accuracy
        avg = np.mean(np.array(lprob_list))
        train_acc += [avg]
        print("Number of states: ", len(hmm_learn[k].q))
        print("Avg probability for entire sequence over test samples is", avg, " (log), ", np.exp(avg)*100, "%")
        print("Normalized score: " , np.exp(avg)*100* (10**nStates))

    return hmm_learn, train_acc


def main():
    import pickle
    import pandas as pd
    train_data, test_data, labels = dataprep("database_inc_sampchar", 10)
    class_state_nums = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    chars = []
    #class_state_nums[6] = 6
    hmm_learn, train_acc = modeltrain(train_data, labels, 12, class_state_nums, chars=chars)

    # name = "varStateTest" + str(labels[chars[0]]) + str(class_state_nums[chars[0]]) + ".model"
    name = "finiteTest.model"
    with open(name, "wb") as fp:  # Pickling
        pickle.dump(hmm_learn, fp)


if __name__ == "__main__":
    main()