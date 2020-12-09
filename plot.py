import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------
# For User: change you model directory here

model = "models/snake_1/test_result_1"

# range of your episodes
ep_range = [i for i in range(40, 100, 1)]

# specific epoch for Move-Score Scatter
epoch_MSC = 100

# plot which graphes
plot_reward = False
plot_score = False 
plot_MSC = True


#-----------------------------------------------
# you don't need to edit the following
# if you are not adding new plots

ep_index = np.loadtxt("{}/ep_index.txt".format(model))
performance = np.loadtxt("{}/performance.txt".format(model))

n_test = int(performance.shape[0]/len(ep_index))
# reshape performance to 3D array
# axis 0: entire epoch;
# axis 1: instances within each epoch
# axis 2: reward, score, moves
performance = np.reshape(performance, (len(ep_index), n_test, 3))

# performance analysis
mean = np.mean(performance, axis=1)
std  = np.std(performance, axis=1)
max_ = np.max(performance, axis=1)
min_ = np.min(performance, axis=1)

#-----------------------------------------------
# Cumulative Reward
if plot_reward:
    # mean
    plt.plot(ep_index[ep_range], mean[ep_range, 0], "b")
    # mean + std
    plt.plot(ep_index[ep_range], mean[ep_range, 0]+std[ep_range, 0], "r:")
    # mean - std
    plt.plot(ep_index[ep_range], mean[ep_range, 0]-std[ep_range, 0], "r:")
    # Dot Plot
    for i in ep_range:
        plt.plot([ep_index[i] for _ in range(n_test)], performance[i,:,0], ".")

    plt.xlabel("epoches")
    plt.ylabel("cumulative reward")
    plt.show()


#-----------------------------------------------
# Score
if plot_score:
    # mean
    plt.plot(ep_index[ep_range], mean[ep_range, 1], "b")
    # mean + std
    plt.plot(ep_index[ep_range], mean[ep_range, 1]+std[ep_range, 1], "r:")
    # mean - std
    plt.plot(ep_index[ep_range], mean[ep_range, 1]-std[ep_range, 1], "r:")
    # Dot Plot
    for i in ep_range:
        plt.plot([ep_index[i] for _ in range(n_test)], performance[i,:,1], ".")

    plt.xlabel("epoches")
    plt.ylabel("score")
    plt.show()


#-----------------------------------------------
# Move-Score Scatter
if plot_MSC:
    plt.scatter(performance[epoch_MSC-1, :, 2], performance[epoch_MSC-1, :, 1])

    plt.xlabel("moves")
    plt.ylabel("score")
    plt.show()