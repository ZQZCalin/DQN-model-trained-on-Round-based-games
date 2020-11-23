import matplotlib.pyplot as plt
import numpy as np

# change you model directory here
model = "models/snake_7"

ep_index = np.loadtxt("{}/test_all_result/ep_index.txt".format(model))
performance = np.loadtxt("{}/test_all_result/performance.txt".format(model))

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

ep_range = [i for i in range(40, 100, 1)]

#-----------------------------------------------
# Cumulative Reward
# mean
plt.plot(ep_index[ep_range], mean[ep_range, 0], "b")
# mean + std
plt.plot(ep_index[ep_range], mean[ep_range, 0]+std[ep_range, 0], "r:")
# mean - std
plt.plot(ep_index[ep_range], mean[ep_range, 0]-std[ep_range, 0], "r:")
# Dot Plot
for i in ep_range:
    plt.plot([ep_index[i] for _ in range(n_test)], performance[i,:,0], ".")

plt.show()


#-----------------------------------------------
# Score
# mean
plt.plot(ep_index[ep_range], mean[ep_range, 1], "b")
# mean + std
plt.plot(ep_index[ep_range], mean[ep_range, 1]+std[ep_range, 1], "r:")
# mean - std
plt.plot(ep_index[ep_range], mean[ep_range, 1]-std[ep_range, 1], "r:")
# Dot Plot
for i in ep_range:
    plt.plot([ep_index[i] for _ in range(n_test)], performance[i,:,1], ".")


plt.show()
