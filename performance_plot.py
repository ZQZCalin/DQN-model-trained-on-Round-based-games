import matplotlib.pyplot as plt 
from util import *
sys.path.append(os.path.abspath('Snake'))
from snakeEnv import *
from DQN_model import *

if False:
    performance = pd.read_csv("models/snake_6/performance.csv", index_col=0)
    episode = performance["episode"]
    cum_reward = performance["cumulative_reward"]
    avg_reward = cum_reward.copy()
    for i in range(20, len(avg_reward)):
        avg_reward[i] = np.mean(cum_reward[i-19:i+1])
    moves = performance["moves"]
    score = performance["score"]

    # plt.plot(episode, cum_reward, "blue")
    # plt.plot(episode, moves, "orange")
    # plt.show()
    # plt.plot(episode, score)
    # plt.show()

    plt.show(episode, avg_reward)
    plt.show()

def simulate_game(env, agent, max_moves, fps):

    state = env.reset()
    state = np.reshape(state, [1,state_size])
    cum_reward = 0
    score = 0

    for move in range(max_moves):
            if fps != 0:
                env.render(FPS=fps)

            # use exploit() instead of act()
            action = agent.exploit(state)
            next_state, reward, done, score = env.step(action)
            # update state
            state = np.reshape(next_state, [1,state_size])
            cum_reward += reward

            if done:
                break
    
    return cum_reward, move, score

if True:

    # fetch parameters
    state_size = 12
    action_size = 4
    n_tests = 50
    max_moves = 1000
    fps = 0

    model_dir = "models/snake_6"

    save_data = True

    # save data

    test_perform = pd.DataFrame({"epoch":[],"cum_reward":[],"moves":[],"score":[]})

    for epoch in range(5,101,5):

        env, agent = load_env_agent("{}/env_agent_backup/model_{}.pkl".format(model_dir, epoch))
        agent.load("{}/model_weights/weights_{}.hdf5".format(model_dir, epoch))

        # start testing
        data = []

        for e in range(n_tests):
            new_data = simulate_game(env, agent, max_moves, fps)
            data.append(new_data)
            
            if e % 25 == 0:
                print(new_data)
        
        data = np.mean(data, axis=0)
        data = np.concatenate(([epoch], data))
        new_row = pd.DataFrame(np.reshape(data,[1,4]),columns=list(test_perform.keys()))
        test_perform = test_perform.append(new_row, ignore_index=True)

        print("Finished epoch: {}".format(epoch))

    if save_data:
        test_perform.to_csv("{}/test_perform.csv".format(model_dir),index=False)
    
    cum_reward = test_perform["cum_reward"]
    avg_reward = cum_reward.copy()

    for i in range(20, len(avg_reward)):
        avg_reward[i] = np.mean(cum_reward[i-19:i+1])
    
    plt.plot(test_perform["epoch"],avg_reward)