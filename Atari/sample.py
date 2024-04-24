import gym
import matplotlib.pyplot as plt
import random

# Create the Pong environment
#env = gym.make("ALE/Pong-v5",render_mode="human")

# Reset the environment to get the initial state
#observation = env.reset()

# Define the action meanings
LEFT = 0
RIGHT = 1
i=0
#print(env.observation_space.shape)
from atariari.benchmark.wrapper import AtariARIWrapper
env = AtariARIWrapper(gym.make('Breakout-v4'))
y=env.reset()
print(y)
# x=env.step(1)
# print(env.step(1)[-1])
i=0
done=False
while(not done):
    action=random.randrange(4)
    obs, reward, done, info,  = env.step(action)
    print(f"for image {i}, info is {info}")
    plt.imshow(obs)
    plt.savefig("./images/state"+str(i)+".png")
    i+=1

# print(obs.shape)
# print(info)
# plt.imshow(obs)
# plt.show()
# x=dict()
# print(tuple(map(tuple,obs)))
# Play a game of Pong
# while i<5:
#     # Render the environment (optional)
#     env.render()

#     # Choose an action randomly (for demonstration purposes)
#     action = env.action_space.sample()

#     # Perform the chosen action
#     observation, reward, done, info, _ = env.step(action)

#     # Check if the episode is done (game over)
#     if done:
#         break
#     i+=1

# Close the environment
# env.close()
