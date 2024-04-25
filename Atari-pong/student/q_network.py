import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal,Categorical
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image 

class Qtable(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.init_network()
    
    def init_network(self):
        self.convs = nn.Sequential(
            nn.Conv2d(self.num_inputs, 32, 8, stride=4, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten())
        
        
        self.q_net = nn.Sequential(
            nn.Linear(2304, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_outputs)
        )
    
    def forward(self, x):
        x = self.convs(x)
        value = self.q_net(x)
        return value
    
class QNetwork:
    def __init__(self, num_inputs, num_outputs, hidden_size):
        self.model = Qtable(num_inputs, num_outputs, hidden_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-4)
    
    def update(self, state, action, q_value):
        self.optimizer.zero_grad()
        outputs = self.model.forward(state)
        target_distribution = outputs[:, action]  # Extract the target distribution
        loss = self.criterion(target_distribution, torch.tensor([q_value], dtype=torch.float32))  # Calculate the loss only for the target node
        loss.backward()  # Backward pass
        self.optimizer.step()    

    
# env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
# visualize the original image vs. the newly processed array.

# raw_image = env.reset()
# _, preprocessed_image = prepro(raw_image[0])
# print(preprocessed_image.shape)
# plt.imshow(preprocessed_image.squeeze(0))
# # im = Image.fromarray(raw_image[0]) 
# # im.save("your_file.jpeg")
# # print(raw_image[0].shape)
# # plt.imshow(raw_image[0])
# # plt.show()
# env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
# model = Qtable(1, 4, 256)
# obs = env.reset()[0]
# obs, _ = prepro(obs)