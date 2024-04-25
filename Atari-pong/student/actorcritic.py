import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal,Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten())

        self.critic = nn.Sequential(
            nn.Linear(2304, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )   

        self.actor = nn.Sequential(
            nn.Linear(2304, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        #print(f"passing {x,x.shape} to the network")
        embeddings = self.convs(x)
        value = self.critic(embeddings)
        mu    = self.actor(embeddings)
        lol = Categorical(logits=f.log_softmax(mu))
        return lol, value