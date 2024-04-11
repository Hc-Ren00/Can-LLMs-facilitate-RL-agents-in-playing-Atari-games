import numpy as np
from .actorcritic import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

class PPO:
    def __init__(self,num_inputs,num_outputs,hidden_size,device,lr, env, num_steps=20):
        self.env = env
        self.device = device
        self.model = ActorCritic(num_inputs,num_outputs,hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_steps = num_steps
    
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        #print(returns)
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            #print(returns[rand_ids, :])
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]      

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test_env(self):
        print("testing------>>>>>.")
        state = self.env.reset()
        done = False
        total_reward = 0
        n=0
        while not done:
            state = torch.FloatTensor(state).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = self.env.step(dist.sample().cpu().numpy())
            state = next_state
            n+=1
            total_reward += reward
        print("-----test returned -----:",total_reward,n)
        return total_reward
    
    def plot(self, frame_idx, rewards):
        #clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
        plt.plot(rewards)
        plt.show()

    def train(self):
        max_frames = 20000
        threshold_reward = -10
        frame_idx  = 0
        test_rewards = []   
        state = self.env.reset()
        early_stop = False
        ppo_epochs = 4
        mini_batch_size = 5
        q_table={}
        episodes=0
        visits={}

        while frame_idx < max_frames and not early_stop:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0

            for _ in range(self.num_steps):
                #print("original state:", state)
                state = torch.FloatTensor(state).to(self.device)
                fstate = tuple(state.tolist()[0])
                if fstate in visits:
                    visits[fstate]+=1
                else:
                    visits[fstate]=1
                #print("changed state:",state)
                dist, value = self.model(state)
                #print("returned values from forward propagation - ",dist,value)

                action = dist.sample()
                #print(action,action.cpu().numpy())
                if action == 0:  # Up
                    dir = "up"
                elif action == 1:  # Down
                    dir = "down"
                elif action == 2:  # Left
                    dir = "left"
                elif action == 3:  # Right
                    dir = "right"
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                if done:
                    episodes+=1
                #print(f"Executed {dir} for {state}, reached {next_state} with reward {reward}")
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state
                frame_idx += 1
                
                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test_env() for _ in range(10)])
                    test_rewards.append(test_reward)
                    print("---------------STATE-PICKING TIME---------------------")
                    struggling_state=None
                    sq=float('-inf')
                    for s in q_table:
                        maxq=0
                        minq=0
                        unexplored=False
                        for a in q_table[s]:
                            q_value = q_table[s][a]
                            if q_value==0:
                                unexplored=True
                                break
                            else:
                                minq=min(minq,q_value)
                                maxq=max(maxq,q_value)
                        if unexplored==True:
                            print(f"skipping state {s} due to it being unexplored")
                            continue
                        rangeq=maxq-minq
                        print(f"state {s} has range {rangeq} and visits {visits[s]}")
                        if sq<rangeq and visits[s]>(frame_idx/10):
                            sq=rangeq
                            struggling_state = s
                    print(f"struggling state - {struggling_state}")

                    
                    

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            #update q table
            for state1,value,action,adv in zip(states,values,actions,advantage):
                state1=tuple(state1.tolist())
                if state1 not in q_table:
                    q_table[state1]={0:0,1:0,2:0,3:0}
                q_table[state1][action.tolist()]=adv.tolist()[0]+value.tolist()[0]
            print(f"-------------------After {frame_idx} steps-------------------------")
            print(q_table, episodes)
            #break
            
            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        #print(frame_idx)
        self.test_env()
