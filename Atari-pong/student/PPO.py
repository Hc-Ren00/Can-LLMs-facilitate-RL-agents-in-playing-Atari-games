import numpy as np
from .actorcritic import *
from .q_network import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from utils import *
import ast

class PPO:
    def __init__(self,num_inputs,num_outputs,hidden_size,device,lr, env, num_steps=20):
        self.env = env
        self.device = device
        self.model = ActorCritic(num_inputs,num_outputs,hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_steps = num_steps
        self.max_frames = 20000
        self.struggling_states = set()
        self.teacher_recommendations = dict()
        #threshold_reward = -10
        self.test_rewards = [] 
        self.data_points=[]  
        self.curr_state = self.env.reset()[0]
        self.early_stop = False
        self.ppo_epochs = 4
        self.mini_batch_size = 5
        self.q_network=QNetwork(num_inputs, num_outputs, hidden_size)
        self.local_q_buffer={}
        self.episodes=0
        self.visits={}
        self.num_steps_with_kickstart_loss=0
        

    def generate_teacher_logits(self,states):
        teacher_probs = []
        print(states, self.teacher_recommendations)
        for s in states:
            #processed_state = tuple([int(x) for x in s.tolist()])
            s_teacher_probs = [0]*6
            fstate = str(s.tolist())
            if fstate in self.teacher_recommendations:
                s_teacher_probs[self.teacher_recommendations[fstate]]=1
            teacher_probs.append(s_teacher_probs)
        return torch.from_numpy(np.array(teacher_probs))
    
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
                # state = torch.tensor(ast.literal_eval(str_state))
                dist, value = self.model(state)
                #print(state,dist.logits,value)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)
                #print(dist.logits, self.generate_teacher_logits(state), -(dist.logits * self.generate_teacher_logits(state)).sum(dim=-1).mean())
                kickstarting_loss = -(dist.logits * self.generate_teacher_logits(state)).sum(dim=-1).mean()
                print(f"calculating kickstarting loss - {kickstarting_loss}")
                #print(state,self.struggling_states)
                # if state in self.struggling_states:
                #     teacher_prob_batch = np.zeros(4)
                #     teacher_prob_batch[self.teacher_recommendations[state]]=1
                #     kickstarting_loss = -(dist.logits * self.generate_teacher_logits(state)).sum(dim=-1).mean()
                #     print(f"calculating kickstarting loss - {kickstarting_loss}")

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                if kickstarting_loss>0:
                    self.num_steps_with_kickstart_loss+=1
                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy + 12*kickstarting_loss
                print(f"Total loss - {loss}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test_env(self):
        print("testing------>>>>>.")
        state = self.env.reset()
        done = False
        total_reward = 0
        gamma=0.9
        n=0
        while not done:
            state = torch.FloatTensor(state).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, terminated, truncated, info  = self.env.step(dist.sample().cpu().numpy())
            state = next_state
            total_reward += ((gamma**n)*reward)
            n+=1
        print("-----test returned -----:",total_reward,n)
        return total_reward
    
    def plot(self, frame_idx, rewards):
        #clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
        plt.plot(rewards)
        plt.show()

    def train(self, frame_idx):

        while frame_idx < self.max_frames and not self.early_stop:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            infos     = []
            prev_infos = []
            prev_info = ''
            entropy = 0

            for _ in range(self.num_steps):
                print("original state:", self.curr_state.shape)
                self.curr_state, _ = preprocess_image_pong(self.curr_state)#torch.FloatTensor(self.curr_state).to(self.device)
                print("processed_state:", self.curr_state.shape)
                list_tensor = self.curr_state.tolist()
                fstate = str(list_tensor)
                if fstate in self.visits:
                    self.visits[fstate]+=1
                else:
                    self.visits[fstate]=1
                #print("changed state:",state)
                dist, value = self.model(self.curr_state)
                print("returned values from forward propagation of actor-critic - ",dist,value)
                action = dist.sample()
                print(action,action.cpu().numpy())
                next_state, reward, terminated, info  = self.env.step(action.cpu().numpy()[0])
                print(f"got the following params from step - {next_state},{reward},{terminated}")
                if terminated:
                    self.episodes+=1
                #print(f"Executed {dir} for {state}, reached {next_state} with reward {reward}")
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                print(terminated)
                log_probs.append(log_prob)
                values.append(value)
                infos.append(info)
                prev_infos.append(prev_info)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - terminated]).unsqueeze(1).to(self.device))
                print(self.curr_state.shape)
                states.append(self.curr_state)
                actions.append(action)
                
                self.curr_state = next_state
                frame_idx += 1
                prev_info = info

                # if frame_idx%100==0:
                #     self.data_points.append(np.mean([self.test_env() for _ in range(15)]))
                
                # if frame_idx % 1000 == 0:
                #     test_reward = np.mean([self.test_env() for _ in range(10)])
                #     self.test_rewards.append(test_reward)

            next_state, _ = preprocess_image_pong(next_state)
            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            print("-----------", states)
            states    = torch.cat(states)
            print(states.shape,states)
            actions   = torch.cat(actions)
            advantage = returns - values
            #update q table
            #replace q-network
            for state1,value,action,adv,information,pinfo in zip(states,values,actions,advantage,infos,prev_infos):
                list_tensor = state1.tolist()
                str_tensor = str(list_tensor)
                if str_tensor not in self.local_q_buffer:
                    self.local_q_buffer[str_tensor]={"q-values":{0:0,1:0,2:0,3:0}}
                q_value = adv.tolist()[0]+value.tolist()[0]
                self.local_q_buffer[str_tensor]["q-values"][action.tolist()]=adv.tolist()[0]+value.tolist()[0]
                self.local_q_buffer[str_tensor]["info"] = information
                self.local_q_buffer[str_tensor]["prev_info"]=pinfo
                self.q_network.update(state1.unsqueeze(0),action.tolist(),q_value)
            print(f"-------------------After {frame_idx} steps-------------------------")
            #print(self.local_q_buffer, self.episodes)
            #break
            
            self.ppo_update(self.ppo_epochs, self.mini_batch_size, states, actions, log_probs, returns, advantage)
        #print(frame_idx)
        #self.test_env()
        #return self.num_steps_with_kickstart_loss
