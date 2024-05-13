# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI
from dotenv import load_dotenv
import tyro
import ast
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from atariari.benchmark.wrapper import AtariARIWrapper
import torchvision.transforms.functional as TF
from process_image import ProcessPong

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL-raw-ppo"
    """the wandb's project name"""
    wandb_entity: str = "kkapoor-13"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Pong-v4"
    """the id of the environment"""
    total_timesteps: int = 3e6
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def save_img(next_obs, ctr):
    img = next_obs.squeeze(0)
    for i in range(4):
        plt.imshow(img[i])  # Assuming grayscale image, change cmap if needed
        plt.savefig("observations/image-"+str(ctr)+"-"+str(i)+".png")

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = ProcessPong(env, (80, 80))
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = AtariARIWrapper(env)
        
        return env

    return thunk

def preprocess_image_pong(image, device):
    pong_inputdim = (1, 80, 80)
    image = image.squeeze(axis=0)
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    tens = TF.to_tensor(image).to(device)
    tens = tens.unsqueeze(3)
    return tens, np.reshape(image, pong_inputdim)

struggling_states_n = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

"""
2M-3M timesteps
try enabling checkpoint 
1 - MaxAndSkip + FrameStack + entire stack as an imp state
2 - SkipNoMax + FrameStack + entire stack as an imp state
3 = MaxAndSkip + FrameStack + last state as an imp state
4 - SkipNoMax + FrameStack + last state as an imp state
5 - MaxAndSkip + FrameStack + similarity
same project
log the kick-starting loss
"""
def find_imp_states(q_table, q_network, device, struggling_states):
    global struggling_states_n
    struggling_state=None
    struggling_qvalues=None
    sq=float('-inf')
    #print(len(q_table))
    for s in q_table:
        fstate = torch.tensor(ast.literal_eval(s)).unsqueeze(0).to(device)
        q_values = q_network.model(fstate)
        rangeq=torch.max(q_values)-torch.min(q_values).to(device)
        key = s+"***"+str(q_table[s]['info'])+"#"+str(q_table[s]['prev_info'])
        if key not in struggling_states and sq<rangeq:# and self.student.visits[s]>(frame_idx/10):
            sq=rangeq
            struggling_qvalues=q_values
            struggling_state = s
    #print(q_table[struggling_state])
    si = q_table[struggling_state]['info']
    spi = q_table[struggling_state]['prev_info']
    x = q_table[struggling_state]["counter"]
    print(f"********IDENTIFIED {x} AS A STRUGGLING STATE*******")
    print(f"STRUGGLING STATE NO {struggling_states_n} has info - {si}, prev_info - {spi}")
    save_img(torch.tensor(ast.literal_eval(struggling_state)),struggling_states_n)
    # print("*******Collecting 1 important state**********")
    struggling_states_n+=1
    return (struggling_state,si,spi)

def generate_teacher_logits(states, info, prev_info, struggling_states, device):
        teacher_probs = []
        for s,info_s,prev_info_s in zip(states,info,prev_info):
            s_teacher_probs = [0]*6
            fstate = str(s.tolist())
            key = fstate+"***"+str(info_s)+"#"+str(prev_info_s)
            if key in struggling_states:
                # print("***********STRUGGLING STATE SEEN***************")
                s_teacher_probs[struggling_states[key]]=1
            teacher_probs.append(s_teacher_probs)
        #print(teacher_probs)
        return torch.from_numpy(np.array(teacher_probs)).to(device)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2304, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        #x = x.permute(0, 3, 1, 2)
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        #x = x.permute(0, 3, 1, 2)
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), logits
    
class Teacher():
    def __init__(self, model, prefix=''):
        self.prompt_prefix = prefix
        self.llm_model = model
        self.saved_teacher_recommendations={}
        load_dotenv(".env")
        self.client = OpenAI(api_key=os.environ["KEY"])

    def create_prompt(self, info, prev_info):
        player_coords = (info['player_x'],210-info['player_y'])
        enemy_coords = (info['enemy_x'],info['enemy_y'])
        ball_pos = (info['ball_x'],210-info['ball_y'])
        prev_ball_pos = (prev_info['ball_x'],210-prev_info['ball_y'])
        output = f"My x-coordinate is {player_coords[0]} and my y-coordinate is {player_coords[1]}. The ball is at the position {ball_pos}. Right before this state, the ball was at {prev_ball_pos}.\n"
        instructions = """
Following are the six actions available in the format (action id: action) -
(0: do nothing),
(1: FIRE),
(2: RIGHT),
(3: LEFT),
(4: RIGHTFIRE),
(5: LEFTFIRE). 
Which action would be the best to take in this situation to win? Please give me the answer in the format of (action: <action ID>). Example response:(action: 3)"""
        return output + instructions

    def RL2LLM(self, info, prev_info):
        key = str(info)+"#"+str(prev_info)
        if key in self.saved_teacher_recommendations:
            return self.saved_teacher_recommendations[key],False
        context = self.create_prompt(info, prev_info)
        #context = f'You are currently at {state}.\nMove up will reach {up}, move down will reach {down}, move left will reach {left}, move right will reach {right}.\nWhat\'s the best action you should take? Please give me the answer just in the format of (action: <action ID>).'
        return context,True

    def return_action(self,response):
        index = response.find("(action: ")
        action = response[index+9]
        return int(action)
        
    def query_codex(self, prompt_text):
        result=''
        while True:
            try:
                server_error_cnt = 0
                while server_error_cnt < 10:
                    try:
                        result = self.client.chat.completions.create(
                                model=self.llm_model,
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": self.prompt_prefix + prompt_text
                                            }
                                        ]
                                    }
                                ],
                                max_tokens=4096
                            ).choices[0].message.content
                        break                            
                    except Exception as e:
                        server_error_cnt += 1
                        continue
                        # print(f"fail to query")
                result = self.return_action(result)
                break
            except:
                # print(result)
                continue
        return result
    
    def prompt(self, states):
        plans = {}
        for x,y in states:
            text,need_to_prompt = self.RL2LLM(x,y)
            #print(f"Return value from RL2LLM - {text,need_to_prompt}")
            if need_to_prompt:
                # llm_outputs = np.zeros(6)
                # for i in range(5):
                action = self.query_codex(text)
                # llm_outputs[action]+=1
                # action = llm_outputs.argmax()
                key = str(x)+"#"+str(y)
                if key not in self.saved_teacher_recommendations:
                    self.saved_teacher_recommendations[key] = action
                print(f"Teacher LLM returned: plans - {action}")
            else:
                action = text
            #plans[state]=action
        #print(plans)
        return action#self.saved_teacher_recommendations
    
    
class Qtable(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2304, 512)),
            nn.ReLU(),
        )
        self.q_net = layer_init(nn.Linear(512, envs.single_action_space.n))
      
    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)
        x = self.network(x)
        value = self.q_net(x)
        return value
    
class QNetwork:
    def __init__(self,envs, device):
        self.device = device
        self.model = Qtable(envs).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-4)
    
    def update(self, state, action, q_value):
        self.optimizer.zero_grad()
        outputs = self.model.forward(state)
        target_distribution = outputs[:, action]  # Extract the target distribution
        loss = self.criterion(target_distribution, torch.tensor([q_value], dtype=torch.float32).to(self.device))  # Calculate the loss only for the target node
        loss.backward()  # Backward pass
        self.optimizer.step()    


if __name__ == "__main__":
    state_seen_again_n = 0
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.kickstart_coeff = 12
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    q_network = QNetwork(envs, device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    teacher = Teacher(model='gpt-3.5-turbo-1106', prefix="""
                      I'm playing a game of Atari Pong-v4 using the openai gym environment of grid size 210x210 with origin at the bottom-left corner.
My x-coordinate is 188 and my y-coordinate is 137. The ball is at the position (173,37). Right before this state, the ball was at (159,51). 
Following are the six actions available in the format (action id: action) -
(0: do nothing),
(1: FIRE),
(2: RIGHT),
(3: LEFT),
(4: RIGHTFIRE),
(5: LEFTFIRE). 
Which action would be the best to take in this situation to win? Please give me the answer in the format of (action: <action ID>). 
Reasoning: Considering the trajectory of the ball, it's moving towards the bottom right. Comparing my position and the ball's position, let's calculate the difference of y-coordinates only since x-coordinate of the bat is constant - 
my y-coordinate - ball y-coordinate = 137 - 37 = 100
Since this number is positive, it seems like I have to move downwards, or LEFT to save the ball.
Hence the output should be (action: 3). Example response:(action: 3)
Output: (action: 3)

Question:I'm playing a game of Atari Pong-v4 using the openai gym environment of grid size 210x210 with origin at the bottom-left corner.""")
    print("obs shape - ",envs.single_observation_space.shape)
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    info_arr = [None] * args.num_steps
    prev_info_arr = [None] * args.num_steps

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    img_ctr=0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    #save_img(next_obs,img_ctr)
    next_done = torch.zeros(args.num_envs).to(device)
    struggling_states = dict()
    local_q_buffer = {}
    prev_info = None
    x = []
    y = []
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        #next_obs, reward, terminations, truncations, infos = envs.step([3])
        ############
        if (iteration+1)%10==0:
            torch.save({
            'epoch': iteration,
            'actor_critic_state_dict': agent.state_dict(),
            'q_network_state_dict': q_network.model.state_dict(),
            'q_network_criterion': q_network.criterion.state_dict(),
            'q_network_optimizer_dict': q_network.optimizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"./checkpoints/checkpoint_{iteration}.pt")

            import json
            with open(f'./checkpoints/checkpoint_teacher_recommend_{iteration}.json', 'w') as fp:
                json.dump(teacher.saved_teacher_recommendations, fp)
            with open(f'./checkpoints/checkpoint_struggling_states_{iteration}.json', 'w') as fp:
                json.dump(struggling_states, fp)


        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if iteration%2==1 and iteration>1:
            ss, s_info, spinfo = find_imp_states(local_q_buffer, q_network, device, struggling_states)
            rec_action = teacher.prompt([(s_info, spinfo)])
            key = ss+"***"+str(s_info)+"#"+str(spinfo)
            struggling_states[key] = rec_action
            local_q_buffer = {}
        image_counter = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            info_arr[step] = prev_info
            prev_info_arr[step] = info_arr[step-1]

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            img_ctr+=1
            #save_img(next_obs, img_ctr)
            image_counter.append(img_ctr)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if global_step % 1000 == 0:
                x.append(global_step)
                def test_env():
                    envs_test = gym.vector.SyncVectorEnv(
                        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
                    )
                    obs_test, _ = envs_test.reset(seed=args.seed)
                    #obs_test, _ = preprocess_image_pong(obs_test, device)
                    obs_test = torch.Tensor(obs_test).to(device)
                    terminated = False
                    total_reward = 0
                    #gamma=0.9
                    while not terminated:
                        with torch.no_grad():
                            action, logprob, _, value, _ = agent.get_action_and_value(obs_test)
                        next_obs, reward, terminated, truncations, infos = envs.step(action.cpu().numpy())
                        #next_obs, _ = preprocess_image_pong(next_obs, device)
                        next_obs = torch.Tensor(next_obs).to(device)
                        obs_test = next_obs
                        total_reward += reward
                        #print("-----test returned -----:",total_reward,n)
                    return total_reward
                test_reward = np.mean([test_env() for _ in range(5)])
                y.append(test_reward)
                # print(f"At {global_step} steps, the test reward is ------- ", test_reward)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        prev_info = info['labels']
            else:
                prev_info = infos['labels'][0]

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue, logits = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                #print(mb_inds)
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #q buffer update
                mb_infos = [info_arr[i] for i in mb_inds]
                mb_prev_infos = [prev_info_arr[i] for i in mb_inds]
                mb_img_ctr = [image_counter[i] for i in mb_inds]
                mb_values = b_values[mb_inds]
                mb_q_values = mb_advantages + mb_values
                mb_states = b_obs[mb_inds]
                mb_actions = b_actions.long()[mb_inds]
                mb_teacher_probs = generate_teacher_logits(mb_states, mb_infos, mb_prev_infos, struggling_states, device)
                kickstarting_loss = -(logits * mb_teacher_probs).sum(dim=-1).mean()
                if kickstarting_loss>0:
                    state_seen_again_n +=1
                for idx in range(len(mb_states)):
                    list_tensor = mb_states[idx].tolist()
                    str_tensor = str(list_tensor)
                    if str_tensor not in local_q_buffer:
                        local_q_buffer[str_tensor]={"q-values":{0:0,1:0,2:0,3:0,4:0,5:0}}
                    q_value = mb_advantages[idx].tolist()+mb_values[idx].tolist()
                    #print(mb_actions[idx].tolist())
                    local_q_buffer[str_tensor]["q-values"][mb_actions[idx].tolist()]=q_value
                    local_q_buffer[str_tensor]["info"] = mb_infos[idx]
                    local_q_buffer[str_tensor]["prev_info"]= mb_prev_infos[idx]
                    local_q_buffer[str_tensor]["counter"] = mb_img_ctr[idx]
                    #print("^^^^^^^^^^^^",mb_states[idx][-1].shape)
                    q_network.update(mb_states[idx].unsqueeze(0),mb_actions[idx],q_value)
                # for state in states:
                #     print(state.shape)
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.kickstart_coeff * kickstarting_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/kickstarting_loss", kickstarting_loss, global_step)
        writer.add_scalar("charts/Contribution_of_KL",state_seen_again_n,global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # print("struggling_states_n: ",struggling_states_n)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    print(x)
    print(y)