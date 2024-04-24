
import torch
import gymnasium as gym
from teacher_policy import *
from utils import *
from student.PPO import PPO as PPOAlgo1
import matplotlib.pyplot as plt
import ast

class Game:
    def __init__(self, game_id):
        self.env = gym.make(game_id, obs_type = "grayscale")
        self.student = None
        self.teacher = None
        self.total_timesteps = 1000
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def init_student_policy(self, num_inputs, num_outputs):
        #num_outputs = self.env.action_space
        #print(num_inputs.shape[0], num_outputs.n)
        # self.student = PPOAlgo1(self.env.observation_space.shape[0],4,256,
        #                         self.device,3e-4,self.env)
        self.student = PPOAlgo1(num_inputs, num_outputs, 256, self.device, 3e-4, self.env)
    
    def init_teacher(self):
        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        #NEED TO FIX PREFIX
        self.teacher = teacher_policy(model='gpt-3.5-turbo-1106', prefix="You are in a 3*3 minigrid world,\nState (2, 2) is the goal state, you get a reward of 10 points when you reach it.\nState (2, 0) is the water state, you lose 10 points when you reach it.\nState (1, 1) is a wall, you cannot enter it. If you reach a wall, you will not change your position.\nReaching any state other than the water and goal state has a reward of 0.\nHere are the choices of actions. 0: move up, 1: move down, 2: move left, and 3: move right. The goal is to reach to the goal state with the maximum points possible.")

    def return_most_imp_states(self, frame_idx):
        #loop through a quantized representation of the continuous states - bins of the states- as many as possible
        #[1|2]
        # - -
        #[3|4]
        # state is img of 210x160x255
        # binning - (0-5,160,255) (5-10,160,255)
        #take a uniform random sample from state space
        # how often to call? - do it less often 

        ############
        #create q_table for the states that you encountered
        #calculate q-value functions using network
        #might see some sort of pattern - that critical states are all around certain area in env
        # 
        q_table = self.student.q_table
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
            print(f"state {s} has range {rangeq} and visits {self.student.visits[s]}")
            if s not in self.student.struggling_states and sq<rangeq and self.student.visits[s]>(frame_idx/10):
                sq=rangeq
                struggling_state = s
        print(f"struggling state - {struggling_state}")
        self.student.struggling_states.add(s)
        return struggling_state
    
    def update_policy(self, teacher_input):
        pass

    # if action == 0:  # Up
    #                 dir = "up"
    #             elif action == 1:  # Down
    #                 dir = "down"
    #             elif action == 2:  # Left
    #                 dir = "left"
    #             elif action == 3:  # Right
    #                 dir = "right"
    def train(self):
        #self.student.train()
        tt=0
        self.student.max_frames = 1000
        while(tt<self.total_timesteps):
            n=self.student.train(tt)
            break
            print(self.student.visits)
            self.student.test_env()
            struggling_state = self.return_most_imp_states(tt)
            self.student.teacher_recommendations = {(0,0):3,(0,1):3,(0,2):1,(1,0):0,(1,1):3,(1,2):1,(2,0):3,(2,1):3}
            #{(1, 0): 0, (0, 2): 1, (1, 2): 1}#self.teacher.prompt([struggling_state])
            #self.update_policy(teacher_actions)
            tt+=1000
            self.student.max_frames+=1000
            self.contribution_of_kick_start_loss.append(n/tt)
            print(tt)
            #break
            #np.mean([self.student.test_env() for _ in range(10)])

breakout_id = "ALE/Breakout-v5"
game = Game(breakout_id)
game.init_student_policy(1,4)
game.init_teacher()
game.train()
print(len(game.student.local_q_buffer))
print(game.student.visits.values())
#print(game.student.data_points)
#print(game.contribution_of_kick_start_loss)

for img,val in game.student.visits.items():
    if val > 100:
        img_array = torch.tensor(ast.literal_eval(img)).squeeze(0,1).numpy()
        plt.imshow(img_array)
        plt.show()

