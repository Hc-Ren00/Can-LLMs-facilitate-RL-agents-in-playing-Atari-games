
import torch
import gym
from teacher_policy import *
from utils import *
from student.PPO import PPO as PPOAlgo1
import matplotlib.pyplot as plt
from atariari.benchmark.wrapper import AtariARIWrapper
import ast

class Game:
    def __init__(self, game_id):
        self.env = AtariARIWrapper(gym.make(game_id))#, obs_type = "grayscale")
        self.student = None
        self.teacher = None
        self.total_timesteps = 1000000
        self.x=[]
        self.y=[]
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def init_student_policy(self, num_inputs, num_outputs):
        #num_outputs = self.env.action_space
        #print(num_inputs.shape[0], num_outputs.n)
        # self.student = PPOAlgo1(self.env.observation_space.shape[0],4,256,
        #                         self.device,3e-4,self.env)
        self.student = PPOAlgo1(num_inputs, num_outputs, 256, self.device, 3e-4, self.env)
    
    def init_teacher(self):
        self.teacher = teacher_policy(model='gpt-3.5-turbo-1106', prefix="I'm playing a game of Atari pong using the openai gym environment which represents the game as a grid of (210,160).")

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
        q_table = self.student.local_q_buffer
        struggling_state=None
        struggling_qvalues=None
        sq=float('-inf')
        for s in q_table:
            # maxq=0
            # minq=0
            # unexplored=False
            # for a in q_table[s]:
            #     q_value = q_table[s][a]
            #     if q_value==0:
            #         unexplored=True
            #         break
            #     else:
            #         minq=min(minq,q_value)
            #         maxq=max(maxq,q_value)
            fstate = torch.tensor(ast.literal_eval(s)).unsqueeze(0).to(self.device)
            q_values = self.student.q_network.model(fstate)
            # if unexplored==True:
            #     print(f"skipping state {s} due to it being unexplored")
            #     continue
            rangeq=torch.max(q_values)-torch.min(q_values).to(self.device)
            #print(f"state {s} has range {rangeq} and visits {self.student.visits[s]}")
            if s not in self.student.struggling_states and sq<rangeq:# and self.student.visits[s]>(frame_idx/10):
                sq=rangeq
                struggling_qvalues=q_values
                struggling_state = s
        #print(f"struggling state - {sq}, struggling q_values - {struggling_qvalues}, info - {q_table[struggling_state]['info']}, prev_info - {q_table[struggling_state]['prev_info']}")
        #self.student.struggling_states.add((q_table[struggling_state]['info'],q_table[struggling_state]['prev_info']))
        si = q_table[struggling_state]['info']
        spi = q_table[struggling_state]['prev_info']
        print("*******Collecting 1 important state**********")
        return (struggling_state,si,spi)
    
    def update_policy(self, teacher_input):
        pass

    def train(self):
        tt=0
        self.student.max_frames = 1000
        print("========Training the student for 1000 steps============")
        self.student.train(tt)
        struggling_state, curr_info, prev_info = self.return_most_imp_states(tt)
        rec_action = self.teacher.prompt([(curr_info, prev_info)])
        self.student.teacher_recommendations = {struggling_state: rec_action}
        tt+=1000
        self.student.max_frames+=200
        self.student.local_q_buffer = {}
        last_info = ''
        while(tt<self.total_timesteps):
            self.x.append(tt)
            self.y.append(np.mean([self.student.test_env() for _ in range(5)]))
            last_info=self.student.train(tt,last_info)
            print(f"========Training the student for {self.student.max_frames} steps============")
            struggling_state, curr_info, prev_info = self.return_most_imp_states(tt)
            self.student.local_q_buffer = {}
            #print("Querying the teacher --- ")
            rec_action = self.teacher.prompt([(curr_info, prev_info)])
            key = (struggling_state,str(curr_info)+"#"+str(prev_info))
            if key not in self.student.teacher_recommendations:
                self.student.teacher_recommendations[key] = rec_action
            #{(0,0):3,(0,1):3,(0,2):1,(1,0):0,(1,1):3,(1,2):1,(2,0):3,(2,1):3}
            #{(1, 0): 0, (0, 2): 1, (1, 2): 1}#self.teacher.prompt([struggling_state])
            #self.update_policy(teacher_actions)
            tt+=200
            self.student.max_frames+=200
            #self.contribution_of_kick_start_loss.append(n/tt)
            #print(tt)
            #break
            if tt%1000==0:
                print(f"----------------TESTING AFTER {tt} STEPS----------------")
                reward_arr = []
                for _ in range(5):
                    reward = self.student.test_env()
                    #print(f"Iteration {_}, reward: {reward}")
                    reward_arr.append(reward)
                print("Average - ", np.mean([self.student.test_env() for _ in range(5)]))

breakout_id = "Pong-v4"
game = Game(breakout_id)
game.init_student_policy(1,6)
game.init_teacher()
game.train()
print("Episodes -- ",game.student.episodes)
print("teacher recommendations - ",len(game.student.teacher_recommendations))
print(game.x,game.y)
#print(game.student.visits.values())
#print(game.student.data_points)
#print(game.contribution_of_kick_start_loss)

# for img in game.student.struggling_states:
#         img_array = torch.tensor(ast.literal_eval(img)).squeeze(0,1).numpy()
#         plt.imshow(img_array)
#         plt.show()

