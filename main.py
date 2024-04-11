from stable_baselines3.common.env_util import make_vec_env
from student_policy import *
from teacher_policy import *
from algos.PPO import PPO as PPOAlgo1
from sampleEnv import MazeGameEnv

class Game:
    def __init__(self, maze,):
        self.env = make_vec_env(MazeGameEnv, n_envs=1, env_kwargs={'maze':maze})
        self.total_timesteps = 2e5
        self.student = None
        self.teacher=None
        self.struggling_states=set()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def init_student_policy(self):
        # self.student = PPOAlgo(self.total_timesteps, self.env)
        # self.student.init_model()
        num_inputs  = self.env.observation_space
        num_outputs = self.env.action_space
        #print(num_inputs.shape[0], num_outputs.n)
        self.student = PPOAlgo1(self.env.observation_space.shape[0],4,256,
                                self.device,3e-4,self.env)
    
    def init_teacher(self):
        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.teacher = teacher_policy(model='gpt-3.5-turbo-1106', prefix="You are in a 3*3 minigrid world,\nState (2, 2) is the goal state, you get a reward of 10 points when you reach it.\nState (2, 0) is the water state, you lose 10 points when you reach it.\nState (1, 1) is a wall, you cannot enter it. If you reach a wall, you will not change your position.\nReaching any state other than the water and goal state has a reward of 0.\nHere are the choices of actions. 0: move up, 1: move down, 2: move left, and 3: move right. The goal is to reach to the goal state with the maximum points possible.")

    def return_most_imp_states(self, q_table):
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
            if s not in self.struggling_states and sq<rangeq and visits[s]>(frame_idx/10):
                sq=rangeq
                struggling_state = s
        print(f"struggling state - {struggling_state}")
        self.struggling_states.add(struggling_state)
        return struggling_state
    
    def update_policy(self, teacher_input):
        pass

    def train(self):
        #self.student.train()
        tt=0
        self.student.max_frames = 1000
        while(tt<self.total_timesteps):
            self.student.train(tt)
            self.student.test()
            struggling_state = self.return_most_imp_states()
            #teacher_actions = self.teacher.prompt(struggling_state)
            #self.update_policy(teacher_actions)
            tt+=1000
            self.student.max_frames+=1000
            print(tt)
            break

maze = [
    ['S', '.', '.'],
    ['.', 'B', '.'],
    ['W', '.', 'G'],
]
game = Game(maze)
game.init_student_policy()
game.init_teacher()
game.train()

