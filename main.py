from stable_baselines3.common.env_util import make_vec_env
from student_policy import *
from teacher_policy import *
from sampleEnv import MazeGameEnv

class Game:
    def __init__(self, maze,):
        self.env = make_vec_env(MazeGameEnv, n_envs=1, env_kwargs={'maze':maze})
        self.total_timesteps = 2e5
        self.student = None
        self.teacher=None
    
    def init_student_policy(self):
        self.student = PPOAlgo(self.total_timesteps, self.env)
        self.student.init_model()
    
    def init_teacher(self):
        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.teacher = teacher_policy(model='gpt-3.5-turbo-1106', prefix="You are in a 3*3 minigrid world,\nState (2, 2) is the goal state, you get a reward of 10 points when you reach it.\nState (2, 0) is the water state, you lose 10 points when you reach it.\nState (1, 1) is a wall, you cannot enter it. If you reach a wall, you will not change your position.\nReaching any state other than the water and goal state has a reward of 0.\nHere are the choices of actions. 0: move up, 1: move down, 2: move left, and 3: move right. The goal is to reach to the goal state with the maximum points possible.")

    def return_most_imp_states(self):
        return [(0,0),(1,0)]
    
    def update_policy(self, teacher_input):
        pass

    def train(self):
        tt=1000
        while(tt<self.total_timesteps):
            self.student.train(tt)
            self.student.test()
            struggling_states = self.return_most_imp_states()
            teacher_actions = self.teacher.prompt(struggling_states)
            self.update_policy(teacher_actions)
            tt+=1000
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
