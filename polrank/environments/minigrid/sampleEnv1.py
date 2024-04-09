import numpy as np
import gym
import gymnasium as gym
from gymnasium import spaces
import pygame

class MazeGameEnv(gym.Env):
    def __init__(self):
        maze = [
                ['S', '.', '.'],
                ['.', 'B', '.'],
                ['W', '.', 'G'],
                ]
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)  # Maze represented as a 2D numpy array
        self.start_pos = (0,0)  # Starting position
        self.goal_pos = (2,2)  # Goal position
        self.water_pos = (2,0)
        self.wall_pos = (1,1)
        self.current_pos = self.start_pos #starting position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape
        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)  

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(2,), dtype=np.float32
        )#spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize Pygame
        # pygame.init()
        self.cell_size = 125

        # setting display size
        # self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self):
        self.current_pos = self.start_pos
        return_value = np.array(self.current_pos).astype(np.float32)
        #print(return_value)
        return return_value, {}  #7*3*3  action*width*high

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = list(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = tuple(new_pos)

        reward = 0
        done = False
        # Reward function
        if self.current_pos==self.goal_pos:
            reward = 10.0
            done = True
        elif self.current_pos==self.water_pos:
            reward = -10.0
            done = False
        return_value = np.array(self.current_pos).astype(np.float32)
        #(return_value)
        return return_value, reward, done, False, {}

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols or (row,col)==self.wall_pos:
            return False

        return True

    # def render(self):
    #     # Clear the screen
    #     self.screen.fill((255, 255, 255))  

    #     # Draw env elements one cell at a time
    #     for row in range(self.num_rows):
    #         for col in range(self.num_cols):
    #             cell_left = col * self.cell_size
    #             cell_top = row * self.cell_size
            
    #             try:
    #                 print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
    #             except Exception as e:
    #                 print('Initial state')

    #             if self.maze[row, col] == 'W':  # Obstacle
    #                 pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
    #             elif self.maze[row, col] == 'S':  # Starting position
    #                 pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
    #             elif self.maze[row, col] == 'G':  # Goal position
    #                 pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

    #             if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1,1)):  # Agent position
    #                 pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

    #     pygame.display.update()  # Update the display