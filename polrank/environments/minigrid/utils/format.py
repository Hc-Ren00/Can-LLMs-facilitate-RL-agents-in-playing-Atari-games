import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space):
    def preprocess_obss(obss, device=None):
        return obss
    
    return obs_space, preprocess_obss
