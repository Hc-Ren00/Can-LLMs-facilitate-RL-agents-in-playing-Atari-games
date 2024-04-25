import numpy as np
import torchvision.transforms.functional as TF

def preprocess_image_breakout(image):
    breakout_inputdim = (1, 82, 72)
    image = image[32:196]  # crop
    copy = np.transpose(image, (1,0))
    image = copy[8:152]
    image = np.transpose(image, (1,0))
    image = image[::2, ::2]
    tens = TF.to_tensor(image)
    tens = tens.unsqueeze(0)
    return tens, np.reshape(image, breakout_inputdim)

def preprocess_image_pong(image):
    pong_inputdim = (1, 80, 80)
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    tens = TF.to_tensor(image)
    tens = tens.unsqueeze(0)
    return tens, np.reshape(image, pong_inputdim)