import numpy as np
import torchvision.transforms.functional as TF

def preprocess_image(image):
    breakout_inputdim = (1, 82, 72)
    image = image[32:196]  # crop
    copy = np.transpose(image, (1,0))
    image = copy[8:152]
    image = np.transpose(image, (1,0))
    image = image[::2, ::2]
    tens = TF.to_tensor(image)
    tens = tens.unsqueeze(0)
    return tens, np.reshape(image, breakout_inputdim)