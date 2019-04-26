# Uses a model found here: https://github.com/natanielruiz/deep-head-pose
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from deep_head_pose.code.hopenet import Hopenet


class PoseEstimator:

    def __init__(self, weights_path, gpu_available=False):
        map_location = 'cpu'
        if gpu_available:
            map_location = 'gpu'

        self.model = Hopenet(torchvision.models.resnet.Bottleneck,
                             [3, 4, 6, 3],
                             66)
        self.model.load_state_dict(torch.load(weights_path,
                                   map_location=map_location))

    def estimate_pose(self, input_image):
        # Convert from numpy to torch vector
        input_image = torch.from_numpy(input_image)

        # Move channels to be in pytorch format
        input_image = input_image.permute(0, 3, 1, 2)
        input_image = input_image.type('torch.FloatTensor')

        print(f'Input shape: {input_image.shape}')
        # Change dimensions to correct size
        desired_dim = 300

        # Interpolate to reduce dimensions
        scale_factor = desired_dim / max(input_image.shape[2], input_image.shape[3])
        print(scale_factor)
        input_image = F.interpolate(input_image,
                                    size=(int(input_image.shape[2] * scale_factor),
                                          int(input_image.shape[3] * scale_factor)))
        print(f'Interpolated: {input_image.shape}')

        # Increase dimensions
        pad_y = max(desired_dim - input_image.shape[2], 0)
        pad_x = max(desired_dim - input_image.shape[3], 0)
        pad_y = math.ceil(pad_y/2)
        pad_x = math.ceil(pad_x/2)
        input_image = torch.nn.ConstantPad2d((pad_x, pad_x, pad_y, pad_y), 0)(input_image)
        print(f'Padded: {input_image.shape}')

        yaw, pitch, roll = self.model(input_image.float())
        yaw = F.softmax(yaw, dim=1)
        pitch = F.softmax(pitch, dim=1)
        roll = F.softmax(roll, dim=1)

        outputs = torch.stack((yaw, pitch, roll), 1)

        index_tensor = torch.FloatTensor([index for index in range(66)])
        index_tensor = index_tensor.view(1, 1, -1)
        index_tensor = index_tensor.repeat(input_image.shape[0], 3, 1)

        outputs = torch.sum(outputs * index_tensor, 2) * 3 - 99

        return outputs.data.numpy()


if __name__ == '__main__':
    pe = PoseEstimator('pose_weights/hopenet_robust_alpha1.pkl',
                       gpu_available=False)
    random_inputs = torch.from_numpy(np.random.rand(10, 3, 200, 200))
    outputs = pe.estimate_pose(random_inputs)
    print(outputs)
