# Uses a model found here: https://github.com/natanielruiz/deep-head-pose
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
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
        desired_dim = 224
        self.transform_input = transforms.Compose([
            transforms.Resize(desired_dim),
            transforms.CenterCrop(desired_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def estimate_pose(self, input_image):
        input_image = self.transform_input(input_image)

        # Convert to batch
        input_image = input_image.view(1,
                                       input_image.shape[0],
                                       input_image.shape[1],
                                       input_image.shape[2])

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
