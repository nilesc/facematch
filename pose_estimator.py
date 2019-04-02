# Uses a model found here: https://github.com/natanielruiz/deep-head-pose
import torch
import torchvision
from deep_head_pose.code.hopenet import Hopenet

model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
model.load_state_dict(torch.load('pose_weights/hopenet_robust_alpha1.pkl', map_location='cpu'))
print(model.eval())
