import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
# model = TheModelClass()
#
# # Initialize optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

x = torch.tensor([
    [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]
])
print("xshape():", x.size())
print("x.size():", x.size())
x.reshape([1, 12])
print(x.reshape([1,12]))
print(x.reshape([1,12]).shape)

print("after squeeze x: ", x.reshape ([1, 12]).squeeze())
print("after unsqueeze x: ", x.reshape ([1, 12]).squeeze().unsqueeze(dim = 0))

#cat
x = (torch.rand(2, 3, 4)*100).int()
print(x)
y = (torch.rand(2, 3, 4)*100).int()
print(y)
z_zero = torch.cat((x,y), 0)
z_one = torch.cat((x,y), 1)
z_two = torch.cat((x,y), 2)

print(z_zero.size())
print(z_one.size())
print(z_two.size())

#torch.clamp

boxes0 = torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]])
print("boxes: ", boxes0[..., :2])#the first two element
print("the last 2 elements of boxes: ", boxes0[..., 2:])

gt_boxes = torch.tensor([[0.5567, 0.2560, 0.6072, 0.3018],
        [0.7032, 0.2599, 0.7394, 0.2995],
        [0.6683, 0.2599, 0.7026, 0.3297]])


gt_boxes_unsqueeze0 = gt_boxes.unsqueeze(0)
gt_boxes_unsqueeze1 = gt_boxes.unsqueeze(1)

print("gt_boxes_unsqueeze0: ", gt_boxes_unsqueeze0)
print("gt_boxes_unsqueeze0: size", gt_boxes_unsqueeze0.size())
print("gt_boxes_unsqueeze1: ", gt_boxes_unsqueeze1)
print("gt_boxes_unsqueeze1: size", gt_boxes_unsqueeze1.size())

locations = torch.tensor([[0.5567, 0.2560, 0.6072, 0.3018],
        [0.7032, 0.2599, 0.7394, 0.2995],
        [0.6683, 0.2599, 0.7026, 0.3297]])
print(locations.size())

k_per_priors = [4, 6, 6, 6, 4, 3]
priors_area = [38*38, 19*19, 10*10, 5*5, 3*3, 1*1]
priors_area_ssd = [35*35, 17*17, 9*9, 5*5, 3*3, 1*1]
num_priors = np.dot (k_per_priors, priors_area_ssd)
#num_priors = (35*35*4 + 17*17*6 + 9*9*6 + 5*5*6 + 3*3*4 + 1*1*4)*4
print("num_priors: ", num_priors)

#confidence (batch_size, num_priors, num_classes)
confidence = torch.tensor([[[10, 9, 8],[11, 8, 7],[2, 18, 15],[11, 1, 12], [12, 14, 15]],
                           [[1, 2, 3],[1, 3, 4],[12, 1, 1], [1, 4, 1], [2, 1, 1]],
                            [[1, 3, 5],[1, 3, 2],[1, 1, 1], [1, 14, 2], [1, 1, 1]]])

#loss = -F.log_softmax(confidence, dim=2)[:, :, 0]

