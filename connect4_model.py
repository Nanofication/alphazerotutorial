from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Connect4Model(nn.Module):
    def __init__(self, device): # initialize all layers of network
        # Device is CPU or GPU to train
        super().__init__()
        self.device = device
        # Define the layers themselves

        # convolution layer
        # 3 Boards to the network, one from player 1's player, tell where empty pieces are, one from opponent
        # Out_channels = Arbitrary choice but defining the size of the network, the paper used 128 which is best results
        # Kernel_Size = filter size 3x3. Small window pan of the board and check what's going on. 3x3 view of the board
        # Stride = how many spaces to move over while going through convolution. How to view
        # Dilation
        # Padding - if you add 1, you increase the board size by 1, then reduce it back down not needed here
        # Bias - used as True
        # Padding_Mode - As the filter passes over the board, we want to ensure board stays the same size. Make sure data isn't getting smaller
        self.initial_conv = nn.Conv2d(3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        # Batch Norm 2d - doesn't change shape at all

        self.initial_bn = nn.BatchNorm2d(128)


        # Res Block 1
        # Padding maintains the size
        # Block of layers, you can have any number. Input is equal to ouput
        self.res1_conv1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(128)
        self.res1_conv2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(128)

        # Res Block 2
        # In practice you can have multiple 7 to 8 layers
        # Define layers, don't do it like this. There's a conventional way, the below is the way to understand
        self.res2_conv1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(128)

        # Value Head
        # If player lose -1, If Win + 1
        self.value_conv = nn.Conv2d(128, out_channels=3, kernel_size=1, stride=1, bias=True)
        self.value_bn = nn.BatchNorm2d(3) # Specify input for each output. Normal board
        self.value_fc = nn.Linear(3*6*7, 32) # How many neurons in the layer. We're expecting 32 outputs
        self.value_head = nn.Linear(32,1) # 1 Neuron in that final value head

        # Policy Head
        # Gives you actions from 0 to 6, tells you which action to best take
        self.policy_conv = nn.Conv2d(128, out_channels=32, kernel_size=1, stride=1, bias=True) # Want to output 7 neurons because that's the action for the board
        self.value_bn = nn.BatchNorm2d(32)
        self.policy_head = nn.Linear(32 * 6 * 7, 7) # 7 outputs
        self.policy_ls = nn.LogSoftmax(dim=1) # Use softmax. Which axis of array we're computing softmax on

        self.to(device=device)

    if __name__ == '__main__':
        def forward(self, x):
            # Define the connections between the layers
            # Reshape the data x will be shaped (3,6,7) - 3 channels, 6 rows, 7 columns

            # Add dimension for batch size
            x = x.view(-1, 3, 6, 7)
            x = self.initial_bn(self.initial_conv(x)) # Pass first few parts of network
            x = F.relu(x) # Pass through activation function. Relu is commonly used Check tensorspace.org. good wayto view data

            # Res Block 1
            res = x
            x = F.relu(self.res1_bn1(self.res1_conv1(x)))
            x = F.relu(self.res1_bn2(self.res1_conv2(x)))
            x+= res # This is having residual info added. It's in the paper
            x = F.relu(x)

            # Res Block 2
            res = x
            x = F.relu(self.res2_bn1(self.res2_conv1(x)))
            x = F.relu(self.res2_bn2(self.res2_conv2(x)))
            x+= res # This is having residual info added. It's in the paper
            x = F.relu(x)

            # Innovation in Alpha Zero vs Alpha Go - by combining the Neural Network - Network needs to distill the same info
            # Having information passed is the same
            # Capitalizing that the agent can make both policy and action judgement

            # value head
            v = F.relu(self.value_bn(self.value_conv(x)))
            v = v.view(-1, 3*6*7) # Reshape to 1 for each previous layer
            v = F.relu(self.value_fc(v))
            v = F.tanh(v) # Hyperbolic tangeant function

            # policy head
            p = F.relu(self.policy_bn(self.policy_conv(x)))
            p = p.view(-1, 3 * 6 * 7)  # Reshape to 1 for each previous layer
            p = F.relu(self.value_fc(p))
            p = self.policy_ls(p)

            return v, p

            # V is -1 - 1 P is 7 value array indication probabilities from 0 to 1
            # Highest P is action we will take

if __name__ == '__main__':
    # print(torch.zeros(1).cuda())

    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = Connect4Model(device) # initiate model
    architecture_summary = summary(model, input_size=(16, 3, 6, 7), verbose=0)

    print(architecture_summary)
