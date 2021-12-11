from torch import nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3))
        self.norm = nn.LayerNorm(normalized_shape=6)
        self.relu = nn.ReLU(inplace=True) #
        self.pool = nn.MaxPool2d(kernel_size=(2, 2)) #
        self.max = nn.Softmax(dim=1) #


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.max(out)
        return out

