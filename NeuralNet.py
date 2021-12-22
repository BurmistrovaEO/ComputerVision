from torch import nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))
        self.norm = nn.BatchNorm2d(5) #4 ?
        self.act = nn.ReLU(inplace=True) #
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.max = nn.Softmax(dim=0) #

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        #out = self.relu(out)

        out = self.pool(out)

        out = out.view(out.size(0), -1)
        out = self.max(out)
        return out
