from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Dataset import transforms
from NeuralNet import NeuralNet
import torch.utils.data
import torchvision
from torchvision import transforms
import torch.optim as optim
import tqdm
import splitfolders

#data_folder = 'C:/Users/Kate/Documents/datasets/cifar-10/'
#train_path = data_folder + 'train'
#test_path = data_folder + 'test'

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = NeuralNet()
    loss_f = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,  momentum=0.9)


    epochs = 3

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for i, batch in enumerate(trainloader,0):
             optimizer.zero_grad()
             input, target = batch
             #print(batch[0].shape, batch[1].shape)
             output = model(input)
             #print(output.shape, target.shape)
             loss = loss_f(output, target)
             loss.backward()
             optimizer.step()
             train_loss += loss.item()

             if i % 2000 == 1999:  # print every 2000 mini-batches
                 print('[%d, %5d] loss: %.3f' %
                     (epoch + 1, i + 1, train_loss / 2000))
                 train_loss = 0.0
        #train_loss /= len(train_loss)


    print("Triaining done!")