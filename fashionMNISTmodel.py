import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data

# Hyper Parameters
inputSize = 28*28
hiddenSize = 500
outputSize = 10
learningRate = 0.001
batchSize = 50

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(NeuralNetwork, self).__init__()
        self.fcl1 = nn.Linear(inputSize, hiddenSize)
        self.fcl2 = nn.Linear(hiddenSize, hiddenSize)
        self.fcl3 = nn.Linear(hiddenSize, outputSize)
        self.relu = nn.ReLU()

    def forward(self, value):
        out = self.relu(self.fcl1(value))
        out = self.relu(self.fcl2(out))
        out = self.fcl3(out)
        return out

class model():
    def __init__(self):
        self.model = NeuralNetwork(inputSize, hiddenSize, outputSize)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learningRate)
        self.criterion = nn.CrossEntropyLoss()

    def loadDataset(self):
        transform = transforms.Compose([transforms.ToTensor(),])

        train = dataset.FashionMNIST(root = "./data", train = True, transform = transform, download = True)
        test = dataset.FashionMNIST(root = "./data", train = False, transform = transform, download = True)

        self.trainSet = data.DataLoader(dataset = train, batch_size = batchSize, shuffle = True)
        self.testSet = data.DataLoader(dataset = test, batch_size = batchSize, shuffle = False)
















