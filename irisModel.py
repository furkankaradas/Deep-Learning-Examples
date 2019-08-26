import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
inputSize = 4
hiddenSize = 4
outputSize = 3
learningRate = 0.01
epochNumber = 500

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(NeuralNetwork, self).__init__()
        self.fcl1 = nn.Linear(inputSize, hiddenSize)
        self.fcl2 = nn.Linear(hiddenSize, outputSize)
        self.relu = nn.LeakyReLU()

    def forward(self, value):
        out = self.relu(self.fcl1(value))
        return self.fcl2(out)

class model():
    def __init__(self):
        self.model = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = learningRate)

    def readDataset(self):
        self.dataset = pd.read_csv("./data/IRIS/IRIS.csv")
        self.dataset.loc[self.dataset['species'] == 'Iris-setosa', 'species'] = 0
        self.dataset.loc[self.dataset['species'] == 'Iris-versicolor', 'species'] = 1
        self.dataset.loc[self.dataset['species'] == 'Iris-virginica', 'species'] = 2

        self.x = self.dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        self.y = self.dataset['species'].values

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, test_size = 0.1)

    def train(self):
        self.readDataset()
        for epoch in range(epochNumber):
            x = Variable(torch.Tensor(self.xtrain).float()).to(device)
            y = Variable(torch.Tensor(self.ytrain).long()).to(device)

            output = self.model(x)
            loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if ((epoch + 1) % 100 == 0 or (epoch == 0)):
                print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, epochNumber, loss.item()))

    def test(self):
        with torch.no_grad():
            x = Variable(torch.Tensor(self.xtest).float()).to(device)
            y = Variable(torch.Tensor(self.ytest).long()).to(device)

            output = self.model(x)
            _, predict = torch.max(output.data, 1)
            correct = torch.sum(y == predict)
            print('Accuracy of the network: {} %'.format(100 * correct / len(self.xtest)))

    def workModel(self):
        self.train()
        self.test()

def main():
    mdl = model()
    mdl.workModel()

main()
