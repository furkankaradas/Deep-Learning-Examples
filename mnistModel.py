import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transform
import torchvision.datasets as dataset

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

#Hyper Parameters
inputSize = 28*28
hiddenSize = 600
outputSize = 10
batchSize = 70
learningRate = 0.001
epochNumber = 25

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(NeuralNetwork, self).__init__()
        self.fcl1 = nn.Linear(inputSize, hiddenSize)
        self.fcl2 = nn.Linear(hiddenSize, hiddenSize)
        self.fcl3 = nn.Linear(hiddenSize, hiddenSize)
        self.fcl4 = nn.Linear(hiddenSize, outputSize)
        self.reluFunction = nn.LeakyReLU()

    def forward(self, value):
        out = self.fcl1(value)
        out = self.reluFunction(out)
        out = self.fcl2(out)
        out = self.reluFunction(out)
        out = self.fcl3(out)
        out = self.reluFunction(out)
        out = self.fcl4(out)
        return out

class model():
    def __init__(self):
        self.model = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = learningRate)

    def loadTrainDataset(self):
        transforms = transform.Compose([transform.ToTensor(),transform.Normalize((0.5,),(0.5,)),])

        self.trainSet = dataset.MNIST(root = './data', train = True, transform = transforms, download = True)
        self.trainLoader = data.DataLoader(dataset = self.trainSet, batch_size = batchSize, shuffle = True)

    def loadTestDataset(self):
        transforms = transform.Compose([transform.ToTensor(),transform.Normalize((0.5,),(0.5,)),])

        self.testSet = dataset.MNIST(root = './data', train = False, transform = transforms)
        self.testLoader = torch.utils.data.DataLoader(dataset = self.testSet, batch_size = batchSize, shuffle = False)

    def trainModel(self):
        print("Train Started!")
        self.loadTrainDataset()
        for epoch in range(epochNumber):
            for i, (data, target) in enumerate(self.trainLoader):
                data = data.reshape(-1, 28*28).to(device)
                target = target.to(device)

                # Forward Pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward Pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if ((i + 1) % 100 == 0):
                    print("Epoch [{}/{}], Step [{}/{}], Loss [{:.4f}]".format(epoch + 1, epochNumber, i + 1, len(self.trainLoader), loss.item()))
        print("Train Successfully Completed!")

    def testModel(self):
        print("Test Started!")
        self.loadTestDataset()
        with torch.no_grad():
            correct, total = 0, 0
            for data, target in (self.testLoader):
                data = data.reshape(-1, 28*28).to(device)
                target = target.to(device)

                output = self.model(data)

                _, prediction = torch.max(output.data, 1)
                total +=  target.size(0)

                correct += (prediction == target).sum().item()
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        print("Test Successfully Completed!")

    def workModel(self):
        self.trainModel()
        self.testModel()

def startModel():
    mdl = model()
    mdl.workModel()

def main():
    startModel()

main()