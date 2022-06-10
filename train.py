from mimetypes import init
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import models as models



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Asynchronous_Simulator():
    def __init__(self, num_workers=3, batch_size = 4, model_name='small'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = batch_size

        # todo num_workers in the dataloader
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.num_workers = num_workers
        self.init_simulator(model_name)


    def init_simulator(self, model_name):
        
        Net = getattr(models, model_name)
        self.para_server = Net().to(self.device)
        workers = []

        for i in range(self.num_workers):
            worker = Net().to(self.device)
            worker.load_state_dict(self.para_server.state_dict())
            workers.append(worker)

        self.workers = workers

    def SGD(self, worker_idx, lr):
        server_dict = self.para_server.state_dict()
        """
        for param, grad in zip(self.para_server.parameters(),self.workers[worker_idx].parameters()):
                param.data.sub_(grad.grad * lr)
        
        for key, value, param in zip(server_dict.keys(), server_dict.values(), self.workers[worker_idx].parameters()):
            server_dict[key] = value - lr*param.grad
        """

        for name, param in self.workers[worker_idx].named_parameters():
            server_dict[name] = server_dict[name] - lr*param.grad

        self.para_server.load_state_dict(server_dict)# .to(self.device)
        self.workers[worker_idx] = copy.deepcopy(self.para_server)
        

    def train(self, max_epoch = 2, lr = 0.001):
        # for epoch in tqdm(range(max_epoch)):
            # dataiter = iter(self.trainloader)
        for epoch in range(max_epoch):
            running_loss = 0.0
            for i, data in tqdm(enumerate(self.trainloader, 0)):
  
                worker_dix = i%self.num_workers
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                worker = self.workers[worker_dix]
                outputs = worker(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                loss.backward()
                self.SGD(worker_idx=worker_dix, lr=lr)
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        
    def test(self,):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.para_server(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

