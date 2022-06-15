from mimetypes import init
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import models as models



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
        self.momentum = None

    def init_simulator(self, model_name):
        
        Net = getattr(models, model_name)
        self.para_server = Net().to(self.device)
        workers = []

        for i in range(self.num_workers):
            worker = Net().to(self.device)
            worker.load_state_dict(self.para_server.state_dict())
            workers.append(worker)

        self.workers = workers

    def SGD(self, worker_idx, lr, decay=False, epoch=1, decay_rate = 2, momentum=0, dampening = 0):
        server_dict = self.para_server.state_dict()
        """
        for param, grad in zip(self.para_server.parameters(),self.workers[worker_idx].parameters()):
                param.data.sub_(grad.grad * lr)
        
        for key, value, param in zip(server_dict.keys(), server_dict.values(), self.workers[worker_idx].parameters()):
            server_dict[key] = value - lr*param.grad
        """
        # lr_list = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005] # for regular sgd and dropout
        # lr_list = [0.001, 0.001, 0.0005, 0.0005, 0.0002, 0.0001] # for momentum
        lr = lr_list[epoch]
        if decay:
            lr = lr/decay_rate**epoch
        if momentum:
            if self.momentum is None:
                momentum_dict = {}
                for name, param in self.workers[worker_idx].named_parameters():
                    momentum_dict[name] = param.grad
                self.momentum = momentum_dict
            else:
                for name, param in self.workers[worker_idx].named_parameters():
                    self.momentum[name] = momentum * self.momentum[name] + (1-dampening)*param.grad

        for name, param in self.workers[worker_idx].named_parameters():
            if not momentum:
                server_dict[name] = server_dict[name] - lr * param.grad
            else:
                server_dict[name] = server_dict[name] - lr*self.momentum[name]

        self.para_server.load_state_dict(server_dict)# .to(self.device)
        self.workers[worker_idx] = copy.deepcopy(self.para_server)
        

    def train(self, max_epoch = 6, lr = 0.001, method='SGD', decay_rate = 0, momentum=0, dampening = 0):
        # for epoch in tqdm(range(max_epoch)):
            # dataiter = iter(self.trainloader)
        loss_list = []
        acc_list = []
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
                self.SGD(worker_idx=worker_dix, lr=lr, epoch=epoch, momentum=momentum)
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    loss_list.append(running_loss/2000)
                    running_loss = 0.0
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    acc = self.test()
                    acc_list.append(acc)
        return loss_list, acc_list
        
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
            acc = correct/total
            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
        return acc