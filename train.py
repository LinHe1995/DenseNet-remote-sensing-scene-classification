#from dense_model import *
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from torch.backends import cudnn
from dense201pro import densenet201 as densenet

plt_loss = []
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

learnrate = 1e-3
numepoch = 1000

#print(torch.cuda.is_available())

def train(epoch, model, lossFunction, optimizer, device, trainloader):
    print('\nEpoch: %d' % epoch)
    model.train()  # enter train mode
    train_loss = 0  # accumulate every batch loss in a epoch
    correct = 0  # count when model's prediction is correct in train set
    total = 0  # total number of prediction in train set
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # For example
        # 3 classes, each class contains 70 samples
        # 21 iterations, each iteration contains 10 samples when batch_size=10
        inputs, targets = inputs.to(device), targets.to(device)  # load data to gpu device
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()  # clear gradients of all optimized torch.Tensors'
        outputs = model(inputs)  # forward propagation return the value of softmax function
        loss = lossFunction(outputs, targets)  # compute loss
        loss.backward()  # compute gradient of loss over parameters
        optimizer.step()  # update parameters with gradient descent

        train_loss += loss.item()  # accumulate every batch loss in a epoch
        _, predicted = outputs.max(1)  # make prediction according to the outputs
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()  # count how many predictions is correct

        # if (batch_idx + 1) % 100 == 0:
        #     # print loss and acc
        #     print('***Train loss: %.3f | Train Acc: %.3f%% (%d/%d)'
        #           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    plt_loss.append(train_loss)
    print('Train loss: %.3f | Train Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def eval(model, lossFunction, optimizer, device, testloader):
    global best_acc
    model.eval()  # enter test mode
    test_loss = 0  # accumulate every batch loss in a epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = lossFunction(outputs, targets)  # compute loss

            test_loss += loss.item()  # accumulate every batch loss in a epoch
            _, predicted = outputs.max(1)  # make prediction according to the outputs
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # count how many predictions is correct
        # print loss and acc
        print('Test Loss: %.3f  | Test Acc: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def data_loader():
    # define method of preprocessing data for evaluating
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # Normalize a tensor image with mean and standard variance
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard variance
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # prepare dataset by ImageFolder, data should be classified by directory
    trainset = torchvision.datasets.ImageFolder(root='./dataset/AID/train', transform=transform_train)

    testset = torchvision.datasets.ImageFolder(root='./dataset/AID/test', transform=transform_test)

    # Data loader.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
    return trainloader, testloader

def run(model, num_epochs):
    # load model into GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model.to(device)
    #     if device == 'cuda:0':
    #         model = torch.nn.DataParallel(model)
    #         cudnn.benchmark = True

    # define the loss function and optimizer

    lossFunction = nn.CrossEntropyLoss()
    lr = learnrate
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    trainloader, testloader = data_loader()
    for epoch in range(num_epochs):
        train(epoch, model, lossFunction, optimizer, device, trainloader)
        eval(model, lossFunction, optimizer, device, testloader)
        if (epoch + 1) % 50 == 0:
            lr = lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    fig = plt.figure()
    plt.plot(plt_loss, c='r')
    plt.show()

if __name__ == '__main__':

    model = densenet(pretrained=True)

    run(model, num_epochs=numepoch)
    modelsave = "./logs/model_weights_densenetpro_AID_"+str(learnrate)+"_"+str(numepoch)+".pth"
    torch.save(obj=model.state_dict(), f=modelsave)
