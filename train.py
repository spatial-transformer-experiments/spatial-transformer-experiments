# -*- coding: utf-8 -*-
# License: BSD
# Based on Code of: Ghassen Hamrouni, https://github.com/GHamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models import Net, Net_CoordConv
from confusion_matrix import ConfusionMatrix
from six.moves import urllib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import json
import os
import pathlib
# fix random seeds for reproducibility (https://github.com/victoresque/pytorch-template/blob/master/train.py)
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

def train(model,experiment_name):
    #create experiment dir
    pathlib.Path(f"experiments/{experiment_name}").mkdir(exist_ok=experiment_name=="test")
    


    ######################################################################
    # Loading the data
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)

    ######################################################################
    # Training the model
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, 20 + 1):
        train_epoch(model=model,epoch=epoch,train_loader=train_loader,optimizer=optimizer)
        test(model=model,test_loader=test_loader)

    ######################################################################
    # Visualize the STN transformation on some input batch
    fig = visualize_stn(model=model,test_loader=test_loader)
    fig.savefig(f"experiments/{experiment_name}/visualized_transformation.png")

    ######################################################################
    #Compute confusion matrix and log to experiment dir
    test_final(model=model,test_loader=test_loader,experiment_name=experiment_name)

def train_epoch(model,epoch,train_loader,optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model,test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

def test_final(model,test_loader,experiment_name):
    confusion_matrix = ConfusionMatrix()

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            confusion_matrix.add_batch(pred=pred.view_as(target),target=target)

        #fig = confusion_matrix.plot_confusion_matrix(save_path=f"experiments/{experiment_name}/confusion_matrix.png",title=experiment_name)   
        fig = confusion_matrix.plot_confusion_matrix(title=experiment_name)
        fig.savefig(f"experiments/{experiment_name}/confusion_matrix.png")  

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
            .format(test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

        result_dict = {
            "experiment_name":experiment_name,
            "test_loss":test_loss,
            "test_accuracy": 100. * correct / len(test_loader.dataset),
            "test_correct": correct,
            "test_total":len(test_loader.dataset)
            }
        json.dump(result_dict, open(f"experiments/{experiment_name}/results.json","w"))

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(model,test_loader):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
        return f



if __name__=="__main__":
    model=Net_CoordConv(bypass_localisation=True)
    train(model=model,experiment_name="test")