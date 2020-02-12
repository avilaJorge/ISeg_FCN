#!/usr/bin/env python
# coding: utf-8

from torchvision import utils
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Change name to FCN to use this model instead I think

class FCN_bak(torch.nn.Module):

    def __init__(self, n_class):
        super(FCN_bak, self).__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=(3,5), stride=(2,4), padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.deconv5  = nn.ConvTranspose2d(32, 3, kernel_size=(3, 5), stride=(2,4), padding=1, dilation=1, output_padding=1)
        self.bn5= nn.BatchNorm2d(3)
        self.classifier = nn.Conv2d(3,n_class, kernel_size=1, stride=1, padding=0, dilation=1)
        
    def forward(self, x):
        pool = nn.MaxPool2d(2, stride=2,return_indices = True)
        unpool = nn.MaxUnpool2d(2, stride=2)
        
        x1, indice1 = pool(self.relu(self.conv1(x)))
        x2, indice2 = pool(self.relu(self.conv2(self.bnd1(x1))))
        x3, indice3 = pool(self.relu(self.conv3(self.bnd2(x2))))
        x4, indice4 = pool(self.relu(self.conv4(self.bnd3(x3))))
        x5, indice5 = pool(self.relu(self.conv5(self.bnd4(x4))))
        
        z1 = self.deconv1(self.bnd5(self.relu(unpool((x5), indice5))))
        z2 = self.deconv2(self.bn1(self.relu(unpool((z1), indice4))))
        z3 = self.deconv3(self.bn2(self.relu(unpool((z2), indice3))))
        z4 = self.deconv4(self.bn3(self.relu(unpool((z3), indice2))))
        z5 = self.deconv5(self.bn4(self.relu(unpool((z4), indice1))))
        
        out_decoder = self.classifier(self.bn5(z5))                  

        return out_decoder  # size=(N, n_class, x.H/1, x.W/1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def print_GPU_stats():
    print("total GPU Mem: ", torch.cuda.get_device_properties(device).total_memory)
    print("total GPU Cached: ", torch.cuda.memory_cached(device))
    print("total GPU Allocated: ", torch.cuda.memory_allocated(device))
    print("Available GB: ", (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device))/(10**9))
print_GPU_stats()


batch_size = 5
num_wrkrs = 4
train_dataset = CityScapesDataset(csv_file='train.csv')
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=num_wrkrs,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_size,
                          num_workers=num_wrkrs,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          num_workers=num_wrkrs,
                          shuffle=True)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.xavier_uniform(m.bias.data)
        m.bias.data.zero_()

epochs     = 100
start_epoch = 9
criterion = torch.nn.CrossEntropyLoss()
fcn_model = FCN_bak(n_class=34)
# fcn_model.apply(init_weights)
fcn_model.load_state_dict(torch.load('model_02_12_01_16.pt'))
optimizer = optim.Adam(fcn_model.parameters(), lr=5e-1)

dt = datetime.now().strftime("%m_%d_%H_%M")
output_fn = "model_output_" + dt + ".txt"
best_model_fn = "best_model_" + dt + ".pt"
model_fn = "model_" + dt + ".pt"

def print_info(out_str):
    f = open(output_fn,"a")
    print(out_str)
    f.write(out_str)
    f.close()

print_info("Started: %s\nFrom a previously trained model which left off on start of epoch 9.\n" % datetime.now())
# print_info("Started: %s\n" % datetime.now())

use_gpu = torch.cuda.is_available()
# use_gpu = False
if use_gpu:
    fcn_model = fcn_model.to(device)
    
best_loss = float('inf')
prev_loss = float('inf')
loss_inc_cnt = 0
stop_early = False

def train():
    softmax = nn.Softmax(dim=1)
    print("Starting Training")

    for epoch in range(start_epoch, epochs):
        
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.to(device)
                labels_cat = Y.to(device)
            else:
                inputs, labels_cat, labels_enc = X, Y, tar
            
            outputs = softmax(fcn_model(inputs))
            loss = criterion(outputs, labels_cat)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print_info("epoch{}, iter{}, loss: {} \n".format(epoch, iter, loss.item()))
                
        
        print_info("Finish epoch {}, time elapsed {} \n".format(epoch, time.time() - ts))
    
        loss, acc, IoU = evaluate(train_loader)

        print_info("Training Check:\tLoss: %f\tAccuracy: %f\tIoU: %f \n" % (loss, acc * 100, IoU))
        
        val(epoch)
        if stop_early: return

def evaluate(data_loader, validation=False, verbose=False):

    global best_loss
    global prev_loss
    global loss_inc_cnt
    global stop_early
    
    with torch.no_grad():
        losses = []
        accs = []
        ious = []
        softmax = nn.Softmax(dim=1)
        ts = time.time()
        print("Starting Evaluation")
        
        for iter, (X, tar, Y) in enumerate(data_loader):

            if use_gpu:
                inputs = X.to(device)
                labels_cat = Y.to(device)
            else:
                inputs, labels_cat, labels_enc = X, Y, tar

            outputs = softmax(fcn_model(inputs))

            output_labels = outputs.argmax(dim=1)

            losses.append(criterion(outputs, labels_cat).item())

            accs.append(pixel_acc(output_labels, labels_cat))

            ious.append(np.nanmean(iou(output_labels, labels_cat)))

        print("Finished evaluation. Time elapsed %f" % (time.time() - ts))

        # This probably should not be a straight average, but just doing this for now
        loss = np.mean(losses)
        acc = np.mean(accs)
        IoU = np.mean(ious)
        
        if validation:
            if best_loss > loss:
                best_loss = loss
                print_info("Best Loss: " + str(best_loss) + "\n")
                torch.save(fcn_model.state_dict(), best_model_fn)
            loss_inc_cnt = loss_inc_cnt + 1 if prev_loss < loss else 0
            if loss_inc_cnt > 3: stop_early = True
            torch.save(fcn_model.state_dict(), model_fn)
        
        return loss, acc, IoU

def val(epoch):
    # fcn_model.eval()
    # Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    loss, acc, IoU = evaluate(val_loader, validation=True)
    print_info("Validation Results: Loss: %f\tAccuracy: %f\tIoU: %f \n" % (loss, acc * 100, IoU))
    if stop_early: print_info("Epoch %d:\tStopping Early" % (epoch))
    
def test():
    print(' ')
    # Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    loss, acc, IoU = evaluate(test_loader)
    print_info("Test Results:\tLoss: %f\tAccuracy: %f\tIoU: %f \n" % (loss, acc * 100, IoU))
    
if __name__ == "__main__":
#     val(0)  # show the accuracy before training
#     print_info("---------Above is accuracy before training.---------\n")
    train()
    test()




