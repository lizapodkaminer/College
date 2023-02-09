
### Usual Python tools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import MultipleLocator;
### For working with directory
from os import listdir
from os.path import isfile, isdir, join
import zipfile
import sys
### For timing
import time
!pip install tqdm
from tqdm.notebook import tqdm
### Skilearn tools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
### Open CV
!pip3 install opencv-python
import cv2

#func for grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def pad(img, shape, dif):    
    res = np.full((shape[0], shape[0], shape[2]), 255)
    left_border = dif // 2
    right_border = shape[1] + left_border
    for i in range(len(img)):
        res[i][left_border:right_border] = img[i]
    return res

def padding(img):
    shape = np.shape(img)
    dif = shape[0] - shape[1]
    if dif > 0:
        return pad(img, shape, dif)
    elif dif < 0:
        img = np.moveaxis(img, [0, 1], [1, 0])
        return np.moveaxis(pad(img, np.shape(img), -dif), [0, 1], [1, 0])
    else:
        return img

# Function for downloading data to array of int from directory
def download_imgs(dir_img = 'hhd_dataset'):
    dir_img = dir_img
    results_files = [f for f in listdir(dir_img) 
             if isdir(join(dir_img, f))]
    results_files = [int(file) for file in results_files]
    results_files.sort()
    ###
    img_files = []
    for directory in results_files:
        img = [f for f in listdir(dir_img + '/' + str(directory)) 
             if isfile(join(dir_img  + '/' + str(directory), f))] 
        img_files.append(img)
    images = []
    for i, file in tqdm(enumerate(img_files)):
        images_in_file = []
        for img_path in file:        
            # path
            path = dir_img + '/'+ str(i) +'/' + img_path
            # Using cv2.imread() method
            image = cv2.imread(path)
#             print(np.shape(image))
            image_padd = padding(image)
            image_neg = 255 - image_padd
#             print(np.shape(image_neg))
            images_in_file.append(rgb2gray(cv2.resize(image, (32,32))))
        images.append(images_in_file)
    return images

#func for dividing into train, test, val
def trn_val_tst(images, train_len = 80, val_len = 10):
    train_img, val_img, test_img = [], [], []
    train_lbl, val_lbl, test_lbl = [], [], []
    for i, letter in enumerate(images):
        #taking random by shuffling
        shuffled_img = letter.copy()
        random.shuffle(shuffled_img)
        for j, img in enumerate(shuffled_img[:(len(shuffled_img) * train_len // 100)]):
            train_img.append(img)
            train_lbl.append(i)
        train_img = np.array(train_img)
        train_lbl = np.array(train_lbl)
        for j, img in enumerate(shuffled_img[(len(shuffled_img) * train_len // 100) : (len(shuffled_img) * (train_len + val_len) // 100)]):
            val_img.append(img)    
            val_lbl.append(i)
        val_img = np.array(val_img)
        val_lbl = np.array(val_lbl)
        for j, img in enumerate(shuffled_img[(len(shuffled_img) * (train_len + val_len) // 100)]):
            test_img.append(img) 
            test_lbl.append(i)
        test_img = np.array(test_img)
        test_lbl = np.array(test_lbl)
    return train_img, train_lbl, val_img, val_lbl, test_img, test_lbl

train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = trn_val_tst(download_imgs())

"""# Write NN"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()   # interactive mode

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, img, lbl):#, transform=None, target_transform=None):
        img_torch = torch.as_tensor(img)
        mean_img, std_img = torch.mean(img_torch.to(float)), torch.std(img_torch.to(float))
        self.list_of_img = (img_torch - mean_img) / std_img
        self.list_of_lbl = torch.as_tensor(lbl)

    def __len__(self):
        return len(self.list_of_lbl)

    def __getitem__(self, idx):
        image = self.list_of_img[idx]
        label = self.list_of_lbl[idx]
        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 27),            
        )


    def forward(self, x):
        x = self.flatten(x.to(torch.float))
        logits = self.linear_relu_stack(x)
        probs = F.softmax(logits, dim=-1)
        return probs

batch_size = 128

dataset = {
    'train': ImageDataset(img = train_img, lbl = train_lbl),
    
    'val':  ImageDataset(img = val_img, lbl = val_lbl),

    'test': ImageDataset(img = test_img, lbl = test_lbl),
    }

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val', 'test']}
dataset_sizes

dataloaders = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size,
                                          shuffle=True, num_workers=2),
    'val': torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size,
                                         shuffle=False, num_workers=2),
    'test': torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    }

model = NeuralNetwork()
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(params=model.parameters())
num_epochs = 50

from numpy.core.fromnumeric import argmax

"""# 'Clean' Model """

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    # Init variables that will save info about the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_loss_train, epoch_loss_val = [], []
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
                
            running_loss = 0.0
            
            for batch_img, batch_lbl in dataloaders[phase]:
                batch_img = batch_img.to(device)
                batch_lbl = batch_lbl.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(batch_img) # apply the model to the inputs. The output is the softmax probability of each class
                    torch_lbl = torch.zeros((batch_lbl.shape[0], 27), dtype=torch.float).to(device)
                    torch_lbl[np.arange(batch_lbl.shape[0]), batch_lbl] = 1              
                    loss = criterion(outputs, torch_lbl)

                    if phase == 'train':
                      
                        loss.backward() # Perform a step in the opposite direction of the gradient
                        optimizer.step() # Adapt the optimizer      
                        
                running_loss += loss.item() * batch_img.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
              epoch_loss_train.append(epoch_loss)
            else:
              epoch_loss_val.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' :
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

    model.load_state_dict(best_model_wts)
    return model, epoch_loss_train, epoch_loss_val

model, epoch_loss_train, epoch_loss_val = train_model(model, 
                    dataloaders,
                    criterion, 
                    optimizer_ft, 
                    num_epochs=num_epochs)

plt.plot(epoch_loss_train);
plt.plot(epoch_loss_val);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

"""# Model l1, lamb = 0.01"""

def train_model_l1(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, l1_lambda = 0.01):
    since = time.time()

    # Init variables that will save info about the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_loss_train, epoch_loss_val = [], []
    best_mae = np.inf
    get_mae = nn.L1Loss()
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
                
            running_loss = 0.0
            running_mae = 0.0
            
            for batch_img, batch_lbl in dataloaders[phase]:
                batch_img = batch_img.to(device)
                batch_lbl = batch_lbl.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(batch_img) # apply the model to the inputs. The output is the softmax probability of each class
                    torch_lbl = torch.zeros((batch_lbl.shape[0], 27), dtype=torch.float).to(device)
                    torch_lbl[np.arange(batch_lbl.shape[0]), batch_lbl] = 1              
                    loss = criterion(outputs, torch_lbl)
                    

                    if phase == 'train':
                        loss.backward() # Perform a step in the opposite direction of the gradient
                        optimizer.step() # Adapt the optimizer      
                        
                running_loss += loss.item() * batch_img.size(0)
                running_mae += get_mae(outputs, torch_lbl) * l1_lambda * batch_img.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mae = running_mae / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}')
            
            if phase == 'train':
              epoch_loss_train.append(epoch_loss)
            else:
              epoch_loss_val.append(epoch_loss)

            if phase == 'val' and epoch_mae < best_mae:
                best_mae = epoch_mae
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val: {max(epoch_loss_val):4f}')
    print(f'Best val MAE: {best_mae:4f}')

    model.load_state_dict(best_model_wts)
    return model, epoch_loss_train, epoch_loss_val

model_l1_lam_01 = NeuralNetwork()
model_l1 = model_l1_lam_01.to(device)
optimizer_ft = optim.Adam(params=model_l1_lam_01.parameters())

model_l1_lam_01, epoch_loss_train_l1_lam_01, epoch_loss_val_l1_lam_01 = train_model_l1(model_l1_lam_01, 
                                                                        dataloaders,
                                                                        criterion, 
                                                                        optimizer_ft, 
                                                                        num_epochs=num_epochs)

plt.plot(epoch_loss_train_l1_lam_01);
plt.plot(epoch_loss_val_l1_lam_01);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

"""# Model l1, lamb = 0.001"""

model_l1_lam_001 = NeuralNetwork()
model_l1_lam_001 = model_l1_lam_001.to(device)
optimizer_ft = optim.Adam(params=model_l1_lam_001.parameters())

model_l1_lam_01, epoch_loss_train_l1_lam_001, epoch_loss_val_l1_lam_001 = train_model_l1(model_l1_lam_001, 
                                                                                        dataloaders,
                                                                                        criterion, 
                                                                                        optimizer_ft, 
                                                                                        num_epochs=num_epochs)

plt.plot(epoch_loss_train_l1_lam_001);
plt.plot(epoch_loss_val_l1_lam_001);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

"""# Model l2, lamb = 0.01"""

def train_model_l2(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, l2_lambda = 0.01):
    since = time.time()

    # Init variables that will save info about the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_loss_train, epoch_loss_val = [], []
    best_mse = np.inf
    get_mse = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
                
            running_loss = 0.0
            running_mse = 0.0
            
            for batch_img, batch_lbl in dataloaders[phase]:
                batch_img = batch_img.to(device)
                batch_lbl = batch_lbl.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(batch_img) # apply the model to the inputs. The output is the softmax probability of each class
                    torch_lbl = torch.zeros((batch_lbl.shape[0], 27), dtype=torch.float).to(device)
                    torch_lbl[np.arange(batch_lbl.shape[0]), batch_lbl] = 1              
                    loss = criterion(outputs, torch_lbl)
                    
                    if phase == 'train':
                        loss.backward() # Perform a step in the opposite direction of the gradient
                        optimizer.step() # Adapt the optimizer      
                        
                running_loss += loss.item() * batch_img.size(0)
                running_mse += torch.square(torch.subtract(outputs, torch_lbl)).mean() * l2_lambda * batch_img.size(0)#get_mse(outputs, torch_lbl) * batch_img.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mse = running_mse / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} MAE: {epoch_mse:.4f}')
            
            if phase == 'train':
              epoch_loss_train.append(epoch_loss)
            else:
              epoch_loss_val.append(epoch_loss)

            if phase == 'val' and epoch_mse < best_mse:
                best_mse = epoch_mse
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val: {max(epoch_loss_val):4f}')
    print(f'Best val MSE: {best_mse:6f}')

    model.load_state_dict(best_model_wts)
    return model, epoch_loss_train, epoch_loss_val

model_l2_lam_01 = NeuralNetwork()
model_l2_lam_01 = model_l2_lam_01.to(device)
optimizer_ft = optim.Adam(params=model_l2_lam_01.parameters())

model_l2_lam_01, epoch_loss_train_l2_lam_01, epoch_loss_val_l2_lam_01 = train_model_l2(model_l2_lam_01, 
                                                                                      dataloaders,
                                                                                      criterion, 
                                                                                      optimizer_ft, 
                                                                                      num_epochs=num_epochs,
                                                                                      l2_lambda = 0.01)

plt.plot(epoch_loss_train_l2_lam_01);
plt.plot(epoch_loss_val_l2_lam_01);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

"""# Model l2, lamb = 0.001"""

model_l2_lam_001 = NeuralNetwork()
model_l2_lam_001 = model_l2_lam_001.to(device)
optimizer_ft = optim.Adam(params=model_l2_lam_001.parameters())

model_l2_lam_001, epoch_loss_train_l2_lam_001, epoch_loss_val_l2_lam_001 = train_model_l2(model_l2_lam_001, 
                                                                                          dataloaders,
                                                                                          criterion, 
                                                                                          optimizer_ft, 
                                                                                          num_epochs=num_epochs,
                                                                                          l2_lambda = 0.01)

plt.plot(epoch_loss_train_l2_lam_001);
plt.plot(epoch_loss_val_l2_lam_001);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

plt.figure(figsize=(20, 15))
plt.plot(epoch_loss_train);
plt.plot(epoch_loss_val);
plt.plot(epoch_loss_train_l1_lam_01);
plt.plot(epoch_loss_val_l1_lam_01);
plt.plot(epoch_loss_train_l1_lam_001);
plt.plot(epoch_loss_val_l1_lam_001);
plt.plot(epoch_loss_train_l2_lam_01);
plt.plot(epoch_loss_val_l2_lam_01);
plt.plot(epoch_loss_train_l2_lam_001);
plt.plot(epoch_loss_val_l2_lam_001);
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.legend(['train', 'val', 'train_l1_01', 'val_l1_01', 'train_l1_001', 'val_l1_001', 'train_l2_01', 'val_l2_01', 'train_l2_001', 'val_l2_001'])
plt.show();

sns.set(rc={'figure.figsize':(12,8)})

sns.lineplot(data = [epoch_loss_train, epoch_loss_train_l1_lam_01, epoch_loss_train_l1_lam_001, epoch_loss_train_l2_lam_01, epoch_loss_train_l2_lam_001], palette = 'GnBu', dashes = False);
sns.lineplot(data = [epoch_loss_val, epoch_loss_val_l1_lam_01, epoch_loss_val_l1_lam_001, epoch_loss_val_l2_lam_01, epoch_loss_val_l2_lam_001], palette = 'OrRd', dashes = False);
plt.legend(['train', 'train_l1_01', 'train_l1_001', 'train_l2_01', 'train_l2_001', 'val', 'val_l1_01', 'val_l1_001', 'val_l2_01', 'val_l2_001']);

"""# Dropout model"""

class NeuralNetwork_drop(nn.Module):
    def __init__(self):
        super(NeuralNetwork_drop, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 27),            
        )


    def forward(self, x):
        x = self.flatten(x.to(torch.float))
        logits = self.linear_relu_stack(x)
        probs = F.softmax(logits, dim=-1)
        return probs

model_drop = NeuralNetwork_drop()
mode_drop = model_drop.to(device)
optimizer_ft = optim.Adam(params=model_drop.parameters())

model_drop, epoch_loss_train_drop, epoch_loss_val_drop = train_model(model_drop, 
                                                                    dataloaders,
                                                                    criterion, 
                                                                    optimizer_ft, 
                                                                    num_epochs=num_epochs)

plt.plot(epoch_loss_train_drop);
plt.plot(epoch_loss_val_drop);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

"""# Dropout l2, lamb = 0.01"""

model_l2_lam_01_drop = NeuralNetwork_drop()
model_l2_lam_01_drop = model_l2_lam_01_drop.to(device)
optimizer_ft = optim.Adam(params=model_l2_lam_01_drop.parameters())

model_l2_lam_01_drop, epoch_loss_train_l2_lam_01_drop, epoch_loss_val_l2_lam_01_drop = train_model_l2(model_l2_lam_01_drop, 
                                                                                                      dataloaders,
                                                                                                      criterion, 
                                                                                                      optimizer_ft, 
                                                                                                      num_epochs=num_epochs,
                                                                                                      l2_lambda = 0.01)

plt.plot(epoch_loss_train_l2_lam_01_drop);
plt.plot(epoch_loss_val_l2_lam_01_drop);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();



"""# Dropout l2, lamb = 0.001"""

model_l2_lam_001_drop = NeuralNetwork_drop()
model_l2_lam_001_drop = model_l2_lam_001_drop.to(device)
optimizer_ft = optim.Adam(params=model_l2_lam_001_drop.parameters())

model_l2_lam_001_drop, epoch_loss_train_l2_lam_001_drop, epoch_loss_val_l2_lam_001_drop = train_model_l2(model_l2_lam_001_drop, 
                                                                                                      dataloaders,
                                                                                                      criterion, 
                                                                                                      optimizer_ft, 
                                                                                                      num_epochs=num_epochs,
                                                                                                      l2_lambda = 0.001)

plt.plot(epoch_loss_train_l2_lam_001_drop);
plt.plot(epoch_loss_val_l2_lam_001_drop);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();

train_list = [epoch_loss_train, 
              epoch_loss_train_l1_lam_01, epoch_loss_train_l1_lam_001, 
              epoch_loss_train_l2_lam_01, epoch_loss_train_l2_lam_001, 
              epoch_loss_train_drop,
              epoch_loss_train_l2_lam_01_drop, epoch_loss_train_l2_lam_001_drop]

val_list = [epoch_loss_val, 
            epoch_loss_val_l1_lam_01, epoch_loss_val_l1_lam_001, 
            epoch_loss_val_l2_lam_01, epoch_loss_val_l2_lam_001, 
            epoch_loss_val_drop,
            epoch_loss_val_l2_lam_01_drop, epoch_loss_val_l2_lam_001_drop]

sns.set(rc={'figure.figsize':(12,8)})

sns.lineplot(data = train_list, palette = 'OrRd', dashes = False);
sns.lineplot(data = val_list, palette = 'GnBu', dashes = False);

"""# Accuracy"""

best_val = [min(i) for i in val_list]
best = np.argmin(np.array(best_val))

fig = plt.figure()
plt.plot(train_list[best]);
plt.plot(val_list[best]);
plt.legend(['train', 'val'])
plt.xlabel('Epoh')
plt.ylabel('Loss')
plt.show();
fig.savefig('train_val.png')

def test():
  outputs_all = torch.as_tensor([]).to(device)
  torch_lbl_all = torch.as_tensor([]).to(device)
  for batch_img, batch_lbl in dataloaders['test']:
    batch_img = batch_img.to(device)
    batch_lbl = batch_lbl.to(device)
    optimizer_ft.zero_grad()

    outputs = model(batch_img) 
    outputs_all = torch.cat((outputs_all, outputs.argmax(dim=1)), 0)
    torch_lbl = torch.zeros((batch_lbl.shape[0], 27), dtype=torch.float).to(device)
    torch_lbl[np.arange(batch_lbl.shape[0]), batch_lbl] = 1 
    torch_lbl_all = torch.cat((torch_lbl_all, torch_lbl.argmax(dim=1)), 0)
  return torch_lbl_all.to(int).tolist(), outputs_all.to(int).tolist()

torch_lbl_all, outputs_all = test()

hebw_alp = ['alef', 'bet', 'gimel', 'dalet', 'hey', 'vav', 'zain', 'het', 'tet', 'yod', 'kaf', 'kaf_sofit', 'lamed', 'mem', 'mem sofit', 'nun', 'nun_sofit', 'samekh','ayin', 'pey', 'pey_sofit', 'tsadi', 'tsadi_sofit', 'kuf', 'resh', 'shin', 'taf']

def confusion_matrix(torch_lbl_all, outputs_all):
  conf_matr = np.zeros([27, 27])
  for i in range(len(torch_lbl_all)):
    conf_matr[torch_lbl_all[i], outputs_all[i]] += 1
  return conf_matr

conf_matr = confusion_matrix(torch_lbl_all, outputs_all)
pd.DataFrame(conf_matr, columns = hebw_alp, index = hebw_alp).to_csv('/content/gdrive/MyDrive/res_files/confusion_matrix.csv')
pd.read_csv('confusion_matrix.csv', index_col=0)

list_hebw_alp = ['alef', 'bet', 'gimel', 'dalet', 'hey', 'vav', 'zain', 'het', 'tet', 'yod', 'kaf', 'kaf_sofit', 'lamed', 'mem', 'mem sofit', 'nun', 'nun_sofit', 'samekh','ayin', 'pey', 'pey_sofit', 'tsadi', 'tsadi_sofit', 'kuf', 'resh', 'shin', 'taf', 'Avg']

#Calculating accuracy for each letter
def calculate_acc(conf_matrix):
    acc_vec = []
    for i in range(len(conf_matrix)):
        acc_vec.append(conf_matrix[i, i] / float(np.sum(conf_matrix[i, :])))
    acc_vec.append(sum(acc_vec) / 27)
    dframe = pd.DataFrame(acc_vec)
    dframe = dframe.rename(columns={0: "Accuracy"})
    dframe['Letter'] = list_hebw_alp
    dframe['Letter_id'] = range(len(list_hebw_alp))
    dframe = dframe[['Letter_id', 'Letter', 'Accuracy']]
    dframe.set_index('Letter_id', inplace=True)
    dframe.to_csv('acc.csv', index='Letter_id')
    return dframe

dframe = calculate_acc(conf_matr)
dframe

