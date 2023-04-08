import os
import tqdm

from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms, datasets

from model import Net84, Net28
from data import SmoteDataset
from collections import Counter
from train import train

BATCH_SIZE = 48
EPOCH = 300
LEARNING_RATE = 1e-3
# TRAIN_SIZE = 0.8
GPU_FLAG = torch.cuda.is_available()
TRAIN_SET_PATH = '/Users/lixiaoyang/Desktop/CS4486/HW_3 (1)/Topic_5_Data/ISIC84by84/Train/'
TEST_SET_PATH = '/Users/lixiaoyang/Desktop/CS4486/HW_3 (1)/Topic_5_Data/ISIC84by84/Test/'
CSV_PATH = '/Users/lixiaoyang/Desktop/CS4486/HW_3 (1)/hmnist_28_28_RGB.csv'
CSV_FLAG = False

device = torch.device('cuda' if GPU_FLAG else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
model = Net84().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

if CSV_FLAG:
    print('loading dataset...')
    data = pd.read_csv(CSV_PATH, encoding='utf-8')
    labels = data['label']
    images = data.drop(columns=['label'])
    # print('before image: ' + str(type(images)) + str(images.shape))
    # print('before label: ' + str(type(labels)) + str(labels.shape))
    images, labels = SMOTE().fit_resample(images, labels)
    # print('after image: ' + str(type(images)) + str(images.shape))
    # print('after label: ' + str(type(labels)) + str(labels.shape))
    X_train, X_eval, y_train, y_eval = train_test_split(images, labels, train_size=0.8, random_state=21)
    y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    y_eval = torch.from_numpy(np.array(y_eval)).type(torch.LongTensor)
    # print('X: ' + str(type(X_train)) + str(X_train.shape))
    # print('X_test: ' + str(type(X_eval)) + str(X_eval.shape))
    # print('y: ' + str(type(y_train)) + str(y_train.shape))
    # print('y_test: ' + str(type(y_eval)) + str(y_eval.shape))
    train_data = SmoteDataset(df=X_train, labels=y_train, transform=transform)
    eval_data = SmoteDataset(df=X_eval, labels=y_eval, transform=transform)
    data_loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_eval = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=True)
    print('finish loading dataset...')
    summary(model, input_size=(3, 28, 28))
else:
    print('loading dataset...')
    train_set = datasets.ImageFolder(TRAIN_SET_PATH)
    test_set = datasets.ImageFolder(TEST_SET_PATH)
    X_train = np.array([np.array(img).flatten() for (img, label) in train_set])
    y_train = np.array([label for (img, label) in train_set])
    X_eval = np.array([np.array(img).flatten() for (img, label) in test_set])
    y_eval = np.array([label for (img, label) in test_set])
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    # X_eval, y_eval = SMOTE().fit_resample(X_eval, y_eval)
    X_train, X_eval = pd.DataFrame(X_train), pd.DataFrame(X_eval)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    y_eval = torch.from_numpy(y_eval).type(torch.LongTensor)
    # print('X: ' + str(type(X)) + str(X.shape))
    # print('y: ' + str(type(y)) + str(y.shape))
    train_data = SmoteDataset(df=X_train, labels=y_train, transform=transform)
    eval_data = SmoteDataset(df=X_eval, labels=y_eval, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=True)
    print('finish loading dataset...')
    summary(model, input_size=(3, 84, 84))
    train(epoch_n=EPOCH, 
          batch_size=BATCH_SIZE, 
          device=device, model=model, 
          criterion=loss_function, 
          optimizer=optimizer, 
          train_loader=train_loader, 
          test_loader=test_loader)

