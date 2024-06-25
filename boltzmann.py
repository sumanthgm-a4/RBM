#Importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable



#Importing the Dataset

movies = pd.read_csv("ml-1m/movies.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv("ml-1m/users.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv("ml-1m/ratings.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')



#Preparing the training set and the test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t', header = None)
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t', header = None)
test_set = np.array(test_set, dtype = 'int')



#Getting the number of users and movies

nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))



#Converting the data into an array with users in the rows and movies in the columns
#(a list of lists, the first list corresponds to the first user and so on)
#(a list of 943 lists = 943 users, each list having 1682 elements = 1682 movies)

def convert(data):
    new_data = []

    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]

        ratings = np.zeros(1682)
        ratings[id_movies - 1] = id_ratings

        new_data.append(list(ratings))

    return new_data

training_set = convert(training_set)
test_set = convert(test_set)



#Converting the data into Torch tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)



#Converting the ratings into binary ratings: 1 (Liked) or 0 (Not Liked)
#Original 0's in the dataset (user didn't rate the movie) become -1 here

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1



#Creating the architecture of the Neural Network
#Info needed to build the RBM include:
#1. No. of visible nodes and hidden nodes
#2. Weights
#3. Biases

class RBM:
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

    def predict(self, x):
        _, h = self.sample_h(x)
        _, v = self.sample_v(h)
        return v



#Creating an RBM object

nv = len(training_set[0])   #No. of movies
nh = 100    #Arbitrary no. of hidden nodes
batch_size = 100    #Arbitrary batch size

rbm = RBM(nv, nh)



#Training the RBM

nb_epochs = 10

for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, 100):
        vk = training_set[id_user : id_user + batch_size]
        v0 = training_set[id_user : id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):     #k step (10 steps) contrastive divergence
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[vk >= 0]))
        s += 1.
    norm_train_loss = train_loss/s
    print("Epoch:", epoch, "\tLoss:", norm_train_loss)



#Testing the RBM

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user : id_user + 1]
    vt = test_set[id_user : id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
    norm_test_loss = test_loss/s
print("Loss:", norm_test_loss)



#Making predictions

y_pred = rbm.predict(test_set[33:34])
print(y_pred)