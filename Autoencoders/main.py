"""
Author: Enrique Manuel Companioni Valle
Date: 03-17-2024

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#### For get the full datasets

# Importing the dataset
#movies_path = "data/ml-1m/ml-1m/movies.dat"
#users_path = "data/ml-1m/ml-1m/users.dat"
#ratings_path = "data/ml-1m/ml-1m/ratings.dat"

#movies = pd.read_csv(movies_path, sep="::", header=None, encoding="latin-1", engine="python")
#users = pd.read_csv(users_path, sep="::", header=None, encoding="latin-1", engine="python")
#ratings = pd.read_csv(ratings_path, sep="::", header=None, encoding="latin-1", engine="python")

# Show data
#print("Movies sample: \n", movies.head())
#print("\nUsers sample: \n", users.head())
#print("\nRatings sample: \n", ratings.head())

####

# Loading splitted data
training_first_split = "data/ml-100k/u1.base"
training_set = pd.read_csv(training_first_split, delimiter="\t").to_numpy(dtype=int)

test_first_split = "data/ml-100k/u1.test"
test_set = pd.read_csv(test_first_split, delimiter="\t").to_numpy(dtype=int)

# Conform two matrix, one for the training dataset and other for the test dataset
# This two matrix represent the total of users and movies and the movies were not rating
# recieve a zero.

# Get the number of users and movies
max_num_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
max_num_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Convert the data into an array with users in rows and movies in columns
def convert(data):
    new_data = []
    for user in range(1, max_num_movies + 1):
        id_movies = data[:,1][data[:,0] == user]
        id_ratings = data[:,2][data[:,0] == user]
        ratings_list = np.zeros(max_num_movies)
        ratings_list[id_movies - 1] = id_ratings
        new_data.append(list(ratings_list))
    return new_data

# Apply the function to training and test set
training_set = convert(training_set)
test_set = convert(test_set)

# Convert to tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
print(training_set.size())
print(test_set.size())

# Creating the Neural Network
class Sparse_AutoEncoder(nn.Module):
    def __init__(self, max_num_movies):
        super(Sparse_AutoEncoder, self).__init__()
        # Encoding
        self.input_layer = nn.Linear(max_num_movies, 20)
        self.hl_1 = nn.Linear(20, 10)
        # Decoding  
        self.hl_2 = nn.Linear(10, 20)  
        self.output_layer = nn.Linear(20, max_num_movies)

        self.activation = nn.Sigmoid()

        # Loss function
        self.loss = nn.MSELoss()
        # Optimizer
        self.optimizer = optim.RMSprop(self.parameters(), lr= 0.001, weight_decay=0.5)

    def forward(self, x):
        X = self.activation(self.input_layer(x))
        X = self.activation(self.hl_1(X))
        X = self.activation(self.hl_2(X))
        X = self.output_layer(X)

        return X

sparse_autoencoder = Sparse_AutoEncoder(max_num_movies)

# Training the Sparse AutoEncoder
epochs = 200

for epoch in range(1, epochs + 1):
    train_loss = 0
    s = 0.
    for id_users in range(max_num_users):
        # Create the batch of data for each user
        input_data = Variable(training_set[id_users]).unsqueeze(0)
        target = input_data.clone()
        # If the user rating at least one movie then compute the training
        if torch.sum(target.data > 0) > 0:            
            output = sparse_autoencoder(input_data)            
            target.requires_grad = False
            output[target == 0] = 0            
            loss = sparse_autoencoder.loss(output, target)            
            mean_corrector = max_num_movies / float(torch.sum(target.data > 0) + 1e-10)        
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)            
            s += 1.
            sparse_autoencoder.optimizer.step()
    print("epoch: " + str(epoch) + " loss: " + str(train_loss/s) )

# Testing the sparse autoencoder
test_loss = 0
s = 0.
for id_users in range(max_num_users):
    # Create the batch of data for each user
    input_data = Variable(training_set[id_users]).unsqueeze(0)
    target = Variable(test_set[id_users])
    # If the user rating at least one movie then compute the training
    if torch.sum(target.data > 0) > 0:
        output = sparse_autoencoder(input_data)
        target.requires_grad = False
        output[target == 0] = 0
        loss = sparse_autoencoder.loss(output, target)
        mean_corrector = max_num_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.        
print("loss: " + str(test_loss/s))













