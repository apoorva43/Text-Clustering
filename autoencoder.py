''' Code to train the autoencoder for feature representation '''

import pandas as pd 
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer


# Hyperparameters
num_epochs = 10
batch_size = 500
learning_rate = 1e-3


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(90 * 90, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 1000),
            nn.ReLU(True), 
            nn.Linear(1000, 100), 
            nn.ReLU(True), 
            nn.Linear(100, 11))
        self.decoder = nn.Sequential(
            nn.Linear(11, 100),
            nn.ReLU(True),
            nn.Linear(100, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 2000),
            nn.ReLU(True), 
            nn.Linear(2000, 90 * 90), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_features(self, x):
        return self.encoder(x)
    
    
data = pd.read_csv('category.csv')
corpus = []
for i in data.index:
    corpus.append(data['DESCRIPTION'][i])

# extract features with tfidf
vectorizer = TfidfVectorizer(ngram_range = (1, 1), stop_words = 'english', max_features = 8100) 
X = vectorizer.fit_transform(corpus)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
X = X.todense() # csr matrix to numpy

weights = []
for epoch in range(num_epochs):
    for d in X:
        d = torch.from_numpy(d)
        d = d.view(d.size(0), -1)
        d = Variable(d).cuda()
        # forward pass
        output = model(d.float())
        loss = criterion(output.float(), d.float())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # write weights for last epoch
        if epoch == 9:
            tensor_ = model.get_features(d.float())
            array = tensor_.cpu().detach().numpy()
            weights.append(array)
                  
    print('Epoch [{}/{}], Loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data))
    
print("Writing features.. ")
wt_file = open('features.category', 'w')

for i in range(11591):
    for j in range(11):
        wt = weights[i][0][j]
        wt_file.write('{wt} '.format(wt = wt))
    wt_file.write('\n')