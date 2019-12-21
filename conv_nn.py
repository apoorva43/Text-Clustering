''' Code to train the CNN for feature representation '''

import pandas as pd
import numpy as np
import torch.nn.init
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tqdm import tqdm
from scipy.stats import truncnorm
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer


def weight_variable(shape):
    initial = torch.Tensor(truncnorm.rvs(-1/0.01, 1/0.01, scale = 0.01, size = shape))
    return Parameter(initial, requires_grad = True)


def bias_variable(shape):
    initial = torch.Tensor(np.ones(shape) * 0.1)
    return Parameter(initial, requires_grad = True)

def conv2d(x, W, stride):
    return F.conv2d(x, W, stride = stride, padding = 2)


data = pd.read_csv('category.csv')
corpus = []
for i in data.index:
    corpus.append(data['DESCRIPTION'][i])

# extract features with tfidf
vectorizer = TfidfVectorizer(ngram_range = (1, 1), stop_words = 'english', max_features = 10000) 
X = vectorizer.fit_transform(corpus)

# Model parameters
weight_conv1 = weight_variable((5, 1, 5, 5)) 
weight_conv2 = weight_variable((10, 5, 5, 5))
weight_fc = weight_variable((64, 25 * 25 * 5 * 2)) 
weight_out = weight_variable((2, 64))
bias_conv1 = bias_variable((5)) 
bias_conv2 = bias_variable((10)) 
bias_fc = bias_variable((64)) 
bias_out = bias_variable((2)) 

X_ = X.todense()
X_np = torch.from_numpy(X_).cuda()
X = X_np.view(-1, 1, 100, 100)
X = X.double()
del X_np # RAM constraints 

print("Training..")
conv1 = conv2d(X, weight_conv1.double().cuda(), 2) # convolution
conv1 = conv1.view(-1, 5) 

conv2 = torch.nn.functional.relu(conv1.double().cuda() + bias_conv1.double().cuda()) # apply ReLU
del X
del conv1 

conv2 = conv2d(conv2.view(-1, 5, 50, 50), weight_conv2.double().cuda(), 2) # convolution 
conv2 = conv2.view(-1, 10)
fc = torch.nn.functional.relu(conv2.double().cuda() + bias_conv2.double().cuda()) # apply ReLU
del conv2

fc = fc.view(11591, -1)
fc = torch.mm(fc.cuda(), weight_fc.double().t().cuda())
fc = torch.nn.functional.relu(fc.double().cuda() + bias_fc.double().cuda()) 
y = torch.addmm(bias_out.double().cuda(), fc.double().cuda(), weight_out.double().t().cuda()) # linear
del fc
weights = y.detach().cpu().numpy()


print("Writing features.. ")
wt_file = open('features.category', 'w')

for i in range(11591):
    for j in range(2):
        wt = weights[i][j]
        wt_file.write('{wt} '.format(wt = wt))
    wt_file.write('\n')