
import os
import pickle
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader


import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

path_data = os.getenv('PATH_DATA')

with open(os.path.join(os.getcwd(), 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

batch_size = CPARAMS['BATCH_SIZE']


mnist_train = torchvision.datasets.MNIST(path_data, train = True, download = True,
                     transform=torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor()
                     ]) )
mnist_test  = torchvision.datasets.MNIST(path_data, train = False, download = True,
                     transform=torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor()
                     ]) )

train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle = True)
test_loader  = DataLoader(mnist_test, batch_size = batch_size, shuffle = False)

train_data   = list(); test_data   = list()
train_labels = list(); test_labels = list()

with tqdm(train_loader) as tdata:
    for idx, (batch, labels) in enumerate(tdata):
        tdata.set_description(f'Train Batch {idx}\t')
        bsize = batch.shape[0]
        train_data.append(batch.reshape(bsize, -1))
        train_labels.append(labels.reshape(bsize, -1).type(torch.float32))
    #end
#end

with tqdm(test_loader) as tdata:
    for idx, (batch, labels) in enumerate(tdata):
        tdata.set_description(f'Test Batch {idx}\t')
        bsize = batch.shape[0]
        test_data.append(batch.reshape(bsize, -1))
        test_labels.append(labels.reshape(bsize, -1).type(torch.float32))
    #end
#end

if not os.path.exists(path_data):
    os.mkdir(path_data)
#end

train_dataset = {'data' : train_data, 'labels' : train_labels}
test_dataset  = {'data' : test_data, 'labels' : test_labels}

pickle.dump( 
    train_dataset,
    open(os.path.join(path_data, 'train_dataset.pkl'), 'wb')
)

pickle.dump( 
    test_dataset,
    open(os.path.join(path_data, 'test_dataset.pkl'), 'wb')
)

