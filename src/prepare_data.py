
import os
import pickle
from tqdm import tqdm

from scipy import io
import torch
import torchvision
from torch.utils.data import DataLoader


import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

PATH_DATA = os.getenv('PATH_DATA')

with open(os.path.join(os.getcwd(), 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

BATCH_SIZE = CPARAMS['BATCH_SIZE']
DATASET_ID = CPARAMS['DATASET_ID'] 

if DATASET_ID == 'MNIST':
    
    mnist_train = torchvision.datasets.MNIST(PATH_DATA, train = True, download = True,
                         transform=torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor()
                         ]) )
    mnist_test  = torchvision.datasets.MNIST(PATH_DATA, train = False, download = True,
                         transform=torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor()
                         ]) )
    
    train_loader = DataLoader(mnist_train, batch_size = BATCH_SIZE, shuffle = True)
    test_loader  = DataLoader(mnist_test, batch_size = BATCH_SIZE, shuffle = False)
    
elif DATASET_ID == 'SZ':
    
    dataset = io.loadmat(os.path.join(PATH_DATA, DATASET_ID, 'StoianovZorzi2012_data.mat'))
    print()
    pass
else:
    raise ValueError('Dataset not valid')
#end

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

path_dump_data = os.path.join(PATH_DATA, DATASET_ID)
if not os.path.exists(path_dump_data):
    os.mkdir(path_dump_data)
#end

train_dataset = {'data' : train_data, 'labels' : train_labels}
test_dataset  = {'data' : test_data, 'labels' : test_labels}

pickle.dump( 
    train_dataset,
    open(os.path.join(path_dump_data, 'train_dataset.pkl'), 'wb')
)

pickle.dump( 
    test_dataset,
    open(os.path.join(path_dump_data, 'test_dataset.pkl'), 'wb')
)

