
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'dbn-utls'))

import torch
import numpy as np

import pickle
import json

from models import DBN

from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


PATH_DATA  = os.getenv('PATH_DATA')
PATH_MODEL = os.getenv('PATH_MODEL')

with open(os.path.join(os.getcwd(), 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

ALG_NAME = CPARAMS['ALG_NAME']
MODEL_NAME = f'{ALG_NAME}DBN'
PATH_MODEL = os.path.join(PATH_MODEL, MODEL_NAME)

if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)
#end

READOUT        = CPARAMS['READOUT']
RUNS           = CPARAMS['RUNS']
EPOCHS         = CPARAMS['EPOCHS']
LAYERS         = CPARAMS['LAYERS']
INIT_MOMENTUM  = CPARAMS['INIT_MOMENTUM']
FINAL_MOMENTUM = CPARAMS['FINAL_MOMENTUM']
LEARNING_RATE  = CPARAMS['LEARNING_RATE']
WEIGHT_PENALTY = CPARAMS['WEIGHT_PENALTY']

train_dataset = pickle.load(open(os.path.join(PATH_DATA, 'train_dataset.pkl'), 'rb'))
test_dataset  = pickle.load(open(os.path.join(PATH_DATA, 'test_dataset.pkl'),  'rb'))

loss_metrics = np.zeros((RUNS, EPOCHS, LAYERS))
acc_metrics  = np.zeros((RUNS, EPOCHS, LAYERS))

for run in range(RUNS):
    
    model = [
        {'W' : torch.normal(0, 0.01, (784, 500)),  'a' : torch.zeros((1, 784)),  'b' : torch.zeros(1, 500)},
        {'W' : torch.normal(0, 0.01, (500, 500)),  'a' : torch.zeros((1, 500)),  'b' : torch.zeros(1, 500)},
        {'W' : torch.normal(0, 0.01, (500, 2000)), 'a' : torch.zeros((1, 500)),  'b' : torch.zeros((1, 2000))}
    ]
    
    for layer in model:
        for key in layer.keys():
            layer[key].to(device)
        #end
    #end
    
    dbn = DBN(model, ALG_NAME, PATH_MODEL, EPOCHS).to(device)
    
    if ALG_NAME == 'g':
        dbn.train_greedy(train_dataset, test_dataset,
                          [INIT_MOMENTUM, FINAL_MOMENTUM],
                          LEARNING_RATE, WEIGHT_PENALTY,
                          readout = READOUT
        )
    elif ALG_NAME == 'i':
        dbn.train_iterative(train_dataset, test_dataset,
                          [INIT_MOMENTUM, FINAL_MOMENTUM],
                          LEARNING_RATE, WEIGHT_PENALTY,
                          readout = READOUT
        )
    #end
    loss_metrics[run,:,:] = dbn.loss_profile
    acc_metrics[run,:,:] = dbn.acc_profile
    dbn.save(name = f'run{run}')
#end

with open(os.path.join(PATH_MODEL, 'loss_metrics.pkl'), 'wb') as f:
    pickle.dump(loss_metrics, f)
f.close()

with open(os.path.join(PATH_MODEL, 'acc_metrics.pkl'), 'wb') as f:
    pickle.dump(acc_metrics, f)
f.close()