
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#end

# -----------------------------------------------------
# LOAD GLOBAL PATH VARIABLES
PATH_DATA  = os.getenv('PATH_DATA')
PATH_MODEL = os.getenv('PATH_MODEL')
# -----------------------------------------------------


with open(os.path.join(os.getcwd(), 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
filestream.close()

DATASET_ID = CPARAMS['DATASET_ID']

# -----------------------------------------------------
# REDEFINE DATASET-SPECIFIC PATH
PATH_DATA = os.path.join(PATH_DATA, DATASET_ID)
PATH_MAIN = os.getcwd()
# -----------------------------------------------------

# -----------------------------------------------------
# GLOBAL VARIABLES
ALG_NAME = CPARAMS['ALG_NAME']
MODEL_NAME = f'{ALG_NAME}DBN'
PATH_MODEL = os.path.join(PATH_MODEL, MODEL_NAME, DATASET_ID)

if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)
#end

READOUT        = CPARAMS['READOUT']
RUNS           = CPARAMS['RUNS']
LAYERS         = CPARAMS['LAYERS']
NUM_DISCR      = CPARAMS['NUM_DISCR']

if DATASET_ID == 'MNIST' and NUM_DISCR:
    raise ValueError('No numerosity discrimination with MNIST')
#end
if DATASET_ID == 'MNIST' and LAYERS != 3:
    raise ValueError('LAYERS with MNIST should be 3')
if DATASET_ID == 'SZ' and LAYERS != 2:
    raise ValueError('LAYERS with SZ should be 2')
#end

with open(os.path.join(os.getcwd(), f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
    LPARAMS = json.load(filestream)
filestream.close()

EPOCHS         = LPARAMS['EPOCHS']
INIT_MOMENTUM  = LPARAMS['INIT_MOMENTUM']
FINAL_MOMENTUM = LPARAMS['FINAL_MOMENTUM']
LEARNING_RATE  = LPARAMS['LEARNING_RATE']
WEIGHT_PENALTY = LPARAMS['WEIGHT_PENALTY']

if NUM_DISCR:
    NUM_LCLASSFRS  = LPARAMS['NUM_LCLASSIFIERS']
    EPOCHS_NDISCR  = LPARAMS['EPOCHS_NDISCR']
#end
# -----------------------------------------------------

# -----------------------------------------------------
# Load data
train_dataset = pickle.load(open(os.path.join(PATH_DATA, 'train_dataset.pkl'), 'rb'))
test_dataset  = pickle.load(open(os.path.join(PATH_DATA, 'test_dataset.pkl'),  'rb'))

# Converto to cuda, if available
train_dataset['data'] = train_dataset['data'].to(DEVICE)
test_dataset['data']  = test_dataset['data'].to(DEVICE)
train_dataset['labels'] = train_dataset['labels'].to(DEVICE)
test_dataset['labels']  = test_dataset['labels'].to(DEVICE)

# -----------------------------------------------------
# Initialize performance metrics data structures
loss_metrics = np.zeros((RUNS, EPOCHS, LAYERS))
acc_metrics  = np.zeros((RUNS, EPOCHS, LAYERS))
Weber_fracs  = list()
psycurves    = list()
# -----------------------------------------------------


# -----------------------------------------------------
# Runs
for run in range(RUNS):
    
    print(f'\n\nRun {run}\n')
    if DATASET_ID == 'MNIST':
        model = [
            {'W' : 0.01 * torch.nn.init.normal_(torch.empty(784, 500), mean = 0, std = 1),  
             'a' : torch.zeros((1, 784)),  
             'b' : torch.zeros((1, 500))},
            {'W' : 0.01 * torch.nn.init.normal_(torch.empty(500, 500), mean = 0, std = 1),  
             'a' : torch.zeros((1, 500)),  
             'b' : torch.zeros((1, 500))},
            {'W' : 0.01 * torch.nn.init.normal_(torch.empty(500, 2000), mean = 0, std = 1), 
             'a' : torch.zeros((1, 500)),  
             'b' : torch.zeros((1, 2000))}
        ]
        
    elif DATASET_ID == 'SZ':
        model = [
            {'W' : 0.1 * torch.nn.init.normal_(0, 1, (900, 80)), 
             'a' : torch.zeros((1, 900)), 
             'b' : torch.zeros((1, 80))},
            {'W' : 0.1 * torch.nn.init.normal_(0, 1, (80, 400)), 
             'a' : torch.zeros((1, 80)),  
             'b' : torch.zeros((1, 400))}
        ]
    #end
    
    dbn = DBN(model, ALG_NAME, PATH_MODEL, EPOCHS).to(DEVICE)
    
    if ALG_NAME == 'g':
        dbn.train_greedy(train_dataset, test_dataset,
                         LPARAMS, readout = READOUT
        )
    elif ALG_NAME == 'i':
        dbn.train_iterative(train_dataset, test_dataset,
                            LPARAMS, readout = READOUT,
                            num_discr = NUM_DISCR
        )
    elif ALG_NAME == 'f':
        dbn.train_fullstack(train_dataset, test_dataset, LPARAMS)
    #end
    
    loss_metrics[run,:,:] = dbn.loss_profile
    acc_metrics[run,:,:] = dbn.acc_profile
    
    if NUM_DISCR:
        Weber_fracs.append(dbn.Weber_fracs) # list of pd.DataFrame
        psycurves.append(dbn.psycurves)     # list of dicts with ratios and percs
    #end
    
    dbn.save(name = f'run{run}')
#end
# -----------------------------------------------------

# -----------------------------------------------------
# Serialize results
with open(os.path.join(PATH_MODEL, 'loss_metrics.pkl'), 'wb') as f:
    pickle.dump(loss_metrics, f)
f.close()

with open(os.path.join(PATH_MODEL, 'acc_metrics.pkl'), 'wb') as f:
    pickle.dump(acc_metrics, f)
f.close()

if NUM_DISCR:
    with open(os.path.join(PATH_MODEL, 'Weber_fracs.pkl'), 'wb') as f:
        pickle.dump(Weber_fracs, f)
    f.close()
    
    with open(os.path.join(PATH_MODEL, 'psycurves.pkl'), 'wb') as f:
        pickle.dump(psycurves, f)
    f.close()
#end
# -----------------------------------------------------


