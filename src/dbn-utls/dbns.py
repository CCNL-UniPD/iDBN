
import os
import torch
import rbms
import random
from tqdm import tqdm


class NonLinear(torch.nn.Linear):
    '''UNDER MAINTENANCE'''
    def __init__(self, Nin, Nout):
        super(NonLinear, self).__init__()
        
        self.linear = torch.nn.Linear(Nin, Nout, bias = False)
        self.bias   = torch.nn.parameter.Parameter(torch.zeros(1, Nout))
    #end
    
    def forward(self, x):
        return self.linear(x).add(self.bias)
    #end
#end


class DBN(torch.nn.Module):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs):
        super(DBN, self).__init__()
        
        if dataset_id == 'MNIST':
            self.rbm_layers = [
                rbms.RBM(784, 500, epochs,
                         layer_id = 0,  
                         init_scheme = init_scheme,
                         dataset_id = dataset_id),
                rbms.RBM(500, 500, epochs,
                         layer_id = 1,
                         init_scheme = init_scheme,
                         dataset_id = dataset_id),
                rbms.RBM(500, 2000, epochs,
                         layer_id = 2, 
                         init_scheme = init_scheme, 
                         dataset_id = dataset_id)
            ]
        #end
        
        self.alg_name = alg_name
        self.init_scheme = init_scheme
        self.dataset_id = dataset_id
        self.path_model = path_model
    #end
    
    def forward(self, v, only_forward = False):
        
        for rbm in self.rbm_layers:
            v, p_v = rbm(v)
        #end
        
        if only_forward:
            return p_v, v
        
        else:
            
            for rbm in reversed(self.rbm_layers):
                v, _ = rbm.backward(v)
            #end
            
            return v
        #end
    #end
    
    def test(self, Xtest, Ytest, mode = 'reproduction'):
        
        eval_fn = torch.nn.MSELoss(reduction = 'mean')
        
        if mode == 'reproduction':
            data = Xtest
        if mode == 'reconstruction':
            # corrupt and reconstruct
            # data = ...
            pass
        if mode == 'denoise':
            # data = ...
            # add torch.normal(mean, std, size).to(DEVICE)
            pass
        #end
        
        out_data = self(data)
        error = eval_fn(data, out_data)
        
        return error, out_data
    #end
    
    def get_name(self):
        
        algo_names = {'i' : 'iterative', 'g' : 'greedy', 'fs' : 'fullstack'}
        algo = algo_names[self.alg_name]
        name = f'dbn_{algo}_{self.init_scheme}_{self.dataset_id}'
        return name
    #end
    
    def save(self, run = None):
        
        for layer in self.rbm_layers:
            layer.flush_gradients()
            layer.to(torch.device('cpu'))
        #end
        
        name = self.get_name()
        if run is not None:
            name += f'_run{run}'
        #end
        torch.save(self.to(torch.device('cpu')),
                   open(os.path.join(self.path_model, f'{name}.pkl'), 'wb'))
    #end
#end


class gDBN(DBN):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs):
        super(gDBN, self).__init__(alg_name, dataset_id, init_scheme, path_model, epochs)
        
        self.algo = 'g'
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams, readout = False):
        
        for rbm in self.rbm_layers:
            
            print(f'--Layer {rbm.layer_id}')
            _Xtrain, _Xtest = rbm.train(Xtrain, Xtest, Ytrain, Ytest, 
                                        lparams, readout = readout)
            Xtrain = _Xtrain
            Xtest = _Xtest
        #end
    #end
#end

class iDBN(DBN):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs):
        super(iDBN, self).__init__(alg_name, dataset_id, init_scheme, path_model, epochs)
        
        self.algo = 'i'
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams, 
              readout = False, num_discr = False):
        
        for rbm in self.rbm_layers:
            rbm.dW = torch.zeros_like(rbm.W)
            rbm.da = torch.zeros_like(rbm.a)
            rbm.db = torch.zeros_like(rbm.b)
        #end
        
        for epoch in range(lparams['EPOCHS']):
            
            print(f'--Epoch {epoch}')
            self.current_epoch = epoch
            self.epochs_loop(Xtrain, Xtest, Ytrain, Ytest, lparams, readout)
            
        #end EPOCHS
        
    #end
    
    def epochs_loop(self, Xtrain, Xtest, Ytrain, Ytest, lparams, readout):
        
        n_train_batches = Xtrain.shape[0]
        n_test_batches = Xtest.shape[0]
        batch_size = Xtrain.shape[1]
        
        for rbm in self.rbm_layers:
            
            _Xtrain = torch.zeros((n_train_batches, batch_size, rbm.Nout))
            _Xtest = torch.zeros((n_test_batches, batch_size, rbm.Nout))
            rbm.current_epoch = self.current_epoch
            train_loss = 0.
            
            _Xtest, _ = rbm(Xtest)
            
            batch_indices = list(range(n_train_batches))
            random.shuffle(batch_indices)
            with tqdm(batch_indices, unit = 'Batch') as tlayer:
                for idx, n in enumerate(tlayer):
                    
                    tlayer.set_description(f'Layer {rbm.layer_id}')
                    _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])
                    pos_v = Xtrain[n,:,:]
                    loss = rbm.CD_params_update(pos_v, lparams)
                    
                    train_loss += loss
                    tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                #end BATCHES
            #end WITH
            
            rbm.loss_profile[self.current_epoch] = train_loss.div(n_train_batches)
            
            if readout:
                readout_acc = rbm.get_readout(_Xtrain, _Xtest, Ytrain, Ytest)
                print(f'Readout accuracy = {readout_acc*100:.2f}')
                rbm.acc_profile[self.current_epoch] = readout_acc
            #end
            
            Xtrain = _Xtrain.clone()
            Xtest  = _Xtest.clone()
        #end LAYERS
    #end
#end

class fsDBN(DBN):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs):
        super(fsDBN, self).__init__(alg_name, dataset_id, init_scheme, path_model, epochs)
        
        self.algo = 'fs'
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams):
        
        for rbm in self.rbm_layers:
            rbm.dW = torch.zeros_like(rbm.W)
            rbm.db = torch.zeros_like(rbm.b)
            rbm.da = torch.zeros_like(rbm.a)
        #end
        
        for epoch in range(lparams['EPOCHS']):
            
            print(f'--Epoch {epoch}')
            self.current_epoch = epoch
            self.epochs_loop(Xtrain, Xtest, Ytrain, Ytest, lparams)
        #end
        
        for rbm in self.rbm_layers:
            rbm.delete_field('act_topdown')
        #end
    #end
    
    def epochs_loop(self, Xtrain, Xtest, Ytrain, Ytest, lparams):
        
        n_train_batches = Xtrain.shape[0]
        batch_size = Xtrain.shape[1]
        
        # Bottom-up loop
        p_act, act = self(Xtrain, only_forward = True)
        
        # Top-down loop
        for rbm in reversed(self.rbm_layers):
            
            rbm.save_topdown_act( (p_act, act) )
            p_act, act = rbm.backward(act)
        #end
        
        # Training loop
        for rbm in self.rbm_layers:
            
            _Xtrain = torch.zeros((n_train_batches, batch_size, rbm.Nout))
            rbm.current_epoch = self.current_epoch
            train_loss = 0.
            
            batch_indices = list(range(n_train_batches))
            random.shuffle(batch_indices)
            with tqdm(batch_indices, unit = 'Batch') as tlayer:
                for idx, n in enumerate(tlayer):
                    
                    tlayer.set_description(f'Layer {rbm.layer_id}')
                    _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])
                    pos_v = Xtrain[n,:,:]
                    topdown_pact, topdown_act = rbm.get_topdown_act()
                    loss = rbm.CD_params_update(pos_v, lparams, 
                            hidden_saved = (topdown_pact[n,:,:], topdown_act[n,:,:]))
                    
                    train_loss += loss
                    tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                #end BATCHES
            #end WITH
            
            rbm.loss_profile[self.current_epoch] = train_loss.div(n_train_batches)
            Xtrain = _Xtrain.clone()
        #end LAYERS
        
#end