
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from wftools import DeltaRule, get_Weber_frac

import os
from tqdm import tqdm
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DBN(torch.nn.Module):
    
    def __init__(self, network, name, path_model, epochs):
        super(DBN, self).__init__()
        
        self.network = network
        self.name = name
        self.loss_profile = np.zeros((epochs, network.__len__()))
        self.acc_profile = np.ones((epochs, network.__len__())) * np.nan
        self.path_model = path_model
        self.epochs = epochs
        self.loss_fn = torch.nn.MSELoss(reduction = 'mean')
        
        for layer in self.network:
            layer['W'].to(DEVICE)
            layer['a'].to(DEVICE)
            layer['b'].to(DEVICE)
        #end
    #end
    
    def train_greedy(self, train_dataset, test_dataset, learning_params, readout = False):
        
        momenta = [ learning_params['INIT_MOMENTUM'], 
                    learning_params['FINAL_MOMENTUM'] ]
        lr      = learning_params['LEARNING_RATE']
        penalty = learning_params['WEIGHT_PENALTY']
        
        train_data = train_dataset['data']
        train_lbls = train_dataset['labels']
        test_data  = test_dataset['data']
        test_lbls  = test_dataset['labels']
        
        train_batches = train_data.shape[0]
        test_batches  = test_data.shape[0]
        batch_size    = learning_params['BATCH_SIZE']
        
        activities   = None
        t_activities = None
        
        print('train')
        for layer_id, layer in enumerate(self.network):
            print(f'Layer {layer_id}')
            
            N_out = layer['W'].shape[1]
            
            if layer_id == 0:
                data   = train_data.clone()
                t_data = test_data.clone()
            else:
                data   = activities.clone()
                t_data = t_activities.clone()
            #end
            
            activities   = torch.zeros(train_batches, batch_size, N_out)
            t_activities = torch.zeros(test_batches, batch_size, N_out)
            
            W = layer['W'].clone(); dW = torch.zeros_like(W)
            a = layer['a'].clone(); da = torch.zeros_like(a)
            b = layer['b'].clone(); db = torch.zeros_like(b)
            
            for epoch in range(self.epochs):
                
                train_loss = 0.
                indices = list(range(train_batches))
                random.shuffle(indices)
                
                with tqdm(indices, unit = 'Batch') as tepoch:
                    for idx, n in enumerate(tepoch):
                        
                        tepoch.set_description(f'Epoch {epoch:03d}')
                        batch_size = data[n].shape[0]
                        
                        pos_v = data[n]
                        pos_ph, pos_h = self.sample(W, b, pos_v)
                        # neg_pv, neg_v = self.sample(W.t(), a, pos_ph)
                        # neg_ph, neg_h = self.sample(W, b, neg_v)
                        neg_ph, neg_v, neg_pv = self.Gibbs_sampling(pos_v, W, a, b)
                        
                        pos_dW = torch.matmul(pos_v.t(), pos_ph).div(batch_size)
                        pos_da = pos_v.mean(dim = 0)
                        pos_db = pos_ph.mean(dim = 0)
                        
                        neg_dW = torch.matmul(neg_v.t(), neg_ph).div(batch_size)
                        neg_da = neg_v.mean(dim = 0)
                        neg_db = neg_ph.mean(dim = 0)
                        
                        if epoch >= 5:
                            momentum = momenta[0]
                        else:
                            momentum = momenta[1]
                        #end
                        
                        dW = momentum * dW + lr * ((pos_dW - neg_dW) - penalty * W)
                        da = momentum * da + lr * (pos_da - neg_da)
                        db = momentum * db + lr * (pos_db - neg_db)
                        
                        W = W + dW
                        a = a + da
                        b = b + db
                        
                        mse = (pos_v - neg_pv).pow(2).mean() 
                        train_loss += mse
                        
                        activities[n], _ = self.sample(W, b, pos_v)
                        
                        tepoch.set_postfix(MSE = train_loss.div(idx + 1).item())
                    #end batches
                #end with batches
                
                t_activities = self.sample(W, b, t_data)[0]
                
                if readout:
                    
                    if (epoch + 1) % 1 == 0:
                        
                        readout_accuracy = self.get_readout(activities, t_activities, 
                                                            train_lbls, test_lbls)
                        self.acc_profile[epoch, layer_id] = readout_accuracy
                        print(f'Readout accuracy = {readout_accuracy*100:.2f}')
                    #end
                #end
                
                self.loss_profile[epoch, layer_id] = train_loss.div(data.__len__()).item()
            #end epochs
            
            self.network[layer_id]['W'] = W.clone()
            self.network[layer_id]['a'] = a.clone()
            self.network[layer_id]['b'] = b.clone()
        #end layers
    #end
    
    def train_iterative(self, train_dataset, test_dataset, learning_params, 
                        readout = False, num_discr = False):
        
        if num_discr:
            self.num_discr = True
            self.Weber_fracs = pd.DataFrame(
                columns = range(learning_params['NUM_LCLASSIFIERS']),
                index   = learning_params['EPOCHS_NDISCR'])
            self.psycurves = dict()
        else:
            self.num_discr = False
        #end
        
        momenta = [ learning_params['INIT_MOMENTUM'], 
                    learning_params['FINAL_MOMENTUM'] ]
        lr = learning_params['LEARNING_RATE']
        penalty = learning_params['WEIGHT_PENALTY']
        
        velocities = list()
        for layer in self.network:
            velocities.append({
                'dW' : torch.zeros_like(layer['W']),
                'da' : torch.zeros_like(layer['a']),
                'db' : torch.zeros_like(layer['b'])
            })
        #end
        
        for epoch in range(self.epochs):
            
            Xtrain = train_dataset['data']
            Ytrain = train_dataset['labels']
            Xtest  = test_dataset['data']
            Ytest  = test_dataset['labels']
            
            train_batches = Xtrain.shape[0]
            test_batches  = Xtest.shape[0]
            batch_size    = learning_params['BATCH_SIZE']
            
            print(f'Epoch {epoch:03d}')
            
            for layer_id, layer in enumerate(self.network):
                
                N_out = layer['W'].shape[1]
                
                _Xtrain = torch.zeros((train_batches, batch_size, N_out))
                _Xtest  = torch.zeros((test_batches, batch_size, N_out))
                _Xtest, _ = self.sample(layer['W'], layer['b'], Xtest)
                
                indices = list(range(train_batches))
                train_loss = 0.
                
                W = layer['W'];  dW = velocities[layer_id]['dW'].clone()
                a = layer['a'];  da = velocities[layer_id]['da'].clone()
                b = layer['b'];  db = velocities[layer_id]['db'].clone()
                
                random.shuffle(indices)
                with tqdm(indices, unit = 'Batch') as tlayer:
                    for idx, n in enumerate(tlayer):
                        
                        tlayer.set_description(f'Layer {layer_id}')
                        pos_v = Xtrain[n,:,:].clone()
                        batch_size = pos_v.shape[0]
                        
                        # Propagation of activity before parameters update!!!
                        _Xtrain[n,:,:], _ = self.sample(W, b, pos_v)
                        
                        pos_ph, pos_h = self.sample(W, b, pos_v)
                        neg_ph, neg_v, neg_pv = self.Gibbs_sampling(pos_v, W, a, b)
                        
                        pos_dW = torch.matmul(pos_v.t(), pos_ph).div(batch_size)
                        pos_da = pos_v.mean(dim = 0)
                        pos_db = pos_ph.mean(dim = 0)
                        
                        neg_dW = torch.matmul(neg_v.t(), neg_ph).div(batch_size)
                        neg_da = neg_v.mean(dim = 0)
                        neg_db = neg_ph.mean(dim = 0)
                        
                        if epoch >= 5:
                            momentum = momenta[0]
                        else:
                            momentum = momenta[1]
                        #end
                        
                        dW = momentum * dW + lr * ((pos_dW - neg_dW) - penalty * W)
                        da = momentum * da + lr * (pos_da - neg_da)
                        db = momentum * db + lr * (pos_db - neg_db)
                        
                        W = W + dW
                        a = a + da
                        b = b + db
                        
                        mse = self.loss_fn(pos_v, neg_pv)#(pos_v - neg_pv).pow(2).mean()
                        train_loss += mse
                        
                        tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                        
                    #end FOR batches
                #end WITH batches
                
                velocities[layer_id]['dW'] = dW.clone()
                velocities[layer_id]['db'] = db.clone()
                velocities[layer_id]['da'] = da.clone()
                
                self.network[layer_id]['W'] = W.clone()
                self.network[layer_id]['a'] = a.clone()
                self.network[layer_id]['b'] = b.clone()
                
                if readout:
                    
                    if (epoch + 1) % 1 == 0:
                        
                        readout_accuracy = self.get_readout(_Xtrain, _Xtest,
                                                            Ytrain, Ytest)
                        self.acc_profile[epoch, layer_id] = readout_accuracy
                        print(f'Readout accuracy = {readout_accuracy*100:.2f}')
                    #end
                #end
                
                self.loss_profile[epoch, layer_id] = train_loss.div(Xtrain.shape[0]).item()
                
                Xtrain = _Xtrain.clone()
                Xtest  = _Xtest.clone()
            #end FOR layers
            
            if self.num_discr and epoch in learning_params['EPOCHS_NDISCR']:
                
                # length of the last hidden layer
                Nfeat = self.network[-1]['b'].shape[1]
                
                for nclass in range(learning_params['NUM_LCLASSIFIERS']):
                    
                    print(f'LC {nclass}')
                    num_discr_dtls = learning_params['NDISCR_RANGES']
                    numbers_ref = list(num_discr_dtls.keys())
                    psycurves_splitted = dict()
                    
                    for nref in numbers_ref:
                        ''' Aggregate the outputs of linear classifier
                            so to pass them to a function to compute the
                            Weber fraction of the concatenated
                            ratios and percs
                        '''
                        
                        dr = DeltaRule(Nfeat, nref, learning_params)
                        lc_losses, lc_accs = dr.train(Xtrain, Ytrain)
                        ratios, percs = dr.test(Xtest, Ytrain)
                        psycurves_splitted.update( {nref : (ratios, percs)} )                        
                    #end
                    
                    Weber_frac = get_Weber_frac(psycurves_splitted)
                    self.Weber_fracs.at[epoch, nclass] = Weber_frac
                    self.psycurves.update({epoch : psycurves_splitted})
                    print(f'Weber fraction = {Weber_frac:.2f}')
                #end
            #end IF discr
            
        #end FOR epochs
    #end
    
    
    def train_fullstack(self, train_dataset, test_dataset, learning_params):
        
        momenta = [ learning_params['INIT_MOMENTUM'], 
                    learning_params['FINAL_MOMENTUM'] ]
        lr = learning_params['LEARNING_RATE']
        penalty = learning_params['WEIGHT_PENALTY']
        
        train_data = train_dataset['data']
        
        train_batches = train_data.shape[0]
        batch_size    = learning_params['BATCH_SIZE']
        
        activities = None
        topdown_activities = None
        
        velocities = list()
        for layer in self.network:
            velocities.append({
                'dW' : torch.zeros_like(layer['W']),
                'da' : torch.zeros_like(layer['a']),
                'db' : torch.zeros_like(layer['b'])
            })
        #end
        
        for epoch in range(self.epochs):
            
            print(f'Epoch {epoch:03d}')
            
            act_saved = dict()
            for layer_id, layer in enumerate(self.network):
                
                N_out = layer['W'].shape[1]
                act_saved.update({f'layer{layer_id}' : torch.zeros((train_batches,
                                                                    batch_size,
                                                                    N_out))})
            #end
            
            indices = list(range(train_batches))
            
            for layer_id, layer in enumerate(self.network):
                
                if layer_id == 0:
                    data = train_data.clone()
                else:
                    data = activities.clone()
                #end
                
                N_out = layer['W'].shape[1]
                activities = torch.zeros(train_batches, batch_size, N_out)
                
                for n in indices:
                    
                    v = data[n].clone()
                    activities[n] = self.sample(layer['W'], layer['b'], v)[0]
                #end
            #end
            
            for layer_id in range(self.network.__len__()):
                
                layer_id_true = self.network.__len__() - layer_id - 1
                
                if layer_id_true == self.network.__len__() - 1:
                    data = activities.clone()
                else:
                    data = topdown_activities.clone()
                #end
                
                N_in = self.network[layer_id_true]['W'].shape[0]
                topdown_activities = torch.zeros((train_batches, batch_size, N_in))
                
                for n in indices:
                    
                    topdown_activities[n] = self.sample(self.network[layer_id_true]['W'].t(),
                                                        self.network[layer_id_true]['a'], 
                                                        data[n])[0]
                    act_saved[f'layer{layer_id_true}'][n] = data[n]
                #end
            #end
            
            for layer_id, layer in enumerate(self.network):
                
                print(f'Layer {layer_id}')
                
                if layer_id == 0:
                    data = train_data.clone()
                else:
                    data = activities.clone()
                #end
                
                N_out = layer['W'].shape[1]
                activities = torch.zeros(train_batches, batch_size, N_out)
                
                W = layer['W'].clone(); dW = velocities[layer_id]['dW'].clone()
                a = layer['a'].clone(); da = velocities[layer_id]['da'].clone()
                b = layer['b'].clone(); db = velocities[layer_id]['db'].clone()
                
                indices = list(range(train_batches))
                train_loss = 0.
                
                # random.shuffle(indices)
                with tqdm(indices, unit = 'Batch') as tlayer:
                    for idx, n in enumerate(tlayer):
                        
                        tlayer.set_description(f'Layer {layer_id}')
                        batch_size = data[n].shape[0]
                        
                        pos_v = data[n].clone()
                        pos_ph, ph = self.sample(W, b, data[n].clone())
                        activities[n] = pos_ph.clone()
                        hidden_topdown = act_saved[f'layer{layer_id}'][n]
                        
                        neg_pv, neg_v = self.sample(W.t(), a, hidden_topdown)
                        neg_ph, neg_h = self.sample(W, b, neg_v)
                        
                        pos_dW = torch.matmul(pos_v.t(), pos_ph).div(batch_size)
                        pos_da = pos_v.mean(dim = 0)
                        pos_db = pos_ph.mean(dim = 0)
                        
                        neg_dW = torch.matmul(neg_v.t(), neg_ph).div(batch_size)
                        neg_da = neg_v.mean(dim = 0)
                        neg_db = neg_ph.mean(dim = 0)
                        
                        if epoch >= 5:
                            momentum = momenta[0]
                        else:
                            momentum = momenta[1]
                        #end
                        
                        dW = momentum * dW + lr * ((pos_dW - neg_dW) - penalty * W)
                        da = momentum * da + lr * (pos_da - neg_da)
                        db = momentum * db + lr * (pos_db - neg_db)
                        
                        W = W + dW; velocities[layer_id]['dW'] = dW.clone()
                        a = a + da; velocities[layer_id]['da'] = da.clone()
                        b = b + db; velocities[layer_id]['db'] = db.clone()
                        
                        mse = (pos_v - neg_pv).pow(2).mean()
                        train_loss += mse
                        
                        tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                    #end BATCHES
                #end WITH
                
                self.network[layer_id]['W'] = W.clone()
                self.network[layer_id]['a'] = a.clone()
                self.network[layer_id]['b'] = b.clone()
                
                self.loss_profile[epoch, layer_id] = train_loss.div(data.__len__()).item()
            #end LAYERS
            
        #end EPOCHS
    #end
    
    
    def test(self, test_dataset):
        
        Xtest  = test_dataset['data']
        
        reconstructions = torch.zeros_like(Xtest)
        _Xtest = torch.zeros_like(Xtest)
        test_loss = 0.
        
        for n in range(Xtest.shape[0]):
            
            hidden = Xtest[n,:,:].clone()
            
            for layer_id, layer in enumerate(self.network):
                
                hidden, _ = self.sample(layer['W'], layer['b'], hidden)
            #end
            
            reco = hidden.clone()
            
            for layer_id, layer in enumerate(reversed(self.network)):
                
                reco, _ = self.sample(layer['W'].t(), layer['a'], reco)
            #end
            
            reconstructions[n,:,:] = reco
            test_loss += (Xtest[n,:,:] - reco).pow(2).mean()
        #end
        
        test_loss = test_loss.div(Xtest.shape[0])
        print(f'Test MSE = {test_loss:.4f}')
        
        return reconstructions
    #end
    
    def sample(self, weight, bias, activity):
        
        # probabilities = torch.sigmoid( torch.matmul(activity, weight).add(bias) )
        probabilities = torch.sigmoid(torch.matmul(activity, weight).add(bias))
        activities = torch.bernoulli(probabilities)
        return probabilities, activities
    #end
    
    def Gibbs_sampling(self, pos_v, W, a, b):
                
        pos_ph, pos_h = self.sample(W, b, pos_v)
        neg_pv, neg_v = self.sample(W.t(), a, pos_h)
        neg_ph, neg_h = self.sample(W, b, neg_v)
        
        # p_h, h = self.sample(W, b, v)
        # p_v, v = self.sample(W.t(), a, h)
        # p_h, h = self.sample(W, b, v)
        
        return neg_ph, neg_v, neg_pv
    #end
    
    def get_readout(self, x_train, x_test, y_train, y_test):
        
        nfeat  = x_train[0].shape[-1]
        Xtrain = x_train.cpu().numpy().reshape(-1, nfeat)
        Ytrain = y_train.cpu().numpy().flatten()
        Xtest  = x_test.cpu().numpy().reshape(-1, nfeat)
        Ytest  = y_test.cpu().numpy().flatten()
        
        classifier = RidgeClassifier().fit(Xtrain, Ytrain)
        y_pred = classifier.predict(Xtest)
        
        return accuracy_score(Ytest, y_pred)
    #end
    
    def save(self, name = None):
        
        if name is None:
            name_save = ''
        else:
            name_save = name
        #end
        
        for layer in self.network:
            layer['W'].to(torch.device('cpu'))
            layer['a'].to(torch.device('cpu'))
            layer['b'].to(torch.device('cpu'))
        #end
        
        torch.save(self.to(torch.device('cpu')), open(os.path.join(self.path_model, 
                                           f'{name_save}_model.mdl'), 'wb'))
    #end
#end

