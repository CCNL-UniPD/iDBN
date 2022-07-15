
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
                
                t_activities = self.sample(W, b, t_data)
                
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
        
        train_data = train_dataset['data']
        train_lbls = train_dataset['labels']
        test_data  = test_dataset['data']
        test_lbls  = test_dataset['labels']
        
        train_batches = train_data.shape[0]
        test_batches  = test_data.shape[0]
        batch_size    = learning_params['BATCH_SIZE']
        
        activities = None
        t_activities = None
        
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
            
            for layer_id, layer in enumerate(self.network):
                
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
                t_activities, _ = self.sample(layer['W'], layer['b'], t_data)
                
                W = layer['W'].clone(); dW = velocities[layer_id]['dW'].clone()
                a = layer['a'].clone(); da = velocities[layer_id]['da'].clone()
                b = layer['b'].clone(); db = velocities[layer_id]['db'].clone()
                
                indices = list(range(train_batches))
                train_loss = 0.
                
                random.shuffle(indices)
                with tqdm(indices, unit = 'Batch') as tlayer:
                    for idx, n in enumerate(tlayer):
                        
                        tlayer.set_description(f'Layer {layer_id}')
                        batch_size = data[n].shape[0]
                        
                        # Propagation of activity before parameters update!!!
                        activities[n], _ = self.sample(W, b, data[n].clone())
                        
                        pos_v = data[n].clone()
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
                        
                        W = W + dW; velocities[layer_id]['dW'] = dW.clone()
                        a = a + da; velocities[layer_id]['da'] = da.clone()
                        b = b + db; velocities[layer_id]['db'] = db.clone()
                        
                        mse = (pos_v - neg_pv).pow(2).mean()
                        train_loss += mse
                        
                        tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                        
                    #end FOR batches
                #end WITH batches
                
                if readout:
                    
                    if (epoch + 1) % 1 == 0:
                        
                        readout_accuracy = self.get_readout(activities, t_activities,
                                                            train_lbls, test_lbls)
                        self.acc_profile[epoch, layer_id] = readout_accuracy
                        print(f'Readout accuracy = {readout_accuracy*100:.2f}')
                    #end
                #end
                
                self.network[layer_id]['W'] = W.clone()
                self.network[layer_id]['a'] = a.clone()
                self.network[layer_id]['b'] = b.clone()
                
                self.loss_profile[epoch, layer_id] = train_loss.div(data.__len__()).item()
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
                        lc_losses, lc_accs = dr.train(activities, train_lbls)
                        ratios, percs = dr.test(t_activities, test_lbls)
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
                
                N_in  = layer['W'].shape[0]
                N_out = layer['W'].shape[1]
                
                act_saved.update({ f'layer{layer_id}' : {
                        'pos_v'  : torch.zeros((train_batches, batch_size, N_in)),
                        'pos_h'  : torch.zeros((train_batches, batch_size, N_out)),
                        'pos_ph' : torch.zeros((train_batches, batch_size, N_out)),
                        'neg_v'  : torch.zeros((train_batches, batch_size, N_in)),
                        'neg_pv' : torch.zeros((train_batches, batch_size, N_in)) } 
                    })
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
                    p_h, h = self.sample(layer['W'], layer['b'], v)
                    act_saved[f'layer{layer_id}']['pos_v'][n]  = v    # pos v
                    act_saved[f'layer{layer_id}']['pos_h'][n]  = h    # pos h
                    act_saved[f'layer{layer_id}']['pos_ph'][n] = p_h  # pos ph
                    activities[n] = p_h.clone()
                #end
            #end
            
            for layer_id in range(self.network.__len__()):
                
                layer_id_true = self.network.__len__() - layer_id - 1
                
                for n in indices:
                    
                    h = act_saved[f'layer{layer_id_true}']['pos_h'][n].clone()
                    p_v, v = self.sample(self.network[layer_id_true]['W'].t(), 
                                         self.network[layer_id_true]['a'], h)
                    act_saved[f'layer{layer_id_true}']['neg_v'][n]  = v    # negative v
                    act_saved[f'layer{layer_id_true}']['neg_pv'][n] = p_v  # negative pv
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
                        
                        pos_ph = self.sample(W, b, data[n].clone())[0]
                        activities[n] = pos_ph.clone()
                        pos_v  = act_saved[f'layer{layer_id}']['pos_v'][n]
                        # pos_ph = act_saved[f'layer{layer_id}']['pos_ph'][n]
                        neg_v  = act_saved[f'layer{layer_id}']['neg_v'][n]
                        neg_pv = act_saved[f'layer{layer_id}']['neg_pv'][n]
                        
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
    
    
    def test(self, train_dataset, test_dataset):
        
        train_data = train_dataset['data']
        train_lbls = train_dataset['labels']
        test_data  = test_dataset['data']
        test_lbls  = test_dataset['labels']
        
        reconstructions = list()
        activities = [None for i in range(train_data.__len__())]
        t_activities = [None for i in range(test_data.__len__())]
        test_loss = 0.
        
        for n, batch in enumerate(test_data):
            
            reconstruction = None
            
            for layer_id, layer in enumerate(self.network):
                
                if layer_id == 0:
                    data = batch.clone()
                else:
                    data = t_activities[n].clone()
                #end
                
                t_activities[n], _ = self.sample(layer['W'], layer['b'], data)
            #end
            
            for layer_id, layer in enumerate(reversed(self.network)):
                
                if layer_id == 0:
                    data = t_activities[n].clone()
                else:
                    data = reconstruction.clone()
                #end
                
                reconstruction, _ = self.sample(layer['W'].t(), layer['a'], data)
            #end
            
            reconstructions.append(reconstruction)
            test_loss += (batch - reconstruction).pow(2).sum(dim = 1).mean(dim = 0)
        #end
        
        for n, batch in enumerate(train_data):
            
            for layer_id, layer in enumerate(self.network):
                
                if layer_id == 0:
                    data = batch.clone()
                else:
                    data = activities[n].clone()
                #end
                
                activities[n], _ = self.sample(layer['W'], layer['b'], data)
            #end
        #end
        
        readout_acc = self.get_readout(activities, t_activities, train_lbls, test_lbls)
        test_loss = test_loss.div(test_data.__len__())
        print(f'Test MSE = {test_loss:.4f}')
        print(f'Test readout = {readout_acc}')
        
        return reconstructions
    #end
    
    def sample(self, weight, bias, activity):
        
        probabilities = torch.sigmoid( torch.matmul(activity, weight).add(bias) )
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
        
        torch.save(self, open(os.path.join(self.path_model, 
                                           f'{name_save}_model.mdl'), 'wb'))
    #end
#end

