
from tqdm import tqdm
import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeltaRule(torch.nn.Module):
    
    def __init__(self, Nin, Nref, learning_params):
        
        self.nref = np.int32(Nref)
        
        discr_range = learning_params['NDISCR_RANGES'][Nref]
        self.discr_range = discr_range
        self.W = torch.normal(0, 0.1, (Nin + 1, 1)) * 0.1
        
        self.lr = learning_params['LC_LEARNING_RATE']
        self.wd = learning_params['LC_WEIGHT_PENALTY']
        self.momenta = [learning_params['LC_INIT_MOMENTUM'], 
                        learning_params['LC_FINAL_MOMENTUM']]
        self.epochs = learning_params['LC_EPOCHS']
        
        self.num_batches = 10
        self.batch_size = 5120
    #end
    
    def get_accuracy(self, y_true, y_pred):
        
        if y_true.shape[0] == y_pred.shape[0]:
            lsize = y_true.shape[0]
        else:
            lsize = None
            raise ValueError('In get_accuracy : SIZE MISMATCH')
        #end
        
        y_true = self.binarize_labels(y_true)
        y_pred = self.binarize_pred(y_pred)
        return torch.sum(y_true == y_pred).float().div(lsize)
    #end
    
    def binarize_pred(self, pred):
        
        pred[pred < 0.5] = torch.Tensor([0.]).to(DEVICE)
        pred[pred > 0.5] = torch.Tensor([1.]).to(DEVICE)
        return pred
    #end
    
    def binarize_labels(self, labels):
        
        labels[labels <= self.nref] = torch.Tensor([0.]).to(DEVICE)
        labels[labels > self.nref]  = torch.Tensor([1.]).to(DEVICE)
        return labels
    #end
    
    def out_sigmoid(self, x):
        
        return torch.Tensor([1.]).div(1 + torch.exp( -torch.matmul(x, self.W) ))
    #end
    
    def train(self, x_train, y_train):
        
        losses = np.zeros(self.epochs)
        accs = np.zeros(self.epochs)
        dW = torch.zeros_like(self.W)
        
        x_train = torch.Tensor(torch.cat([batch for batch in x_train], dim = 0))
        y_train = torch.Tensor(torch.cat([batch for batch in y_train], dim = 0))
        x_train = x_train.reshape(self.num_batches, self.batch_size, -1)
        y_train = y_train.reshape(self.num_batches, self.batch_size, -1)
        
        with tqdm(range(self.epochs), unit = 'Epoch') as tepoch:
            for idx, epoch in enumerate(tepoch):
                
                tepoch.set_description(f'DeltaRule Epoch {epoch}')
                train_loss = 0.
                train_acc = 0.
                
                for n in range(self.num_batches):
                    
                    x = torch.cat((torch.ones(x_train[n,:,:].shape[0], 1),
                                         x_train[n,:,:]), dim = 1)
                    y = self.binarize_labels(y_train[n,:,:].clone())
                    
                    out = self.out_sigmoid(x)
                    out = self.binarize_pred(out.clone())
                    
                    if epoch < 5:
                        momentum = self.momenta[0]
                    else:
                        momentum = self.momenta[1]
                    #end
                    
                    weight_update = torch.matmul(x.t(), (y - out))
                    dW = momentum * dW + self.lr * (weight_update - self.wd * self.W)
                    self.W = self.W + dW
                    
                    train_loss += (y - out).pow(2).sum(dim = 1).mean(dim = 0).item()
                    train_acc += self.get_accuracy(out.clone(), y_train[n,:,:].clone())
                #end
                
                losses[epoch] = train_loss / self.num_batches
                accs[epoch] = train_acc / self.num_batches
                tepoch.set_postfix(MSE = losses[epoch])
            #end
        
        return losses, accs
    #end
    
    def test(self, x_test, y_test):
        
        nfeat = x_test[0].shape[-1]
        
        x_test = torch.Tensor(torch.cat([batch for batch in x_test], dim = 0))
        y_test = torch.Tensor(torch.cat([batch for batch in y_test], dim = 0))
        x_test = x_test.reshape(-1, nfeat)
        y_test = y_test.reshape(-1, 1)
        
        idx_range = ((y_test >= self.discr_range[0]) & 
                     (y_test <= self.discr_range[-1])).flatten()
        x_test = x_test[idx_range]
        y_test = y_test[idx_range]
        
        testset_len = x_test.shape[0]
        
        x_in = torch.cat((torch.ones(testset_len, 1), x_test), dim = 1)
        out = self.out_sigmoid(x_in)
        
        ratios = list()
        percs = list()
        
        for i in self.discr_range:
            
            mask = ( torch.Tensor([i]) == y_test )
            pred = self.binarize_pred(out[mask])
            ratios.append(i / self.nref)
            percs.append( torch.sum(pred).div(testset_len) ) # torch.mean(pred) ?
        #end
        
        return ratios, percs
    #end
#end


def get_Weber_frac(psycurves_splitted):
    
    def fit_function(ratios, Weber_frac):
        return norm.sf(0., loc = np.log(ratios), scale = np.sqrt(2) * Weber_frac)
    #end
    
    ratios8, percs8   = psycurves_splitted['8']
    ratios16, percs16 = psycurves_splitted['16']
    
    ratios8  = np.array(ratios8)
    ratios16 = np.array(ratios16)
    percs8  = np.array( [percs8[i].cpu().numpy() for i in range(ratios8.__len__())] )
    percs16 = np.array([percs16[i].cpu().numpy() for i in range(ratios16.__len__())] )
    
    ratios = np.sort( np.hstack((ratios8, ratios16)) )
    percs  = np.sort( np.hstack((percs8, percs16)) )
    
    pfit, _ = curve_fit(fit_function, ratios, percs, method = 'lm')
    return pfit[0]
#end