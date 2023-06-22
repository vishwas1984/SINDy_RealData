from typing import Optional, Tuple
from gpytorch.priors import Prior
import torch
import gpytorch
from gpytorch.constraints import Interval, Positive
import numpy as np
import os

## Construct training data and test data for GP 
def rolling(a:np.ndarray, width) -> np.ndarray:
    """
    convert a = [a_1,...,a_n] to a (n-wid+1)*(wid) matrix with the rolling window
    a_roll = [a_1,a_2,...,a_{wid};
              a_2,a_3,...,a_{wid+1};
              ...;
              a_{n-wid+1},a_{n-wid+2},...,a_n]

    ------------------------
    Parameters:
    a: [a_1,...,a_n], 1*n numpy array
    width: the width of the rolling window

    ------------------------
    Returns:
    a_roll: [a_1,a_2,...,a_{wid};
             a_2,a_3,...,a_{wid+1};
             ...;
             a_{n-wid+1},a_{n-wid+2},...,a_n], is a (n-wid+1)*(wid) numpy array
    """
    if a.size == 0:
        a_roll = a.reshape((-1,width))
    else:
        shape = (a.size - width + 1, width) # a.size = np.prod(a.shape)
        strides = (a.itemsize, a.itemsize) # itemsize returns the memory size of one element in bytes (8 bytes for a float64)
        a_roll = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_roll


## Exact GP Inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

### gp_mjo class
class gp_mjo:
    ## Initialization
    def __init__(self, dics, dics_ids, kernel, width, n_iter, sigma_eps,fixed_noise) -> None:
        self.dics = dics
        self.dics_ids = dics_ids
        self.kernel = kernel
        self.width = width
        self.n_iter = n_iter
        self.sigma_eps = sigma_eps
        self.fixed_noise = fixed_noise

        keylist = ['RMM1', 'RMM2', 'phase', 'amplitude']
        errlist = ['cor','rmse','phase','amplitude']
        self.obs = {key: None for key in keylist}
        self.preds = {key: None for key in keylist}
        self.lconfs = {key: None for key in keylist}
        self.uconfs = {key: None for key in keylist}
        self.errs = {key: None for key in errlist}

    ## Training the model
    def train_mjo(self, data_name=None, Depend=False):

        dics = self.dics
        kernel = self.kernel
        width = self.width
        n_iter = self.n_iter
        sigma_eps = self.sigma_eps
        fixed_noise = self.fixed_noise

        if Depend:
            input_x1 = np.vstack( ( rolling(dics['RMM1']['train1'][:-1], width) , rolling(dics['RMM1']['train2'][:-1], width) ) )
            output_y1 = np.concatenate( (dics['RMM1']['train1'][width:], dics['RMM1']['train2'][width:]), axis=None )
            input_x2 = np.vstack( ( rolling(dics['RMM2']['train1'][:-1], width) , rolling(dics['RMM2']['train2'][:-1], width) ) )
            output_y2 = np.concatenate( (dics['RMM2']['train1'][width:], dics['RMM2']['train2'][width:]), axis=None )
            
            input_x = np.vstack( (input_x1,input_x2) )
            output_y = np.concatenate( (output_y1,output_y2), axis=None )
        else:
            # RMM1 and RMM2 are independent
            input_x = np.vstack( ( rolling(dics[data_name]['train1'][:-1], width) , rolling(dics[data_name]['train2'][:-1], width) ) )
            output_y = np.concatenate( (dics[data_name]['train1'][width:], dics[data_name]['train2'][width:]), axis=None )

        train_x = torch.from_numpy(input_x).float()
        train_y = torch.from_numpy(output_y).float()

        # initialize likelihood and model
        if fixed_noise:
            noises = torch.ones(train_x.size(dim=0)) * sigma_eps
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=False)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood, kernel)

        # this is for running the notebook in our testing framework
        smoke_test = ('CI' in os.environ)
        training_iter = 2 if smoke_test else n_iter

        # find optimal model hyperparameters
        model.train()
        likelihood.train()

        # use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # zero gradients from previous iteration
            optimizer.zero_grad()
            
            # output from model
            output = model(train_x)
            
            # calc loss and backprop gradients
            loss = (-1) * mll(output, train_y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()
        
        self.model = model
        self.likelihood = likelihood


    ## Make predictions with the model
    def pred_mjo(self, data_name, lead_time=None, n_pred=1):

        model = self.model
        likelihood = self.likelihood
        dics = self.dics
        width = self.width
        if lead_time is None or n_pred == 1:
            lead_time = len(dics[data_name]['test']) - width

        observed_preds = np.zeros((n_pred,lead_time))
        lower_confs = np.zeros((n_pred,lead_time))
        upper_confs = np.zeros((n_pred,lead_time))
        
        for i in range(n_pred):
            for j in range(lead_time):
                if j == 0:
                    input_x = dics[data_name]['test'][j+i:width+j+i]
                else:
                    t_last_pred = observed_preds[i,j-1]
                    temp = input_x[1:]
                    input_x = np.hstack((temp,t_last_pred))
                
                test_x = torch.from_numpy(input_x.reshape((1, width))).float()
                # get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()

                # make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(test_x))
                    lower, upper = observed_pred.confidence_region()

                    observed_preds[i,j] = observed_pred.mean.numpy()
                    lower_confs[i,j] = lower.detach().numpy()
                    upper_confs[i,j] = upper.detach().numpy()

        self.lead_time = lead_time
        self.n_pred = n_pred
        self.preds[data_name] = observed_preds
        self.lconfs[data_name] = lower_confs
        self.uconfs[data_name] = upper_confs



    ## Plot the model fit
    def plot_mjo(self, data_name, ax, color):
        dics = self.dics
        width = self.width
        observed_preds = self.preds[data_name]
        lower_confs = self.lconfs[data_name]
        upper_confs = self.uconfs[data_name]

        lead_time = self.lead_time#len(observed_preds) # n_pred = n_test - width #n_test = len(dics[data_name]['test'])
        id_test = self.dics_ids[data_name]['test']


        with torch.no_grad():

            # plot training data as black stars
            ax.plot(id_test[0:width], dics[data_name]['test'][:width], color='black', marker='^')
            ax.plot(id_test[width:width+lead_time], dics[data_name]['test'][width:width+lead_time], color='black', marker='o')
            # Plot predictive means as blue line
            ax.plot(id_test[width:width+lead_time], observed_preds.reshape(-1), color, linewidth=2, marker='x')
            if data_name == 'RMM1' or data_name == 'RMM2':
                # shade between the lower and upper confidence bounds
                ax.fill_between(id_test[width:width+lead_time], lower_confs.reshape(-1), upper_confs.reshape(-1), alpha=0.5, color=color)
                ax.legend(['starting interval', 'truth', 'predict', 'confidence'],fontsize=14)
            else:
                ax.legend(['starting interval', 'truth', 'predict'],fontsize=14)

    def rmm_to_phase(self, pred_rmm1=None, pred_rmm2=None):
        if pred_rmm1 is None:
            pred_rmm1 = self.preds['RMM1']
        if pred_rmm2 is None:
            pred_rmm2 = self.preds['RMM2']
        rmm_angle = np.arctan2(pred_rmm2,pred_rmm1) * 180 / np.pi
        phase = np.zeros(pred_rmm1.shape)

        for i in range(8):
            lower_angle = - 180. * (1 - i / 4.)
            upper_angle = - 180. * (1 - (i+1) / 4.)
            bool_angle = (rmm_angle > lower_angle) & (rmm_angle <= upper_angle)
            phase += bool_angle.astype('int64')*(i+1)
       
        self.preds['phase'] = phase.astype(int)
        return self.preds['phase']


    def rmm_to_amplitude(self, pred_rmm1=None, pred_rmm2=None):
        """ampltitude is the norm of (RMM1, RMM2)
        """
        if pred_rmm1 is None:
            pred_rmm1 = self.preds['RMM1']
        if pred_rmm2 is None:
            pred_rmm2 = self.preds['RMM2']
        amplitude = np.sqrt( np.square(pred_rmm1) + np.square(pred_rmm2) )
        self.preds['amplitude'] = amplitude
        return amplitude

    def obs_extract(self):
        dics = self.dics
        width = self.width
        lead_time = self.lead_time
        n_pred = self.n_pred

        rmm1_temp = dics['RMM1']['test'][width: width+lead_time+n_pred-1]
        rmm2_temp = dics['RMM2']['test'][width: width+lead_time+n_pred-1]
        amplitude_temp = dics['amplitude']['test'][width: width+lead_time+n_pred-1]
        
        self.obs['RMM1'] = rolling(rmm1_temp, lead_time) # n_pred*lead_time numpy array
        self.obs['RMM2'] = rolling(rmm2_temp, lead_time) # n_pred*lead_time numpy array
        self.obs['amplitude'] = rolling(amplitude_temp, lead_time)

    
    def cor(self):
        """bivariate correlation coefficien
        """
        pred_rmm1 = self.preds['RMM1']
        pred_rmm2 = self.preds['RMM2']

        obs_rmm1 = self.obs['RMM1']
        obs_rmm2 = self.obs['RMM2']

        numerator = np.sum((obs_rmm1*pred_rmm1 + obs_rmm2*pred_rmm2), axis=0) # 1*lead_time numpy array
        denominator = np.sqrt(np.sum((obs_rmm1**2 + obs_rmm2**2),axis=0)) \
            * np.sqrt(np.sum((pred_rmm1**2 + pred_rmm2**2),axis=0))
        self.errs['cor'] = (numerator / denominator).reshape(-1) # shape = (lead_time,) numpy array

        return self.errs['cor']
    
   
    def rmse(self):
        pred_rmm1 = self.preds['RMM1']
        pred_rmm2 = self.preds['RMM2']

        obs_rmm1 = self.obs['RMM1']
        obs_rmm2 = self.obs['RMM2']
        n_pred = pred_rmm1.shape[0]

        sum_rmm1 = np.sum((obs_rmm1-pred_rmm1)**2, axis=0)
        sum_rmm2 = np.sum((obs_rmm2-pred_rmm2)**2, axis=0)
        self.errs['rmse'] = ( np.sqrt( (sum_rmm1 + sum_rmm2) / n_pred ) ).reshape(-1)

        return self.errs['rmse']


    def phase_err(self):
        pred_rmm1 = self.preds['RMM1']
        pred_rmm2 = self.preds['RMM2']

        obs_rmm1 = self.obs['RMM1']
        obs_rmm2 = self.obs['RMM2']
        n_pred = pred_rmm1.shape[0]

        num = obs_rmm1*pred_rmm2 - obs_rmm2*pred_rmm1
        den = obs_rmm1*pred_rmm1

        temp = np.arctan(np.divide(num,den))
        self.errs['phase'] = ( np.sum(temp, axis=0) / n_pred ).reshape(-1)

        return self.errs['phase']

   
    def amplitude_err(self):
        pred_rmm1 = self.preds['RMM1']
        pred_rmm2 = self.preds['RMM2']
        n_pred = pred_rmm1.shape[0]
        preds_amplitude = self.rmm_to_amplitude(pred_rmm1,pred_rmm2)
        
        obs_amplitude = self.obs['amplitude']

        self.errs['amplitude'] = (np.sum((preds_amplitude - obs_amplitude),axis=0) / n_pred).reshape(-1)

        return self.errs['amplitude'] 