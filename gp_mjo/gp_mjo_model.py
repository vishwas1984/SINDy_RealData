#import math
import torch
import gpytorch
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
        self.preds = {key: None for key in keylist}
        self.lconfs = {key: None for key in keylist}
        self.uconfs = {key: None for key in keylist}

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
    def pred_mjo(self, data_name):

        model = self.model
        likelihood = self.likelihood
        dics = self.dics
        width = self.width
        n_pred = len(dics[data_name]['test']) - width

        observed_preds = np.zeros(n_pred)
        lower_confs = np.zeros(n_pred)
        upper_confs = np.zeros(n_pred)
        

        for i in range(n_pred):
            if i == 0:
                input_x = dics[data_name]['test'][i:width+i]
            else:
                t_end = dics[data_name]['test'][width+i-1]
                temp = input_x[1:]
                input_x = np.hstack((temp,t_end))
            
            test_x = torch.from_numpy(input_x.reshape((1, width))).float()
            # get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            # make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))

                lower, upper = observed_pred.confidence_region()
                observed_preds[i] = observed_pred.mean.numpy()
                lower_confs[i] = lower.detach().numpy()
                upper_confs[i] = upper.detach().numpy()

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

        n_pred = len(observed_preds) # n_pred = n_test - width #n_test = len(dics[data_name]['test'])
        id_test = self.dics_ids[data_name]['test']


        with torch.no_grad():

            # plot training data as black stars
            ax.plot(id_test[0:width], dics[data_name]['test'][:width], color='black', marker='^')
            ax.scatter(id_test[width:width+n_pred], dics[data_name]['test'][width:width+n_pred], color='black', marker='o')
            # Plot predictive means as blue line
            ax.plot(id_test[width:width+n_pred], observed_preds, color, linewidth=2, marker='x')
            if data_name == 'RMM1' or data_name == 'RMM2':
                # shade between the lower and upper confidence bounds
                ax.fill_between(id_test[width:width+n_pred], lower_confs, upper_confs, alpha=0.5, color=color)
                ax.legend(['starting interval', 'truth', 'predict', 'confidence'],fontsize=14)
            else:
                ax.legend(['starting interval', 'truth', 'predict'],fontsize=14)

    def rmm_to_phase(self, pred_rmm1= None, pred_rmm2=None):
        if pred_rmm1 is None:
            pred_rmm1 = self.preds['RMM1']
        if pred_rmm2 is None:
            pred_rmm2 = self.preds['RMM2']
        rmm_angle = np.arctan2(pred_rmm2,pred_rmm1) * 180 / np.pi
        phase = np.zeros(len(pred_rmm1))

        for i in range(8):
            lower_angle = - 180. * (1 - i / 4.)
            upper_angle = - 180. * (1 - (i+1) / 4.)
            bool_angle = (rmm_angle > lower_angle) & (rmm_angle <= upper_angle)
            phase += bool_angle.astype('int64')*(i+1)
       
        self.preds['phase'] = phase.astype(int)


    def rmm_to_amplitude(self, pred_rmm1= None, pred_rmm2=None):
        """ampltitude is the norm of (RMM1, RMM2)
        """
        if pred_rmm1 is None:
            pred_rmm1 = self.preds['RMM1']
        if pred_rmm2 is None:
            pred_rmm2 = self.preds['RMM2']
        amplitude = np.sqrt( np.square(pred_rmm1) + np.square(pred_rmm2) )
        self.preds['amplitude'] = amplitude


    ## Make predictions at specific time periods with a lead time
    def leadpred_mjo(self, data_name, n_pred, lead_time):
        """
        predict with fixed lead time
        """
        model = self.model
        likelihood = self.likelihood
        dics = self.dics
        width = self.width

        observed_preds = np.zeros((n_pred,lead_time))
        lower_confs = np.zeros((n_pred,lead_time))
        upper_confs = np.zeros((n_pred,lead_time))
        
        for i in range(n_pred):
            for j in range(lead_time):
                if j == 0:
                    input_x = dics[data_name]['test'][j+i:width+j+i]
                else:
                    t_end = dics[data_name]['test'][width+j+i-1]
                    temp = input_x[1:]
                    input_x = np.hstack((temp,t_end))
                
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
        self.n_pred4leadtime = n_pred
        self.pred_id4leadtime  = np.arange(width+lead_time, width+lead_time+n_pred)
        self.preds[data_name] = observed_preds
        self.lconfs[data_name] = lower_confs
        self.uconfs[data_name] = upper_confs


    def cor(self):
        """bivariate correlation coefficien
        """
        dics = self.dics
        pred_id = self.pred_id4leadtime
        
        preds_rmm1 = self.preds['RMM1'][:,-1]
        preds_rmm2 = self.preds['RMM2'][:,-1]

        obs_rmm1 = dics['RMM1']['test'][pred_id]
        obs_rmm2 = dics['RMM2']['test'][pred_id]

        numerator = np.dot(obs_rmm1, preds_rmm1) + np.dot(obs_rmm2, preds_rmm2)
        denominator = np.sqrt(np.dot(obs_rmm1, obs_rmm1) + np.dot(obs_rmm2, obs_rmm2)) \
            * np.sqrt(np.dot(preds_rmm1, preds_rmm1) + np.dot(preds_rmm2, preds_rmm2))
        cor = numerator / denominator
        self.cor_leadtime = cor
    
    def rmse(self):
        dics = self.dics
        pred_id = self.pred_id4leadtime
        n_pred = self.n_pred4leadtime
        
        preds_rmm1 = self.preds['RMM1'][:,-1]
        preds_rmm2 = self.preds['RMM2'][:,-1]

        obs_rmm1 = dics['RMM1']['test'][pred_id]
        obs_rmm2 = dics['RMM2']['test'][pred_id]

        sum_rmm1 = np.dot(obs_rmm1-preds_rmm1,obs_rmm1-preds_rmm1)
        sum_rmm2 = np.dot(obs_rmm2-preds_rmm2,obs_rmm2-preds_rmm2)
        rmse =np.sqrt( (sum_rmm1 + sum_rmm2) / n_pred )
        self.rmse_leadtime = rmse
