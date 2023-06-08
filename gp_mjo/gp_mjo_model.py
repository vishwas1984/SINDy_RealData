#import math
import torch
import gpytorch
from matplotlib import pyplot as plt
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
    def __init__(self, dic, kernel, width, n_iter, sigma_eps,fixed_noise) -> None:
        self.dic = dic
        self.kernel = kernel
        self.width = width
        self.n_iter = n_iter
        self.sigma_eps = sigma_eps
        self.fixed_noise = fixed_noise

    ## Training the model
    def train_mjo(self):

        dic = self.dic
        kernel = self.kernel
        width = self.width
        n_iter = self.n_iter
        sigma_eps = self.sigma_eps
        fixed_noise = self.fixed_noise

        input_x = np.vstack( ( rolling(dic['train1'][:-1], width) , rolling(dic['train2'][:-1], width) ) )
        output_y = np.concatenate( (dic['train1'][width:], dic['train2'][width:]), axis=None )

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
    def pred_mjo(self):

        model = self.model
        likelihood = self.likelihood
        dic = self.dic
        width = self.width
        n_pred = len(dic['test']) - width

        observed_preds = np.zeros(n_pred)
        lower_confs = np.zeros(n_pred)
        upper_confs = np.zeros(n_pred)

        for i in range(n_pred):
            if i == 0:
                input_x = dic['test'][i:width+i]
            else:
                t_end = dic['test'][width+i-1]
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
        
        self.observed_preds = observed_preds
        self.lower_confs = lower_confs
        self.upper_confs = upper_confs

    ## Plot the model fit
    def plot_mjo(self, ax, color):
        dic = self.dic
        width = self.width
        observed_preds = self.observed_preds
        lower_confs =self.lower_confs
        upper_confs = self.upper_confs
        n_test = len(dic['test'])
        n_pred = len(observed_preds) # n_pred = n_test - width


        with torch.no_grad():

            # plot training data as black stars
            ax.plot(np.arange(width), dic['test'][:width], color='black', marker='x')
            ax.scatter(np.arange(width,width+n_pred), dic['test'][width:width+n_pred], color='black', marker='o')
            # Plot predictive means as blue line
            ax.plot(np.arange(width,width+n_pred), observed_preds, color, linewidth=2)
            # shade between the lower and upper confidence bounds
            ax.fill_between(np.arange(width,width+n_pred), lower_confs, upper_confs, alpha=0.7, color=color)
            ax.legend(['starting interval', 'truth', 'predict', 'confidence'])


    ## Make predictions at specific time periods and with a lead time
    def pred_mjo(self, period_pred, lead_time):
        pass