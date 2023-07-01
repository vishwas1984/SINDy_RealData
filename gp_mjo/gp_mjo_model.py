from typing import Optional, Tuple
from gpytorch.priors import Prior
import torch
import gpytorch
from gpytorch.constraints import Interval, Positive
import numpy as np
import os

from colorama import Fore, Back, Style


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
    def train_mjo(self, data_name=None, Depend=False, season=False):

        dics = self.dics
        kernel = self.kernel
        width = self.width
        n_iter = self.n_iter
        sigma_eps = self.sigma_eps
        fixed_noise = self.fixed_noise

        print(Back.WHITE + Fore.BLACK + 'start preparing for training...' + Style.RESET_ALL)
        input_x = np.array([]).reshape((-1,width))
        output_y = np.array([])
        rmms = ['RMM1','RMM2'] if Depend else [data_name]
        for rmm in rmms:
            for train_set in ['train1', 'train2']:
                if season:
                    train_id_split = np.hstack( ( np.array([0]), 
                                np.where(np.ediff1d(dics['id'][train_set]) != 1 )[0]+1, 
                                np.array([len(dics['id'][train_set])]) ) )
                    diff_train_ids = np.ediff1d(train_id_split)
                    for i, diff_train_id in enumerate(diff_train_ids):
                        if diff_train_id <= width:
                            print(f'width = {width} is greater than the current interval width {diff_train_id}, will skip {i}-th iteration in {train_set} for {rmm}')
                            continue
                        split_start = train_id_split[i]
                        split_end = train_id_split[i+1]
                        train_i = dics[rmm][train_set][split_start:split_end]
                else:
                    train_i = dics[rmm][train_set] 

                input_x = np.vstack(( input_x, rolling(train_i[:-1],width) ))
                output_y = np.concatenate( (output_y, train_i[width:]), axis=None)
        print(Back.WHITE + Fore.BLACK + 'training data setting is done.' + Style.RESET_ALL)
        print() 
        print('data is trained on 4 seasons resepectively') if season else print('data is trained on the entire dataset') 
        print('RMM1 and RMM2 are' + Fore.GREEN + ' dependent' + Style.RESET_ALL + ', input data incorporate RMM1 and RMM2') if Depend else print('RMM1 and RMM2 are' + Fore.GREEN + ' independent' + Style.RESET_ALL + f', input data only incorporate {data_name}')
        print(f'the width of the rolling window is {width}')
        print('the shape of the' + Fore.GREEN + ' input/predictor ' + Style.RESET_ALL + f'is {input_x.shape}')
        print('the shape of the' + Fore.GREEN + ' observation ' + Style.RESET_ALL + f'is {output_y.shape}')
        print()

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

        print(Back.YELLOW + Fore.BLACK + 'start training...' + Style.RESET_ALL)

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

        print(Back.YELLOW + Fore.BLACK + 'training step is done.' + Style.RESET_ALL)
        print()

    ## Make predictions with the model
    def pred_mjo(self, data_name=None, lead_time=None, n_pred=1, Depend=False, season=False):

        model = self.model
        likelihood = self.likelihood
        dics = self.dics
        width = self.width

        print(Back.WHITE + Fore.BLACK + 'start preparing for testing...' + Style.RESET_ALL)
        test_id_split = np.hstack( ( np.array([0]), 
                            np.where(np.ediff1d(dics['id']['test']) != 1 )[0]+1, 
                            np.array([len(dics['id']['test'])]) ) )
        diff_test_ids = np.ediff1d(test_id_split)
        max_diff = np.max(diff_test_ids)
        freq_diff = np.bincount(diff_test_ids).argmax() # return the most frequent value in diff_test_ids

        if lead_time is None or n_pred == 1:
            lead_time = max_diff - width if season else len(dics['RMM1']['test']) - width
    
        if season or len(test_id_split) > 2:
            freq_diff_id = np.where( diff_test_ids >= freq_diff )[0]
            pred_ids = test_id_split[freq_diff_id]
            if width >= freq_diff:
                raise ValueError(f'the width is greater than the season interval, please try a width value < {freq_diff}')
            
            if n_pred > len(pred_ids):
                print(f"the number of predictions is greater than the number of the season intervals..., will set n_pred = {len(pred_ids)}")
                n_pred = len(pred_ids)
            
            if lead_time + width > freq_diff:
                print(f"the sum of the width and lead time is greater than the season interval..., will set lead time = {freq_diff-width}")
                lead_time = freq_diff-width
        else:
            pred_ids = np.arange(n_pred)
        print(Back.WHITE + Fore.BLACK + 'test data setting is done.' + Style.RESET_ALL)
        print()
        print('test data incorporate RMM1 and RMM2') if (Depend or data_name is None) else print(f'test data only incorporate {data_name}')
        print('the number of' + Fore.GREEN + ' predictions ' + Style.RESET_ALL + f'is n_pred = {n_pred}, the' + Fore.GREEN + ' maximal lead time ' + Style.RESET_ALL + f'is lead_time = {lead_time}')
        print('the shape of the' + Fore.GREEN + ' total observations ' + Style.RESET_ALL + f'is {(n_pred,lead_time)}')
        print()

        print(Back.YELLOW + Fore.BLACK + 'start testing...' + Style.RESET_ALL)
        obs = {}
        observed_preds = {}
        lower_confs = {}
        upper_confs = {}
        
        obs['amplitude'] = np.zeros((n_pred,lead_time))
        rmms = ['RMM1','RMM2'] if (Depend or data_name is None) else [data_name]
        for rmm in rmms:
            obs[rmm] = np.zeros((n_pred,lead_time))
            observed_preds[rmm] = np.zeros((n_pred,lead_time))
            lower_confs[rmm] = np.zeros((n_pred,lead_time))
            upper_confs[rmm] = np.zeros((n_pred,lead_time))
        
        for i, pred_i in enumerate(pred_ids):
            input_x_ij = {}
            for j in range(lead_time):
                input_x = np.array([]).reshape((-1,width))
                obs['amplitude'][i,j] = dics['amplitude']['test'][pred_i+j+width]
                for rmm in rmms:
                    obs[rmm][i,j] = dics[rmm]['test'][pred_i+j+width]
                    if j == 0:
                        input_x_ij[rmm] = dics[rmm]['test'][pred_i+j : pred_i+j+width]
                    else:
                        t_last_pred_x = observed_preds[rmm][i,j-1]
                        input_x_ij[rmm] = np.hstack(( input_x_ij[rmm][1:], t_last_pred_x ))

                    input_x = np.vstack((input_x, input_x_ij[rmm]))
                    
                test_x = torch.from_numpy(input_x.reshape((-1, width))).float()
                # get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()

                # make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(test_x))
                    lower, upper = observed_pred.confidence_region()
                
                for k, rmm in enumerate(rmms):
                    observed_preds[rmm][i,j] = observed_pred.mean.numpy()[k]
                    lower_confs[rmm][i,j] = lower.detach().numpy()[k]
                    upper_confs[rmm][i,j] = upper.detach().numpy()[k]
        
        self.lead_time = lead_time
        self.n_pred = n_pred
        self.obs['amplitude'] = obs['amplitude']
        for rmm in rmms:
            self.obs[rmm] = obs[rmm]
            self.preds[rmm] = observed_preds[rmm]
            self.lconfs[rmm] = lower_confs[rmm]
            self.uconfs[rmm] = upper_confs[rmm]

        print(Back.YELLOW + Fore.BLACK + 'test step is done.' + Style.RESET_ALL) 
        print()


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


# import sys
# sys.path.append('../')
# import random
# import pickle
# import torch
# import gpytorch
# from gp_mjo_model import gp_mjo
# from utils.dat_ops import dics_divide

# from prettytable import PrettyTable
# from matplotlib import pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.markers import MarkerStyle

# npzfile = np.load('/Users/hchen/SINDy_RealData/data/mjo_new_data.npz', allow_pickle=True)

# ## Divide new_datas into four seasons
# npz_month = npzfile['month']
# winter_ids = np.where( (npz_month==12) | (npz_month==1) | (npz_month==2) )[0]
# spring_ids = np.where( (npz_month==3) | (npz_month==4) | (npz_month==5) )[0]
# summer_ids = np.where( (npz_month==6) | (npz_month==7) | (npz_month==8) )[0]
# fall_ids = np.where( (npz_month==9) | (npz_month==10) | (npz_month==11) )[0]

# seasons = ['winter','spring','summer','fall']
# seasons_ids = [winter_ids, spring_ids, summer_ids, fall_ids]
# data_names = npzfile.files
# data_names.append('id')
# n_files = len(data_names)

# season_datas = {}
# for j in range(4):
#     season = seasons[j]
#     season_id = seasons_ids[j]

#     new_datas = [0]*n_files
#     for i in range(n_files):
#         if i < n_files-1:
#             new_datas[i] = npzfile[data_names[i]][season_id]
#         if i == n_files-1:
#             new_datas[i] = seasons_ids[j]

#     season_datas[season] = new_datas


# ## Set initial values
# widths = [40]
# n_iter = 200
# sigma_eps = 0.01
# fixed_noise = False

# Ns = [len(winter_ids),len(spring_ids),len(summer_ids),len(fall_ids)]# the total number of days in new dataset
# n = 2000 # the number of days for training
# c = 365 # the number of dropped buffer set
# ms = [N-n-c for N in Ns] # the number of days for testing


# n_cv = 1 # the number of operations for cross-validation
# n1s  = [random.randint(0,n) for i in range(n_cv)]
# n1s = [1165]

# ## Set the kernel of GP
# nu = 0.5 # 1.5,2.5.... smoothness parameter of Matern kernel
# d = 1 # d = width or d = 1
# kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=d)

# lead_time = 60
# n_pred = 20 # 14*365

# from IPython.display import display, Markdown
# from kernels.block_kernel import CustomBlockKernel
# nu =0.5
# kernel = CustomBlockKernel(nu=nu, num_paras = 1, para_upper=2.0)

# # Dependent
# dics_total_maternblock = {}
# cor_total_maternblock = {}
# rmse_total_maternblock = {}
# phase_err_total_maternblock = {}
# amplitude_err_total_maternblock = {}

# t = PrettyTable(["season", "width", "RMMs", "lengthscale", "parameter"])

# for m, season in zip(ms,seasons):
#     print(Back.RED + Fore.BLUE + '=======================================' + Style.RESET_ALL)
#     print(Back.RED + Fore.BLUE + f'train and test for the season = {season}:' + Style.RESET_ALL)
#     print(Back.RED + Fore.BLUE + '=======================================' + Style.RESET_ALL)
#     cor_n1 = {}
#     rmse_n1 = {}
#     phase_err_n1 = {}
#     amplitude_err_n1 = {}
#     for n1 in n1s:
#         dics, dics_ids = dics_divide(season_datas[season], data_names, n1, m, n, c)
#         dics_total_maternblock[n1] = dics

#         cor_width = {}
#         rmse_width = {}
#         phase_err_width = {}
#         amplitude_err_width = {}
#         for width in widths:
#             mjo_model = gp_mjo(dics, dics_ids, kernel, width, n_iter, sigma_eps,fixed_noise)
#             mjo_model.train_mjo(Depend=True, season=True)
#             mjo_model.pred_mjo(lead_time=lead_time, n_pred=n_pred, Depend=True, season=True)
#             t.add_rows( [[f'{season}', f'{width}', 'dependent', 
#                           f'{mjo_model.model.covar_module.base_kernel.lengthscale.detach().numpy()}', 
#                           f'{mjo_model.model.covar_module.base_kernel.paras.detach().numpy()}']] )

#             # compute errors
#             cor_width[width] = mjo_model.cor()
#             rmse_width[width] = mjo_model.rmse()
#             phase_err_width[width] = mjo_model.phase_err()
#             amplitude_err_width[width] = mjo_model.amplitude_err()
            
#         cor_n1[n1] = cor_width
#         rmse_n1[n1] = rmse_width
#         phase_err_n1[n1] = phase_err_width
#         amplitude_err_n1[n1] = amplitude_err_width
    
#     cor_total_maternblock[season] = cor_n1
#     rmse_total_maternblock[season] = rmse_n1
#     phase_err_total_maternblock[season] = phase_err_n1
#     amplitude_err_total_maternblock[season] = amplitude_err_n1

# maternblock = {'cor': cor_total_maternblock, 'rmse': rmse_total_maternblock, 
#                      'phase': phase_err_total_maternblock, 'amplitude': amplitude_err_total_maternblock,
#                      'paras': t.get_string()}
# # dic_pkl = open('../data/preds/season/maternblock.pkl','wb')
# # pickle.dump(maternblock, dic_pkl)

# display(Markdown(rf'$ Block Matern {nu} kernel with dependent RMMs:'))
# print(t)
