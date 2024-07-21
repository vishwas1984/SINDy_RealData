import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, FixedLocator
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.animation as animation

import seaborn as sns
import calendar
from .empgp_mjo_model import EmpGPMJO
from .nn_mjo_model import NNMJO
from .nn_models import FFNNModel

class EmpGPMJOPred:
    def __init__(self, npzfile, s2s_data=None, choose_dir_name='bom', widths = [40,60],
                n = 10000, v = 2000, m = 5000, lead_time = 60, n_pred=None,
                start_train = 0, n_offset = 0, min_dir_len=528) -> None:
        
        self.npzfile = npzfile
        self.s2s_data = s2s_data
        self.choose_dir_name = choose_dir_name
        if self.choose_dir_name == 'ecmwf_txt':
            self.choose_dir_len = len(s2s_data[self.choose_dir_name]['ensemble_mean']['S'])
        else:
            self.choose_dir_len = len(s2s_data[self.choose_dir_name]['ensemble_mean_rmm1.nc']['S'])

        self.widths = widths
        self.n = n
        self.v = v
        self.m = m
        self.lead_time = lead_time
        if n_pred is None:
            self.n_pred = min(365*5, min_dir_len)
        else:
            self.n_pred = n_pred
        self.start_train = start_train
        self.n_offset = n_offset
        

        self.cor_entire = {}
        self.rmse_entire = {}
        self.phase_err_entire = {}
        self.amplitude_err_entire = {}
        self.hss_entire = {}
        self.hss_n_entire = {}
        self.hss_signif_entire = {}
        self.crps_entire = {}
        self.mll_entire = {}


        self.K11_entire = {}
        self.K22_entire = {}
        self.dist_joint = {}
        self.train_jointcov = {}
        self.obs = {}
        self.observed_preds = {}
        self.lower_confs = {}
        self.upper_confs = {}
        self.pred_covs = {}

        self.n_trains = {}
        self.n_vals = {}
        self.start_vals = {}
        self.start_tests = {}
        self.test_ids = {}

    def pred(self, test_on_choose_dir=True, test_ids=None):

        npzfile = self.npzfile
        s2s_data = self.s2s_data

        for width in self.widths:
            self.n_trains[width] = width + self.n
            self.start_vals[width] = self.start_train + self.n_trains[width] + self.n_offset #- v - width - 1
            self.n_vals[width] = width + self.v
            #start_tests[width] = start_vals[width] + n_vals[width] + n_offset
            self.start_tests[width] = self.start_train + max(self.widths) + self.n + self.n_offset + max(self.widths) \
                + self.v + self.n_offset + 10 + max(self.widths) - width
            if test_on_choose_dir:
                if self.choose_dir_name == 'ecmwf_txt':
                    start_dates = s2s_data[self.choose_dir_name]['ensemble_mean']['S'][:self.n_pred, 0]
                else:
                    start_dates = s2s_data[self.choose_dir_name]['ensemble_mean_rmm1.nc']['S'][ : self.n_pred]
                self.test_ids[width] = start_dates - width
            else:
                if test_ids is not None:
                    self.test_ids[width] = test_ids - width

            emp_model = EmpGPMJO(npzfile=npzfile, width=width, lead_time=self.lead_time, n=self.n, start_train=self.start_train)
            emp_model.get_emp()
            emp_model.get_biasvar(start_val=self.start_vals[width], n_pred=365*4, v=self.v, season_bool=False)
            if test_on_choose_dir:
                emp_model.pred(test_ids=self.test_ids[width], n_pred=len(start_dates), m=self.m, season_bool=False) # test on the same dates as choose_dir_name
            else:
                emp_model.pred(start_test=self.start_tests[width], n_pred=self.n_pred, m=self.m, season_bool=False) # test on the consecutive dates

            # compute errors
            self.cor_entire[width] = emp_model.cor()
            self.rmse_entire[width] = emp_model.rmse()
            self.phase_err_entire[width] = emp_model.phase_err()
            self.amplitude_err_entire[width] = emp_model.amplitude_err()
            self.hss_entire[width] = emp_model.hss()
            self.hss_n_entire[width] = emp_model.errs['hss_n']
            self.hss_signif_entire[width] = emp_model.errs['hss_signif']
            self.crps_entire[width] = emp_model.crps()
            self.mll_entire[width] = emp_model.mll()


            self.obs[width] = emp_model.obs
            self.observed_preds[width] = emp_model.observed_preds
            self.lower_confs[width] = emp_model.lower_confs
            self.upper_confs[width] = emp_model.upper_confs
            self.pred_covs[width] = emp_model.pred_covs # [n_pred, lead_time, 2, 2]

            self.emp_model = emp_model
    

    def add_s2s(self, dir_names, hdate_id=19, era5_obs=True, n_pred=None):
        
        self.era5_obs = era5_obs
        npzfile = self.npzfile
        s2s_data = self.s2s_data
        emp_model = self.emp_model
        if n_pred is None:
            n_pred = self.n_pred

        for dir_name in dir_names:
            if dir_name == 'ecmwf_txt':
                lead_time_ensemble_mean = s2s_data[dir_name]['ensemble_mean']['RMM1(forecast)'].shape[1]
                lead_time_ensembles = s2s_data[dir_name]['ensembles']['RMM1(forecast)'].shape[1]
                # lead_time = min (lead_time_ensemble_mean, lead_time_ensembles )
                lead_time = lead_time_ensembles
            else:
                lead_time = len(s2s_data[dir_name]['ensemble_mean_rmm1.nc']['L'])

            if dir_name == 'ecmwf_txt':
                
                # pred_mean_rmm1 = s2s_data[dir_name]['ensemble_mean']['RMM1(forecast)'][-n_pred:, :lead_time] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                # pred_mean_rmm2 = s2s_data[dir_name]['ensemble_mean']['RMM2(forecast)'][-n_pred:, :lead_time] # [n_pred, lead_time]

                ensembles_rmm1 = s2s_data[dir_name]['ensembles']['RMM1(forecast)'][:n_pred, :lead_time, :] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2 = s2s_data[dir_name]['ensembles']['RMM2(forecast)'][:n_pred, :lead_time, :] # [n_pred, lead_time, num_ensembles]

                pred_mean_rmm1 = ensembles_rmm1.mean(axis=-1) # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean_rmm2 = ensembles_rmm2.mean(axis=-1) # [n_pred, lead_time]

                pred_std_rmm1 = ensembles_rmm1.std(axis=-1) # [n_pred, lead_time]
                pred_std_rmm2 = ensembles_rmm2.std(axis=-1) # [n_pred, lead_time]


                ensembles_rmm1_norm = ensembles_rmm1 - ensembles_rmm1.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2_norm = ensembles_rmm2 - ensembles_rmm2.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1) # [n_pred, lead_time]
            
                start_date_ids = s2s_data[dir_name]['ensemble_mean']['S'][:,0]

            elif dir_name in ['ecmwf', 'eccc']:
                
                pred_mean_rmm1 = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][hdate_id][:n_pred, :] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean_rmm2 = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][hdate_id][:n_pred, :] # [n_pred, lead_time]

                ensembles_rmm1 = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][hdate_id][:n_pred, :, :] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2 = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][hdate_id][:n_pred, :, :] # [n_pred, lead_time, num_ensembles]

                pred_std_rmm1 = ensembles_rmm1.std(axis=-1) # [n_pred, lead_time]
                pred_std_rmm2 = ensembles_rmm2.std(axis=-1) # [n_pred, lead_time]

                ensembles_rmm1_norm = ensembles_rmm1 - ensembles_rmm1.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2_norm = ensembles_rmm2 - ensembles_rmm2.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1) # [n_pred, lead_time]

                start_date_ids = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['S'][:n_pred]

            else:
                pred_mean_rmm1 = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][:n_pred, :] 
                pred_mean_rmm2 = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][:n_pred, :]
                
                ensembles_rmm1 = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][:n_pred, :, :]
                ensembles_rmm2 = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][:n_pred, :, :]
                pred_std_rmm1 = ensembles_rmm1.std(axis=-1)
                pred_std_rmm2 = ensembles_rmm2.std(axis=-1)

                ensembles_rmm1_norm = ensembles_rmm1 - ensembles_rmm1.mean(axis=-1)[..., None]
                ensembles_rmm2_norm = ensembles_rmm2 - ensembles_rmm2.mean(axis=-1)[..., None]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1)

                start_date_ids = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['S'][:n_pred]
            
            if dir_name == 'ecmwf_txt' and era5_obs:
                obs_rmm1 = s2s_data[dir_name]['ensembles']['RMM1(obs)'][:n_pred, :lead_time]
                obs_rmm2 = s2s_data[dir_name]['ensembles']['RMM2(obs)'][:n_pred, :lead_time]
            else:
                n_pred_min = n_pred if n_pred <= len(pred_mean_rmm1) else len(pred_mean_rmm1)
                obs_rmm1 = np.zeros((n_pred_min, lead_time))
                obs_rmm2 = np.zeros((n_pred_min, lead_time))
                for i in range(n_pred_min):
                    start_date_id = start_date_ids[i]
                    obs_rmm1[i,:] = npzfile['RMM1'][start_date_id : start_date_id+lead_time]
                    obs_rmm2[i,:] = npzfile['RMM2'][start_date_id : start_date_id+lead_time]

            # errs
            self.cor_entire[dir_name] = emp_model.cor(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.rmse_entire[dir_name] = emp_model.rmse(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.phase_err_entire[dir_name] = emp_model.phase_err(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.amplitude_err_entire[dir_name] = emp_model.amplitude_err(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)

            self.crps_entire[dir_name] = emp_model.crps(pred_mean_rmm1=pred_mean_rmm1, pred_mean_rmm2=pred_mean_rmm2, 
                                                pred_std_rmm1=pred_std_rmm1, pred_std_rmm2=pred_std_rmm2, 
                                                obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.mll_entire[dir_name] = emp_model.mll(pred_mean_rmm1=pred_mean_rmm1, pred_mean_rmm2=pred_mean_rmm2,
                                                pred_std_rmm1=pred_std_rmm1, pred_std_rmm2=pred_std_rmm2, pred_crosscov=pred_crosscov,
                                                obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
    
    
    
    def add_nn(self, model_clses=[FFNNModel], width=300, lead_time=60, 
            hidden_dim=64, num_epochs=10, lr=0.01, seed=99, verbose=True):
        npzfile = self.npzfile
        n_pred = self.n_pred
        emp_model = self.emp_model
        start_test = self.start_tests[self.widths[0]] + self.widths[0] - width

        for model_cls in model_clses:
            if model_cls.__name__ == 'FFNNModel':
                model_name = f"FFNN({width})"

            nn_mjo_model = NNMJO(npzfile=npzfile, width=width, lead_time=lead_time, 
                    n=self.n, start_train=self.start_train, n_offset=self.n_offset,
                    start_val=None, v=self.v)
            nn_mjo_model.train(model_cls=FFNNModel, hidden_dim=hidden_dim, 
                            num_epochs=num_epochs, lr=lr, seed=seed, verbose=verbose)
            nn_mjo_model.pred(start_test=start_test, test_ids=None, 
                        n_pred=n_pred, verbose=verbose)
            
            pred_mean_rmm1 = nn_mjo_model.preds_y[:,:,0].detach().numpy() # [n_pred, lead_time]-size tensor
            pred_mean_rmm2 = nn_mjo_model.preds_y[:,:,1].detach().numpy() # [n_pred, lead_time]-size tensor
            obs_rmm1 = nn_mjo_model.obs_y[:,:,0].detach().numpy() # [n_pred, lead_time]-size tensor
            obs_rmm2 = nn_mjo_model.obs_y[:,:,1].detach().numpy() # [n_pred, lead_time]-size tensor

            # errs
            self.cor_entire[model_name] = emp_model.cor(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.rmse_entire[model_name] = emp_model.rmse(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.phase_err_entire[model_name] = emp_model.phase_err(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
            self.amplitude_err_entire[model_name] = emp_model.amplitude_err(pred_rmm1=pred_mean_rmm1, pred_rmm2=pred_mean_rmm2, obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2)
    
    
    def plot_metrics(self, nrows=2, ncols=3, plot_skill=True):
        plt.rc('xtick', labelsize=38)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=38)  # fontsize of the tick labels
        plt.rcParams['lines.linewidth'] = 3.0 # Change linewidth of plots
        
        palette_colors = plt.get_cmap('tab20').colors  # Use the 'tab20' colormap
        markers_class = list(MarkerStyle.markers.keys())

        fig, axs = plt.subplots(nrows, ncols, figsize=(16*ncols, 12*nrows))

        num_legends = len(self.rmse_entire.keys())
        # colors = palette_colors[0 : 0 + num_legends*2 : 2]
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 
                'tab:olive', 'tab:cyan', 'tab:gray', 'tab:pink', 'tab:brown']
        markers = markers_class[0 : 0 + num_legends]

        metrics = [self.cor_entire, self.rmse_entire, 
                self.phase_err_entire, self.amplitude_err_entire, 
                self.crps_entire, self.mll_entire]
        title_names = ['COR', 'RMSE', 'Phase Error', 'Amplitude Error', 'CRPS', 'Ignorance Score']
        skills = [0.5, 1.4, 0.0, -0.6, 0.6, -2.0]

        i, j = 0, 0
        for (title_name, metric, skill) in zip(title_names, metrics, skills):
            ax = axs[i,j]
            for (color, marker, key) in zip(colors, markers, metric.keys()):
                val = metric[key]
                lead_time = np.arange(1, 1+len(val))
                tick_positions = np.arange(60+1)

                # label = r'$\bf{GP}$' + f"({key})" if isinstance(key, (int, float)) else key.upper()
                if isinstance(key, (int, float)):
                    label = r'$\bf{GP}$' + f"({key})"
                elif key == 'ecmwf_txt':
                    label = f"ECMWF (with ERA5)" if self.era5_obs else f"ECMWF"
                else:
                    label = key.upper()
                ls = '-' if isinstance(key, (int, float)) else '--'
                lw = 4.0 if isinstance(key, (int, float)) else 3.5
                alpha = 1.0 if isinstance(key, (int, float)) else 0.9

                ax.plot(lead_time, val, color=color, 
                        ls=ls, lw=lw, alpha=alpha,
                        marker=marker, markersize=10, 
                        label=label)
                if plot_skill:
                    ax.axhline(y = skill, color = 'black', linestyle = '--', lw=2.8)
                
                ax.set_xlabel("Forecast lead time (days)", fontsize=45, labelpad=18)
                ax.set_title(f'{title_name}', pad=32, fontsize=52, fontweight='bold', color='blue')

                # Set x-ticks to be evenly spaced with values from grid_x
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_positions)
                ax.tick_params(axis='x', labelsize=38, rotation=0, length=10, width=2, colors='black', direction='inout')
                
                # Setting the grid
                # Making the grid more sparse by setting the major ticks
                ax.xaxis.set_major_locator(MaxNLocator(13))  # Number of grid lines on x-axis
                ax.yaxis.set_major_locator(MaxNLocator(9))  # Number of grid lines on y-axis

                # Optional: Setting minor grid lines for better readability (if needed)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.grid(which='major', linestyle=':', linewidth='2.5')

            if j < ncols-1:
                    j += 1
            else:
                i += 1
                j = 0
            
        # Adjust layout to make room for the legend
        plt.subplots_adjust(bottom=0.2)
        # plt.subplots_adjust(top=0.1)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        # fig.suptitle(f"Metrics", fontsize=65, fontweight='bold')

        # lines = [] 
        # labels = [] 
        # for ax in fig.axes: 
        lines, labels = axs[0,0].get_legend_handles_labels() 
        # lines.extend(line) 
        # labels.extend(label)
        ncol_legend = 7 if num_legends > 7 else num_legends
        fig.legend(lines, labels, fontsize=65, loc='lower center', bbox_to_anchor=(0.53, -0.05),
                    ncol=ncol_legend, fancybox=True, shadow=True)

        parent_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../SINDy_RealData
        filename = "metrics_compare_era5.png" if self.era5_obs else "metrics_compare_bom.png"
        file_path = os.path.join(parent_dir_path, "figs", "metrics", filename)
        fig.savefig(file_path, bbox_inches='tight')
        plt.show()
    
    
    def plot_ts(self, nrows=2, ncols=2, hdate_ids=[9, 12, 15, 19], 
                pred_id=0, n_pred=None, Ns=1000, seed=99, 
                angle_mean_from_samples=True, amplitude_mean_from_samples=True, 
                plot_ensembles=False):
        plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=35)  # fontsize of the tick labels
        plt.rcParams['lines.linewidth'] = 3.0 # Change linewidth of plots
        
        palette_colors = plt.get_cmap('tab20').colors  # Use the 'tab20' colormap
        palette_colors = list(mcolors.TABLEAU_COLORS.keys()) # list of Tableau Palette colors
        palette_colors = [color for color in palette_colors if color != 'tab:gray']# Remove 'tab:gray' from the list
        markers_class = list(MarkerStyle.markers.keys())

        fig, axs = plt.subplots(nrows, ncols, figsize=(16*ncols, 12*nrows))

        s2s_data = self.s2s_data
        dir_name = self.choose_dir_name
        if n_pred is None:
            n_pred = self.n_pred
        if dir_name == 'ecmwf_txt':
            s2s_lead_time = s2s_data[dir_name]['ensemble_mean']['RMM1(forecast)'].shape[-1]
            s2s_test_ids = s2s_data[dir_name]['ensemble_mean']['S'][:n_pred,0]
        else:
            s2s_lead_time = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'].shape[-1]
            s2s_test_ids = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['S'][:n_pred]
        
        max_lead_time = max( self.lead_time,  s2s_lead_time)

        # check if choose_dir_name and gp model has the same test dates
        gp_test_ids = self.test_ids[self.widths[0]] + self.widths[0]
        

        if (gp_test_ids == s2s_test_ids).all():
            print("test ids of GP model and selected S2S data are Equal! Continue...")
        else:
            RuntimeError("test ids of GP model and selected S2S data are not Equal! Please check.")
            
        if dir_name in ['ecmwf', 'eccc']:
            for hdate_id in hdate_ids:
                pred_mean = {}
                pred_lower = {}
                pred_upper = {}
                ensembles = {}
                pred_mean['RMM1'] = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][hdate_id][:n_pred, :] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean['RMM2'] = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][hdate_id][:n_pred, :] # [n_pred, lead_time]

                ensembles['RMM1'] = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][hdate_id][:n_pred, :, :] # [n_pred, lead_time, num_ensembles]
                ensembles['RMM2'] = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][hdate_id][:n_pred, :, :] # [n_pred, lead_time, num_ensembles]
                key_name = dir_name + f"_hdate={hdate_id}"

                self.observed_preds[key_name] = pred_mean

                pred_covs = np.zeros((n_pred, pred_mean['RMM1'].shape[1], 2, 2)) # [n_pred, lead_time, 2, 2]
                ensembles_rmm1_norm = ensembles['RMM1'] - ensembles['RMM1'].mean(axis=-1)[..., None]
                ensembles_rmm2_norm = ensembles['RMM2'] - ensembles['RMM2'].mean(axis=-1)[..., None]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1) # [n_pred, lead_time]
                pred_covs[:,:, 0, 0] = ensembles['RMM1'].std(axis=-1) ** 2
                pred_covs[:,:, 1, 1] = ensembles['RMM2'].std(axis=-1) ** 2
                pred_covs[:,:, 0, 1] = pred_covs[:,:, 1, 0] = pred_crosscov
                self.pred_covs[key_name] = pred_covs

                for rmm in ['RMM1', 'RMM2']:
                    pred_std = ensembles[rmm].std(axis=-1)
                    pred_lower[rmm] = pred_mean[rmm] - pred_std
                    pred_upper[rmm] = pred_mean[rmm] + pred_std
                self.lower_confs[key_name] = pred_lower
                self.upper_confs[key_name] = pred_upper
                if plot_ensembles:
                    if hdate_id == hdate_ids[-1]:
                        self.observed_preds['ensembles'+f"_hdate={hdate_id}"] = ensembles
        else:
            pred_mean = {}
            pred_lower = {}
            pred_upper = {}
            ensembles = {}
            if dir_name == 'ecmwf_txt':
                # pred_mean['RMM1'] = s2s_data[dir_name]['ensemble_mean']['RMM1(forecast)'][-n_pred:, :] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                # pred_mean['RMM2'] = s2s_data[dir_name]['ensemble_mean']['RMM2(forecast)'][-n_pred:, :] # [n_pred, lead_time]

                ensembles['RMM1'] = s2s_data[dir_name]['ensembles']['RMM1(forecast)'][:n_pred, :, :] # [n_pred, lead_time, num_ensembles]
                ensembles['RMM2'] = s2s_data[dir_name]['ensembles']['RMM2(forecast)'][:n_pred, :, :] # [n_pred, lead_time, num_ensembles]

                pred_mean['RMM1'] = ensembles['RMM1'].mean(axis=-1) # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean['RMM2'] = ensembles['RMM2'].mean(axis=-1) # [n_pred, lead_time]
                self.observed_preds[dir_name] = pred_mean

                era5_obs = {}
                era5_obs['RMM1'] = s2s_data[dir_name]['ensembles']['RMM1(obs)'][:n_pred, :] 
                era5_obs['RMM2'] = s2s_data[dir_name]['ensembles']['RMM2(obs)'][:n_pred, :]
                era5_obs['Phase'] = np.arctan2( era5_obs['RMM2'], era5_obs['RMM1'] ) * 180 / np.pi + 180
                era5_obs['Amplitude'] = np.sqrt( np.square(era5_obs['RMM1']) + np.square(era5_obs['RMM2']) )
                self.observed_preds['ERA5(obs)'] = era5_obs
            else:        
                ensembles['RMM1'] = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][:n_pred, :, :]
                ensembles['RMM2'] = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][:n_pred, :, :]

                pred_mean['RMM1'] = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][:n_pred, :] 
                pred_mean['RMM2'] = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][:n_pred, :]
                self.observed_preds[dir_name] = pred_mean
            
            pred_covs = np.zeros((n_pred, pred_mean['RMM1'].shape[1], 2, 2)) # [n_pred, lead_time, 2, 2]
            ensembles_rmm1_norm = ensembles['RMM1'] - ensembles['RMM1'].mean(axis=-1)[..., None]
            ensembles_rmm2_norm = ensembles['RMM2'] - ensembles['RMM2'].mean(axis=-1)[..., None]
            pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1) # [n_pred, lead_time]
            pred_covs[:,:, 0, 0] = ensembles['RMM1'].std(axis=-1) ** 2
            pred_covs[:,:, 1, 1] = ensembles['RMM2'].std(axis=-1) ** 2
            pred_covs[:,:, 0, 1] = pred_covs[:,:, 1, 0] = pred_crosscov
            self.pred_covs[dir_name] = pred_covs
            
            for rmm in ['RMM1', 'RMM2']:
                pred_std = ensembles[rmm].std(axis=-1)
                pred_lower[rmm] = pred_mean[rmm] - pred_std
                pred_upper[rmm] = pred_mean[rmm] + pred_std
            self.lower_confs[dir_name] = pred_lower
            self.upper_confs[dir_name] = pred_upper
            if plot_ensembles:
                self.observed_preds['ensembles'] = ensembles
        
        # Phase & Amplitude
        for key in self.lower_confs.keys():
            pred_mean_rmm1 = self.observed_preds[key]['RMM1'] # [n_pred, lead_time]
            pred_mean_rmm2 = self.observed_preds[key]['RMM2'] # [n_pred, lead_time]
            pred_mean = np.stack((pred_mean_rmm1, pred_mean_rmm2), axis=-1) # [n_pred, lead_time, 2]
            pred_cov = self.pred_covs[key] # [n_pred, lead_time, 2, 2]

            # Generate samples
            samples = np.zeros((Ns, pred_mean.shape[0], pred_mean.shape[1], 2))
            rng = np.random.default_rng(seed)
            for i in range(pred_mean.shape[0]):
                for j in range(pred_mean.shape[1]):
                    mean = pred_mean[i, j]
                    cov = pred_cov[i, j]
                    # Ensure covariance matrix is symmetric positive semi-definite
                    cov = (cov + cov.T) / 2
                    cov += np.eye(cov.shape[0]) * 1e-8  # Adding a small value to the diagonal for numerical stability
                    samples[:, i, j, :] = rng.multivariate_normal(mean, cov, Ns)
                    
            
            # Calc phase (arctan2)
            angle_samples = np.arctan2(samples[...,1], samples[...,0]) * 180 / np.pi + 180 # (Ns, n_pred, lead_time) array
            if angle_mean_from_samples:
                angle_mean = np.mean(angle_samples, axis=0) # (n_pred, lead_time) array
            else:
                angle_mean = np.arctan2(pred_mean[...,1],pred_mean[...,0])  * 180 / np.pi + 180  # np.mean(angle_samples, axis=0)# (1, ) array 
            
            angle_norm = angle_samples - angle_mean[None,...] # (Ns, n_pred, lead_time) array
            angle_var = np.multiply(angle_norm, angle_norm).mean(axis=0) # (n_pred, lead_time) array
            angle_std = np.sqrt(angle_var)

            self.observed_preds[key]['Phase'] = angle_mean
            self.lower_confs[key]['Phase'] = angle_mean - angle_std
            self.upper_confs[key]['Phase'] = angle_mean + angle_std

            # Calc amplitude
            amplitude_samples = np.sqrt( np.square(samples[...,0]) + np.square(samples[...,1]) ) # (Ns, n_pred, lead_time) array
            if amplitude_mean_from_samples:
                amplitude_mean = np.mean(amplitude_samples, axis=0) # (n_pred, lead_time) array
            else:
                amplitude_mean = np.sqrt( np.square(pred_mean[...,0]) + np.square(pred_mean[...,1]) ) # (n_pred, lead_time) array
            amplitude_norm = amplitude_samples - amplitude_mean[None,...] # (Ns, n_pred, lead_time) array
            amplitude_var = np.multiply(amplitude_norm, amplitude_norm).mean(axis=0) # (n_pred, lead_time) array
            amplitude_std = np.sqrt(amplitude_var)

            self.observed_preds[key]['Amplitude'] = amplitude_mean
            self.lower_confs[key]['Amplitude'] = amplitude_mean - amplitude_std
            self.upper_confs[key]['Amplitude'] = amplitude_mean + amplitude_std

            if isinstance(key, (int, float)):
                self.obs[key]['Phase'] = np.arctan2( self.obs[key]['RMM2'], self.obs[key]['RMM1'] ) * 180 / np.pi + 180
                self.obs[key]['Amplitude'] = self.obs[key]['amplitude']

        num_legends = len(self.observed_preds.keys())
        # colors = palette_colors[0 : 0 + num_legends*2 : 2]
        colors = palette_colors[0 : 0 + num_legends]
        markers = markers_class[0 : 0 + num_legends]

        title_names = ['RMM1', 'RMM2', 'Phase', 'Amplitude']

        # compute dates at pred_id
        years = self.obs[self.widths[0]]['year'][pred_id, :].astype('int')
        months = self.obs[self.widths[0]]['month'][pred_id, :].astype('int')
        days = self.obs[self.widths[0]]['day'][pred_id, :].astype('int')
        df = pd.DataFrame({'year': years,
                        'month': months,
                        'day': days})
        df_dates = pd.to_datetime(df[["year", "month", "day"]], format="%Y-%m-%d")
        dates = pd.date_range(df_dates[0], periods=max_lead_time, freq='1D')

        # Define tick positions for the x-axis
        num_ticks = 5
        tick_positions = pd.date_range(start=dates[0], end=dates[-1], periods=num_ticks)

        # Select 5 evenly spaced tick positions
        tick_indices = np.linspace(0, len(dates)-1, num_ticks, dtype=int)
        fixed_tick_positions = [dates[i] for i in tick_indices]

        # Generate labels, showing only 0, 2, ..., num_ticks-1-th label
        tick_labels = [date.strftime('%b-%d-%Y') if idx % 2 == 0 else '' for idx, date in enumerate(fixed_tick_positions)]

        i, j = 0, 0
        for title_name in title_names:
            ax = axs[i, j]
            ax.plot(dates[:self.lead_time], self.obs[self.widths[0]][title_name][pred_id,:], 
                color='black', marker='o', label='Truth(BOM)')
            for (color, marker, key) in zip(colors, markers, self.observed_preds.keys()):
                val = self.observed_preds[key][title_name][pred_id, ...]
                lead_time = len(val)

                # label = r'$\bf{GP}$' + f"({key})" if isinstance(key, (int, float)) else key
                # ls = '-' if isinstance(key, (int, float)) else '--'
                # lw = 4.0 if isinstance(key, (int, float)) else 3.5
                # alpha = 1.0 if isinstance(key, (int, float)) else 0.95

                if isinstance(key, (int, float)):
                    label = r'$\bf{GP}$' + f"({key})"
                    ls = '-'
                    lw = 4.0
                    alpha = 1.0
                    markersize = 10
                elif 'ensembles' in key:
                    label = 'Ensembles'
                    ls = '-'
                    lw = 2.5
                    alpha = 0.55
                    markersize = 2
                    color = 'tab:gray'
                else:
                    if key in ['ecmwf_txt','ECMWF_TXT']:
                        label = 'ECMWF'
                    elif key in ['ERA5(obs)', 'ERA5(OBS)']:
                        label = 'ERA5'
                    else:
                        label = key.upper()
                    ls = '--'
                    lw = 3.5
                    alpha = 0.95
                    markersize = 10


                ax.plot(dates[:lead_time], val, color=color, 
                        ls=ls, lw=lw, alpha=alpha,
                        marker=marker, markersize=markersize, 
                        label=label)
                
                if key != 'ERA5(obs)': #isinstance(key, (int, float))
                    ax.fill_between(dates[:lead_time], self.lower_confs[key][title_name][pred_id,:], 
                                self.upper_confs[key][title_name][pred_id,:], 
                                alpha=0.2, color=color)#, label=f'CI ({key})')
                
            # ax.set_xlabel("Forecast lead time (days)", fontsize=38, labelpad=15)
            ax.set_title(f'{title_name}', pad=25, fontsize=40, fontweight='bold', color='blue')

            # Set x-ticks to be evenly spaced with values from grid_x
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions.strftime('%b-%d-%Y'))
            ax.tick_params(axis='x', labelsize=33, rotation=0, length=10, width=2, colors='black', direction='inout')


            # Setting the grid
            # Making the grid more sparse by setting the major ticks
            # Set x-ticks to be evenly spaced with values from fixed_tick_positions
            # ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%Y'))
            ax.set_xticks(fixed_tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.yaxis.set_major_locator(MaxNLocator(9))  # Number of grid lines on y-axis


            # Optional: Setting minor grid lines for better readability (if needed)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which='major', linestyle=':', linewidth='2.5')

            if j < ncols-1:
                    j += 1
            else:
                i += 1
                j = 0
            
        # Adjust layout to make room for the legend
        plt.subplots_adjust(bottom=0.01)
        # plt.subplots_adjust(top=0.1)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        # fig.suptitle(f"60--Days Time Series from {dates.strftime('%b-%d-%Y')[0]} to {dates.strftime('%b-%d-%Y')[-1]}", 
        #             fontsize=55, fontweight='bold', y=1.05)
        fig.suptitle(f"60--Days Time Series", 
                    fontsize=55, fontweight='bold', y=1.00)
        

        # lines = [] 
        # labels = [] 
        # for ax in fig.axes: 
        lines, labels = axs[0,0].get_legend_handles_labels() 
        # lines.extend(line) 
        # labels.extend(label)

        unique_labels = {}
        lines, labels = axs[0,0].get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in unique_labels:
                unique_labels[label] = line
        # Prepare the final lines and labels for the legend
        final_lines = list(unique_labels.values())
        final_labels = list(unique_labels.keys())
        
        ncol_legend = 4 if (num_legends+1) > 5 else (num_legends+1)
        fig.legend(final_lines, final_labels, fontsize=50, loc='lower center', bbox_to_anchor=(0.53, -0.05),
                    ncol=ncol_legend , fancybox=True, shadow=True)

        parent_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../SINDy_RealData
        file_path = os.path.join(parent_dir_path, "figs", "ts", f"ts_{dir_name}.png")
        fig.savefig(file_path, bbox_inches='tight')
        plt.show()