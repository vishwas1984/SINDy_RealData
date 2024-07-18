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
            if test_ids is None:
                if self.choose_dir_name == 'ecmwf_txt':
                    start_dates = s2s_data[self.choose_dir_name]['ensemble_mean']['S'][-self.n_pred :, 0]
                else:
                    start_dates = s2s_data[self.choose_dir_name]['ensemble_mean_rmm1.nc']['S'][-self.n_pred : ]
                self.test_ids[width] = start_dates - width
            else:
                self.test_ids[width] = test_ids - width

            emp_model = EmpGPMJO(npzfile=npzfile, width=width, lead_time=self.lead_time, n=self.n, start_train=self.start_train)
            emp_model.get_emp()
            emp_model.get_biasvar(start_val=self.start_vals[width], n_pred=365*4, v=self.v, season_bool=False)
            if test_on_choose_dir:
                emp_model.pred(test_ids=self.test_ids[width], n_pred=self.n_pred, m=self.m, season_bool=False) # test on the same dates as choose_dir_name
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

            self.emp_model = emp_model
    

    def add_s2s(self, dir_names, hdate_id=19, era5_obs=True):
        
        self.era5_obs = era5_obs
        npzfile = self.npzfile
        s2s_data = self.s2s_data
        n_pred = self.n_pred
        emp_model = self.emp_model

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

                ensembles_rmm1 = s2s_data[dir_name]['ensembles']['RMM1(forecast)'][-n_pred:, :lead_time, :] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2 = s2s_data[dir_name]['ensembles']['RMM2(forecast)'][-n_pred:, :lead_time, :] # [n_pred, lead_time, num_ensembles]

                pred_mean_rmm1 = ensembles_rmm1.mean(axis=-1) # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean_rmm2 = ensembles_rmm2.mean(axis=-1) # [n_pred, lead_time]

                pred_std_rmm1 = ensembles_rmm1.std(axis=-1) # [n_pred, lead_time]
                pred_std_rmm2 = ensembles_rmm2.std(axis=-1) # [n_pred, lead_time]


                ensembles_rmm1_norm = ensembles_rmm1 - ensembles_rmm1.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2_norm = ensembles_rmm2 - ensembles_rmm2.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1) # [n_pred, lead_time]
            
                start_date_ids = s2s_data[dir_name]['ensemble_mean']['S'][:,0]

            elif dir_name in ['ecmwf', 'eccc']:
                
                pred_mean_rmm1 = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][hdate_id][-n_pred:, :] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean_rmm2 = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][hdate_id][-n_pred:, :] # [n_pred, lead_time]

                ensembles_rmm1 = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][hdate_id][-n_pred:, :, :] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2 = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][hdate_id][-n_pred:, :, :] # [n_pred, lead_time, num_ensembles]

                pred_std_rmm1 = ensembles_rmm1.std(axis=-1) # [n_pred, lead_time]
                pred_std_rmm2 = ensembles_rmm2.std(axis=-1) # [n_pred, lead_time]

                ensembles_rmm1_norm = ensembles_rmm1 - ensembles_rmm1.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                ensembles_rmm2_norm = ensembles_rmm2 - ensembles_rmm2.mean(axis=-1)[..., None] # [n_pred, lead_time, num_ensembles]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1) # [n_pred, lead_time]

                start_date_ids = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['S'][-n_pred:]

            else:
                pred_mean_rmm1 = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][-n_pred:, :] 
                pred_mean_rmm2 = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][-n_pred:, :]
                
                ensembles_rmm1 = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][-n_pred:, :, :]
                ensembles_rmm2 = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][-n_pred:, :, :]
                pred_std_rmm1 = ensembles_rmm1.std(axis=-1)
                pred_std_rmm2 = ensembles_rmm2.std(axis=-1)

                ensembles_rmm1_norm = ensembles_rmm1 - ensembles_rmm1.mean(axis=-1)[..., None]
                ensembles_rmm2_norm = ensembles_rmm2 - ensembles_rmm2.mean(axis=-1)[..., None]
                pred_crosscov = np.multiply(ensembles_rmm1_norm, ensembles_rmm2_norm).mean(axis=-1)

                start_date_ids = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['S'][-n_pred:]
            
            if dir_name == 'ecmwf_txt' and era5_obs:
                obs_rmm1 = s2s_data[dir_name]['ensembles']['RMM1(obs)'][-n_pred:, :lead_time]
                obs_rmm2 = s2s_data[dir_name]['ensembles']['RMM2(obs)'][-n_pred:, :lead_time]
            else:
                obs_rmm1 = np.zeros((n_pred, lead_time))
                obs_rmm2 = np.zeros((n_pred, lead_time))
                for i in range(n_pred):
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
    
    
    def plot_metrics(self, nrows=2, ncols=3):
        plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=35)  # fontsize of the tick labels
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

        i, j = 0, 0
        for (title_name, metric) in zip(title_names, metrics):
            ax = axs[i,j]
            for (color, marker, key) in zip(colors, markers, metric.keys()):
                val = metric[key]
                lead_time = np.arange(1, 1+len(val))
                tick_positions = np.arange(60+1)

                # label = r'$\bf{GP}$' + f"({key})" if isinstance(key, (int, float)) else key.upper()
                if isinstance(key, (int, float)):
                    label = r'$\bf{GP}$' + f"({key})"
                elif key == 'ecmwf_txt':
                    label = f"{key.upper()}(with ERA5)" if self.era5_obs else f"{key.upper()}(with BOM)"
                else:
                    label = key.upper()
                ls = '-' if isinstance(key, (int, float)) else '--'
                lw = 4.0 if isinstance(key, (int, float)) else 3.5
                alpha = 1.0 if isinstance(key, (int, float)) else 0.9

                ax.plot(lead_time, val, color=color, 
                        ls=ls, lw=lw, alpha=alpha,
                        marker=marker, markersize=10, 
                        label=label)
                
                ax.set_xlabel("Forecast lead time (days)", fontsize=38, labelpad=15)
                ax.set_title(f'{title_name}', pad=25, fontsize=40, fontweight='bold', color='blue')

                # Set x-ticks to be evenly spaced with values from grid_x
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_positions)
                ax.tick_params(axis='x', labelsize=33, rotation=0, length=10, width=2, colors='black', direction='inout')


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
        fig.legend(lines, labels, fontsize=50, loc='lower center', bbox_to_anchor=(0.53, -0.06),
                    ncol=ncol_legend, fancybox=True, shadow=True)

        parent_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../SINDy_RealData
        filename = "metrics_compare_era5.png" if self.era5_obs else "metrics_compare_bom.png"
        file_path = os.path.join(parent_dir_path, "figs", "metrics", filename)
        fig.savefig(file_path, bbox_inches='tight')
        plt.show()
    
    
    def plot_ts(self, nrows=1, ncols=2, hdate_ids=[9, 12, 15, 19], pred_id=0):
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
        n_pred = self.n_pred
        if dir_name == 'ecmwf_txt':
            s2s_lead_time = s2s_data[dir_name]['ensemble_mean']['RMM1(forecast)'].shape[-1]
            s2s_test_ids = s2s_data[dir_name]['ensemble_mean']['S'][-n_pred:,0]
        else:
            s2s_lead_time = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'].shape[-1]
            s2s_test_ids = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['S'][-n_pred:]
        
        max_lead_time = max( self.lead_time,  s2s_lead_time)

        # check if choose_dir_name and gp model has the same test dates
        gp_test_ids = self.test_ids[self.widths[0]] + self.widths[0]
        

        if (gp_test_ids == s2s_test_ids).all():
            print("test ids of GP model and selected S2S data are Equal! Continue...")
        else:
            RuntimeError("test ids of GP model and selected S2S data are not Equal! Please check.")

        if dir_name == 'ecmwf_txt':
            pred_mean = {}
            ensembles = {}
            # pred_mean['RMM1'] = s2s_data[dir_name]['ensemble_mean']['RMM1(forecast)'][-n_pred:, :] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
            # pred_mean['RMM2'] = s2s_data[dir_name]['ensemble_mean']['RMM2(forecast)'][-n_pred:, :] # [n_pred, lead_time]

            ensembles['RMM1'] = s2s_data[dir_name]['ensembles']['RMM1(forecast)'][-n_pred:, :, :] # [n_pred, lead_time, num_ensembles]
            ensembles['RMM2'] = s2s_data[dir_name]['ensembles']['RMM2(forecast)'][-n_pred:, :, :] # [n_pred, lead_time, num_ensembles]

            pred_mean['RMM1'] = ensembles['RMM1'].mean(axis=-1) # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
            pred_mean['RMM2'] = ensembles['RMM2'].mean(axis=-1) # [n_pred, lead_time]

            self.observed_preds[dir_name.upper()] = pred_mean
            self.observed_preds['Ensembles'] = ensembles

            era5_obs = {}
            era5_obs['RMM1'] = s2s_data[dir_name]['ensembles']['RMM1(obs)'][-n_pred:, :] 
            era5_obs['RMM2'] = s2s_data[dir_name]['ensembles']['RMM2(obs)'][-n_pred:, :]
            self.observed_preds['ERA5(obs)'] = era5_obs
            
        elif dir_name in ['ecmwf', 'eccc']:
            for hdate_id in hdate_ids:
                pred_mean = {}
                ensembles = {}
                pred_mean['RMM1'] = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][hdate_id][-n_pred:, :] # [n_pred, lead_time], select 'hdate' index as 9 (920, 46)
                pred_mean['RMM2'] = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][hdate_id][-n_pred:, :] # [n_pred, lead_time]

                ensembles['RMM1'] = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][hdate_id][-n_pred:, :, :] # [n_pred, lead_time, num_ensembles]
                ensembles['RMM2'] = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][hdate_id][-n_pred:, :, :] # [n_pred, lead_time, num_ensembles]
                key_name = dir_name.upper() + f"_hdate={hdate_id}"
                self.observed_preds[key_name] = pred_mean
                if hdate_id == hdate_ids[-1]:
                    self.observed_preds['Ensembles'+f"_hdate={hdate_id}"] = ensembles
        else:
            pred_mean = {}
            ensembles = {}
            pred_mean['RMM1'] = s2s_data[dir_name]['ensemble_mean_rmm1.nc']['RMM1'][-n_pred:, :] 
            pred_mean['RMM2'] = s2s_data[dir_name]['ensemble_mean_rmm2.nc']['RMM2'][-n_pred:, :]
                
            ensembles['RMM1'] = s2s_data[dir_name]['ensembles_rmm1.nc']['RMM1'][-n_pred:, :, :]
            ensembles['RMM2'] = s2s_data[dir_name]['ensembles_rmm2.nc']['RMM2'][-n_pred:, :, :]
            self.observed_preds[dir_name.upper()] = pred_mean
            self.observed_preds['Ensembles'] = ensembles

        num_legends = len(self.observed_preds.keys())
        # colors = palette_colors[0 : 0 + num_legends*2 : 2]
        colors = palette_colors[0 : 0 + num_legends]
        markers = markers_class[0 : 0 + num_legends]

        title_names = ['RMM1', 'RMM2']#['RMM1', 'RMM2', 'Phase', 'Amplitude']

        # compute dates at pred_id
        years = self.obs[self.widths[0]]['year'][pred_id, :].astype('int')
        months = self.obs[self.widths[0]]['month'][pred_id, :].astype('int')
        days = self.obs[self.widths[0]]['day'][pred_id, :].astype('int')
        df = pd.DataFrame({'year': years,
                        'month': months,
                        'day': days})
        df_dates = pd.to_datetime(df[["year", "month", "day"]], format="%Y-%m-%d")
        dates = pd.date_range(df_dates[0], periods=max_lead_time, freq='1D')

        i, j = 0, 0
        for title_name in title_names:
            ax = axs[j]
            ax.plot(dates[:self.lead_time], self.obs[self.widths[0]][title_name][pred_id,:], color='black', marker='o', label='Truth')
            for (color, marker, key) in zip(colors, markers, self.observed_preds.keys()):
                val = self.observed_preds[key][title_name][pred_id, ...]
                lead_time = len(val)
                tick_positions = dates#np.arange(60+1)

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
                elif 'Ensembles' in key:
                    label = 'Ensembles'
                    ls = '-'
                    lw = 2.5
                    alpha = 0.55
                    markersize = 2
                    color = 'tab:gray'
                else:
                    label = key
                    ls = '--'
                    lw = 3.5
                    alpha = 0.95
                    markersize = 10


                ax.plot(dates[:lead_time], val, color=color, 
                        ls=ls, lw=lw, alpha=alpha,
                        marker=marker, markersize=markersize, 
                        label=label)
                
                if isinstance(key, (int, float)):
                    ax.fill_between(dates[:lead_time], self.lower_confs[key][title_name][pred_id,:], 
                                self.upper_confs[key][title_name][pred_id,:], 
                                alpha=0.2, color=color)#, label=f'CI ({key})')
                
            # ax.set_xlabel("Forecast lead time (days)", fontsize=38, labelpad=15)
            ax.set_title(f'Time Series of {title_name}', pad=25, fontsize=40, fontweight='bold', color='blue')

            # Set x-ticks to be evenly spaced with values from grid_x
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions)
            ax.tick_params(axis='x', labelsize=33, rotation=0, length=10, width=2, colors='black', direction='inout')


            # Setting the grid
            # Making the grid more sparse by setting the major ticks
            # ax.xaxis.set_major_locator(MaxNLocator(13))  # Number of grid lines on x-axis
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%Y'))
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
        # fig.suptitle(f"Metrics", fontsize=65, fontweight='bold')

        # lines = [] 
        # labels = [] 
        # for ax in fig.axes: 
        lines, labels = axs[0].get_legend_handles_labels() 
        # lines.extend(line) 
        # labels.extend(label)

        unique_labels = {}
        lines, labels = axs[0].get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in unique_labels:
                unique_labels[label] = line
        # Prepare the final lines and labels for the legend
        final_lines = list(unique_labels.values())
        final_labels = list(unique_labels.keys())
        
        ncol_legend = 4 if (num_legends+1) > 5 else (num_legends+1)
        fig.legend(final_lines, final_labels, fontsize=50, loc='lower center', bbox_to_anchor=(0.53, -0.25),
                    ncol=ncol_legend , fancybox=True, shadow=True)

        parent_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../SINDy_RealData
        file_path = os.path.join(parent_dir_path, "figs", "ts", f"ts_{dir_name}.png")
        fig.savefig(file_path, bbox_inches='tight')
        plt.show()