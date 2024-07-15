import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm, chi2, fisher_exact

from ..utils.dat_ops import rolling

class EmpGPMJO:
    def __init__(self, npzfile, width=40, lead_time=60, n = 10000, start_train=0, season_bool=False) -> None:
        self.npzfile = npzfile
        self.width = width
        self.lead_time = lead_time
        self.n = n
        self.start_train = start_train
        d = width + lead_time
        self.n_train = d + n    

        errlist = ['cor','rmse','phase','amplitude','hss','hss_n','hss_signif', 'crps', 'mlla']
        self.errs = {key: None for key in errlist}

        train_data = np.array([])
        #train_y = np.array([])
        train_ids = np.arange(start_train, start_train + self.n_train)
        train_rmms = np.array([])
        for rmm in ['RMM1','RMM2']:
            if season_bool:
                train_datarmm = np.array([]).reshape(-1, d)
                #train_yrmm = np.array([]).reshape(-1, 1)
                train_id_split = np.hstack( ( np.array([0]), 
                            np.where(np.ediff1d(npzfile['id'][train_ids]) != 1 )[0]+1, 
                            np.array([len(npzfile['id'][train_ids])]) ) )
                diff_train_ids = np.ediff1d(train_id_split)
                
                for i, diff_train_id in enumerate(diff_train_ids):
                    if diff_train_id <= d:
                        print(f'width+lead_time = {d} is greater than the current interval width {diff_train_id}, will skip {i}-th iteration for {rmm}')
                        continue
                    split_start = train_id_split[i]
                    split_end = train_id_split[i+1]
                    train_ij = npzfile[rmm][train_ids][split_start:split_end]
                    train_rmms = np.hstack((train_rmms, train_ij))

                    train_datarmm = np.vstack(( train_datarmm, rolling(train_ij[:-1], d) )) # (n, d) numpy array
                    #train_yrmm = np.vstack(( train_yrmm, train_ij[d:].reshape(-1,1) )) # (n, 1) numpy array
            
            else:    
                train_ij = npzfile[rmm][train_ids]
                train_rmms = np.hstack((train_rmms, train_ij))

                train_datarmm = rolling(train_ij[:-1], d) # (n, d) numpy array
                #train_yrmm = train_ij[d:].reshape(-1,1) # (n, 1) numpy array
            
            train_data = train_data.reshape(train_datarmm.shape[0], -1)
            #train_y = train_y.reshape(train_yrmm.shape[0], -1)

            train_data = np.hstack(( train_data, train_datarmm )) # (n, 2*d) numpy array
            #train_y = np.hstack( (train_y, train_yrmm) ) # (n, 2) numpy array

        self.train_data = train_data # (n, 2*d) numpy array
        #self.train_y = train_y
        self.train_rmms = train_rmms
        self.d = d
        self.train_ids = train_ids


    def get_emp(self):
        width = self.width
        lead_time = self.lead_time
        d = self.d
        train_data = self.train_data # (n, 2*d) numpy array

        self.train_mean = np.mean(train_data, axis=0) # (2*d, ) numpy array
        self.train_cov = (train_data - self.train_mean).T @ (train_data - self.train_mean) / (train_data.shape[0]-1) # (2*d, 2*d) numpy array   
    
        #===================================================
        # use cubic spline to get cov([RMM1,RMM2],[RMM1,RMM2])
        #===================================================
        pts_spl = np.linspace(0.0, d, num=d, endpoint=True) # (d, ) array
        dist_spl = np.sqrt(( pts_spl.reshape(-1,1) - pts_spl )**2) # (d, d) array, distance matrix
        x_spl = dist_spl[0,:] # (d, ) array, 1st row of dist_spl
        
        # RMM1
        y_spl_rmm1 = self.train_cov[0,:d]
        spl_rmm1 = CubicSpline(x_spl, y_spl_rmm1)
        rmm1_cov = spl_rmm1(dist_spl) # (d, d) array

        # RMM2
        y_spl_rmm2 = self.train_cov[d,d:2*d]
        spl_rmm2 = CubicSpline(x_spl, y_spl_rmm2)
        rmm2_cov = spl_rmm2(dist_spl) # (d, d) array

        # RMM1-RMM2
        yupper_spl = self.train_cov[0,d:2*d]
        spl_rmm12upper = CubicSpline(x_spl, yupper_spl)
        rmm12upper_cov = spl_rmm12upper(dist_spl) # (d, d) array
        
        ylower_spl = np.flip( self.train_cov[d-1,d:2*d])
        spl_rmm12lower = CubicSpline(x_spl, ylower_spl)
        rmm12lower_cov = spl_rmm12lower(dist_spl) # (d, d) array

        #rmm12_cov = np.triu(rmm12upper_cov, k=1) + np.tril(-rmm12upper_cov, k=-1) + np.diag(np.diag( (rmm12upper_cov - rmm12upper_cov)/2 ))
        rmm12_cov = np.triu(rmm12upper_cov, k=1) + np.tril(rmm12lower_cov, k=-1) + np.diag(np.diag( (rmm12upper_cov + rmm12lower_cov)/2 ))

        pred_cov = self.train_cov # np.vstack(( np.hstack((rmm1_cov,rmm12_cov)), np.hstack((rmm12_cov.T,rmm2_cov)) )) # (2*d, 2*d) array
        pred_mean = self.train_mean # (2*d, ) numpy array

        mu1 = {}
        mu2 = {}
        K11 = {}
        K12 = {}
        K21 = {}
        K22 = {}
        K11_inv = {}
        K21_11 = {}
        Kcond = {}
        for j in range(lead_time):
            mu1[j] = np.hstack(( pred_mean[j:j+width],  pred_mean[j+d:j+d+width] )) # (2*width, ) numpy array
            mu2[j] = np.hstack(( pred_mean[j+width],  pred_mean[j+d+width] )) # (2, ) numpy array

            K11[j] = np.vstack((
                            np.hstack(( pred_cov[j:j+width,j:j+width], pred_cov[j:j+width,j+d:j+d+width] )),
                            np.hstack(( pred_cov[j+d:j+d+width,j:j+width], pred_cov[j+d:j+d+width,j+d:j+d+width] ))
                            )) # (2*width, 2*width) numpy array
            K12[j] = np.vstack((
                            np.hstack(( pred_cov[j:j+width,j+width][:,None], pred_cov[j:j+width,j+d+width][:,None] )),
                            np.hstack(( pred_cov[j+d:j+d+width,j+width][:,None], pred_cov[j+d:j+d+width,j+d+width][:,None] ))
                            )) # (2*width, 2) numpy array
            K21[j]  = np.vstack((
                            np.hstack(( pred_cov[j+width,j:j+width][None,:], pred_cov[j+width,j+d:j+d+width][None,:] )),
                            np.hstack(( pred_cov[j+d+width,j:j+width][None,:], pred_cov[j+d+width,j+d:j+d+width][None,:] ))
                            )) # (2, 2*width) numpy array
            K22[j]  = np.vstack((
                            np.hstack(( pred_cov[j+width,j+width], pred_cov[j+width,j+d+width] )),
                            np.hstack(( pred_cov[j+d+width,j+width], pred_cov[j+d+width,j+d+width] ))
                            )) # (2, 2) numpy array
        
            K11_inv[j] = np.linalg.inv(K11[j]) # (2*width, 2*width) numpy array
            K21_11[j] = K21[j] @ K11_inv[j] # (2, 2*width) numpy array
            Kcond[j] = K22[j] - K21_11[j] @ K12[j] # (2, 2) array


        self.pred_mean = pred_mean
        self.pred_cov = pred_cov

        self.mu1 = mu1
        self.mu2 = mu2
        self.K11 = K11
        self.K12 = K12
        self.K21 = K21
        self.K22 = K22
        self.K11_inv = K11_inv
        self.K21_11 = K21_11
        self.Kcond = Kcond


    def get_biasvar(self, start_val, n_pred = 1, v = 2500, season_bool=False):
        npzfile = self.npzfile
        width = self.width
        lead_time = self.lead_time
        d = self.d
        val_ids = np.arange(start_val, start_val + v)

        val_id_split = np.hstack( ( np.array([0]), 
                            np.where(np.ediff1d(npzfile['id'][val_ids]) != 1 )[0]+1, 
                            np.array([len(npzfile['id'][val_ids])]) ) )
        diff_val_ids = np.ediff1d(val_id_split)
        max_diff = np.max(diff_val_ids)
        freq_diff = np.bincount(diff_val_ids).argmax() # return the most frequent value in diff_val_ids
    
        if season_bool or len(val_id_split) > 2:
            if width >= freq_diff:
                raise ValueError(f'the width is greater than the season interval, please try a width value < {freq_diff}')
            if lead_time + width > freq_diff:
                print(f"the sum of the width and lead time is greater than the season interval..., will set lead time = {freq_diff-width}")
                lead_time = freq_diff-width
            
            pred_ids_start = val_id_split[:-1]
            pred_ids_end = (val_id_split - lead_time - width + 1)[1:]
            pred_ids = np.array([],dtype=int)
            for ii in range(len(pred_ids_start)):
                if (pred_ids_start[ii] >= pred_ids_end[ii]):
                    continue
                pred_ids_i = np.arange( start=pred_ids_start[ii], stop=pred_ids_end[ii] )
                pred_ids = np.hstack( (pred_ids,pred_ids_i), dtype=int)
            
            if n_pred > len(pred_ids):
                print(f"the number of predictions is greater than the number of the season intervals..., will set n_pred = {len(pred_ids)}")
                n_pred = len(pred_ids)            
            
        else:
            pred_ids = np.arange(n_pred)
        
        obs = {}
        observed_preds = {}
        lower_confs = {}
        upper_confs = {}
        predvar = {}
        biasvar = {}

        obs['phase'] = np.zeros((n_pred,lead_time))
        obs['amplitude'] = np.zeros((n_pred,lead_time))
        rmms = ['RMM1','RMM2']
        for rmm in rmms:
            obs[rmm] = np.zeros((n_pred,lead_time))
            observed_preds[rmm] = np.zeros((n_pred,lead_time))
            lower_confs[rmm] = np.zeros((n_pred,lead_time))
            upper_confs[rmm] = np.zeros((n_pred,lead_time))
            predvar[rmm] = np.zeros((lead_time))
            biasvar[rmm] = np.zeros((n_pred,lead_time))
        
        y_mean = {}
        y_cov = {}
        for i, pred_i in enumerate(pred_ids[:n_pred]):
            for j in range(lead_time):
                obs['phase'][i,j] = npzfile['phase'][val_ids][pred_i+j+width]
                obs['amplitude'][i,j] = npzfile['amplitude'][val_ids][pred_i+j+width]
                input_x = np.hstack(( npzfile['RMM1'][val_ids][pred_i+j : pred_i+j+width], \
                                      npzfile['RMM2'][val_ids][pred_i+j : pred_i+j+width] )) # (2*width, ) array

                if j == 0:
                    mean_val_x = input_x # (2*width, ) array
                else:
                    mean_val_x_1 = np.hstack(( mean_val_x[1:width], y_mean[i,j-1][0] )) # (width, ) array
                    mean_val_x_2 = np.hstack(( mean_val_x[width+1:2*width], y_mean[i,j-1][1] )) # (width, ) array
                    mean_val_x = np.hstack(( mean_val_x_1, mean_val_x_2 )) # (2*width, ) array
                    
                y_mean[i,j] = self.mu2[j][:,None] + self.K21_11[j] @ (mean_val_x[:,None] - self.mu1[j][:,None]) # (2, 1) numpy array
                #y_cov[i,j] = self.Kcond[j] #+ sample_cov (2,2) numpy array

                for k, rmm in enumerate(rmms):
                    predvar[rmm][j] = self.Kcond[j][k,k]#y_cov[i,j][k,k]
                    obs[rmm][i,j] = npzfile[rmm][val_ids][pred_i+j+width]
                    biasvar[rmm][i,j] = (y_mean[i,j][k,0] - obs[rmm][i,j])**2

        std = {}
        var = {}
        for rmm in rmms:
            biasvar_mean = np.sum(biasvar[rmm],axis=0) / (n_pred) # (lead_time, ) array
            var[rmm] = predvar[rmm] + biasvar_mean # (lead_time, ) array
            std[rmm] = np.sqrt(var[rmm]) # (lead_time, ) array

        Kcond_correct = {}
        for j in range(lead_time):
            rmm1_correct = var['RMM1'][j]
            rmm2_correct = var['RMM2'][j]
            rmm12_correct = np.sqrt( rmm1_correct*rmm2_correct ) * self.Kcond[j][0,1] / np.sqrt( self.Kcond[j][0,0] * self.Kcond[j][1,1] )
            rmm21_correct = np.sqrt( rmm1_correct*rmm2_correct ) * self.Kcond[j][1,0] / np.sqrt( self.Kcond[j][0,0] * self.Kcond[j][1,1] )
            Kcond_correct[j] = np.array( [ [rmm1_correct, rmm12_correct], [rmm21_correct, rmm2_correct] ] ) # (2, 2) array


        self.lead_time = lead_time
        self.v = v
        self.start_val = start_val
        self.std = std
        self.Kcond_correct = Kcond_correct
        

    def pred(self, start_test=None, test_ids=None, n_pred=1, m=1500, season_bool=False):
        npzfile = self.npzfile
        width = self.width
        lead_time = self.lead_time
        d = self.d
        Ns = 1000#self.train_data.shape[0] # the number of samplings is set to the number of training
        if start_test is not None and test_ids is None:
            test_ids = np.arange( start_test, min( start_test+m, len(npzfile['id']) ) )
    
        if season_bool: #or len(test_id_split) > 2:
            test_id_split = np.hstack( ( np.array([0]), 
                            np.where(np.ediff1d(npzfile['id'][test_ids]) != 1 )[0]+1, 
                            np.array([len(npzfile['id'][test_ids])]) ) )
            diff_test_ids = np.ediff1d(test_id_split)
            max_diff = np.max(diff_test_ids)
            freq_diff = np.bincount(diff_test_ids).argmax() # return the most frequent value in diff_test_ids
            
            if width >= freq_diff:
                raise ValueError(f'the width is greater than the season interval, please try a width value < {freq_diff}')
            if lead_time + width > freq_diff:
                print(f"the sum of the width and lead time is greater than the season interval..., will set lead time = {freq_diff-width}")
                lead_time = freq_diff-width
            
            pred_ids_start = test_id_split[:-1]
            pred_ids_end = (test_id_split - lead_time - width + 1)[1:]
            pred_ids = np.array([],dtype=int)
            for ii in range(len(pred_ids_start)):
                if (pred_ids_start[ii] >= pred_ids_end[ii]):
                    continue
                pred_ids_i = np.arange( start=pred_ids_start[ii], stop=pred_ids_end[ii] )
                pred_ids = np.hstack( (pred_ids,pred_ids_i), dtype=int)
            
            if n_pred > len(pred_ids):
                print(f"the number of predictions is greater than the number of the season intervals..., will set n_pred = {len(pred_ids)}")
                n_pred = len(pred_ids)            
            
        else:
            pred_ids = np.arange(n_pred)
        
        obs = {}
        observed_preds = {}
        lower_confs = {}
        upper_confs = {}

        obs['year'] = np.zeros((n_pred,lead_time))
        obs['month'] = np.zeros((n_pred,lead_time))
        obs['day'] = np.zeros((n_pred,lead_time))
        obs['phase'] = np.zeros((n_pred,lead_time))
        obs['amplitude'] = np.zeros((n_pred,lead_time))

        rmms = ['RMM1','RMM2']
        for rmm in rmms:
            obs[rmm] = np.zeros((n_pred,lead_time))
            observed_preds[rmm] = np.zeros((n_pred,lead_time))
            lower_confs[rmm] = np.zeros((n_pred,lead_time))
            upper_confs[rmm] = np.zeros((n_pred,lead_time))
        
        y_mean = {}#[[0]*lead_time]*n_pred
        y_cov = {}#[[0]*lead_time]*n_pred
        for i, pred_i in enumerate(pred_ids[:n_pred]):
            for j in range(lead_time):
                obs_start_id = (test_ids + width)[pred_i] + j
                obs['year'][i,j] = npzfile['year'][obs_start_id]
                obs['month'][i,j] = npzfile['month'][obs_start_id]
                obs['day'][i,j] = npzfile['day'][obs_start_id]
                obs['phase'][i,j] = npzfile['phase'][obs_start_id]
                obs['amplitude'][i,j] = npzfile['amplitude'][obs_start_id]
                input_x = np.hstack(( npzfile['RMM1'][obs_start_id-width : obs_start_id], \
                                      npzfile['RMM2'][obs_start_id-width : obs_start_id] )) # (2*width, ) array

                if j == 0:
                    mean_test_x = input_x # (2*width, ) array
                    #cov_test_x = np.zeros((2*width,2*width)) # (2*width, 2*width) array
                else:
                    mean_test_x_1 = np.hstack(( mean_test_x[1:width], y_mean[i,j-1][0] )) # (width, ) array
                    mean_test_x_2 = np.hstack(( mean_test_x[width+1:2*width], y_mean[i,j-1][1] )) # (width, ) array
                    mean_test_x = np.hstack(( mean_test_x_1, mean_test_x_2 )) # (2*width, ) array
                

                # make predictions by empirical mean and covaraince
                y_mean[i,j] = self.mu2[j][:,None] + self.K21_11[j] @ (mean_test_x[:,None] - self.mu1[j][:,None]) # (2, 1) numpy array
                # y_cov[i,j] = K22 - K21 @ K11_inv @ K12 + sample_cov # (2,2) numpy array

                # generate test_x samplings, test_x ~ N(mean_test_x, cov_test_x), test_x = [z_{k-width},...,z_{k-1}]
                #test_x_samples = np.random.multivariate_normal(mean=mean_test_x, cov=cov_test_x, size=Ns) # (Ns, 2*width) array

                # # generate y samplings, y = [z_{k}], y ~ N(y_mean, y_var)
                # y_samples = np.random.multivariate_normal(mean=y_mean[i,j][:,0], cov=y_cov[i,j], size=Ns) # (Ns, 2) array
                
                
                for k, rmm in enumerate(rmms):
                    obs[rmm][i,j] = npzfile[rmm][obs_start_id]
                    observed_preds[rmm][i,j] = y_mean[i,j][k,0]
                    lower_confs[rmm][i,j] = y_mean[i,j][k,0] - 1.00 * self.std[rmm][j] # np.sqrt(y_cov[i,j][k,k])
                    upper_confs[rmm][i,j] = y_mean[i,j][k,0] + 1.00 * self.std[rmm][j] # np.sqrt(y_cov[i,j][k,k])

        self.lead_time = lead_time
        self.n_pred = n_pred
        self.m = m
        self.start_test = start_test
        self.pred_ids = pred_ids
        
        self.obs = obs
        self.observed_preds = observed_preds
        self.lower_confs = lower_confs
        self.upper_confs = upper_confs

        self.test_ids = test_ids


    def rmm_to_phase(self, pred_rmm1=None, pred_rmm2=None):
        if pred_rmm1 is None and pred_rmm2 is None:
            pred_rmm1 = self.observed_preds['RMM1']
            pred_rmm2 = self.observed_preds['RMM2']
        rmm_angle = np.arctan2(pred_rmm2,pred_rmm1) * 180 / np.pi
        phase = np.zeros(pred_rmm1.shape)

        for i in range(8):
            lower_angle = - 180. * (1 - i / 4.)
            upper_angle = - 180. * (1 - (i+1) / 4.)
            bool_angle = (rmm_angle > lower_angle) & (rmm_angle <= upper_angle)
            phase += bool_angle.astype('int64')*(i+1)
       
        self.observed_preds['phase'] = phase.astype(int)
        return self.observed_preds['phase']

    def rmm_to_amplitude(self, pred_rmm1=None, pred_rmm2=None):
        """ampltitude is the norm of (RMM1, RMM2)
        """
        if pred_rmm1 is None and pred_rmm2 is None:
            pred_rmm1 = self.observed_preds['RMM1']
            pred_rmm2 = self.observed_preds['RMM2']
        amplitude = np.sqrt( np.square(pred_rmm1) + np.square(pred_rmm2) )
        self.observed_preds['amplitude'] = amplitude
        return amplitude

    
    def cor(self, pred_rmm1=None, pred_rmm2=None, obs_rmm1=None, obs_rmm2=None):
        """bivariate correlation coefficien
        """
        if pred_rmm1 is None and pred_rmm2 is None:
            pred_rmm1 = self.observed_preds['RMM1']
            pred_rmm2 = self.observed_preds['RMM2']

        if obs_rmm1 is None and obs_rmm2 is None:
            obs_rmm1 = self.obs['RMM1']
            obs_rmm2 = self.obs['RMM2']
        
        lead_time = min(pred_rmm1.shape[1], obs_rmm1.shape[1])

        pred_rmm1 = pred_rmm1[:,:lead_time]
        pred_rmm2 = pred_rmm2[:,:lead_time]
        obs_rmm1 = obs_rmm1[:,:lead_time]
        obs_rmm2 = obs_rmm2[:,:lead_time]

        numerator = np.sum((obs_rmm1*pred_rmm1 + obs_rmm2*pred_rmm2), axis=0) # 1*lead_time numpy array
        denominator = np.sqrt(np.sum((obs_rmm1**2 + obs_rmm2**2),axis=0)) \
            * np.sqrt(np.sum((pred_rmm1**2 + pred_rmm2**2),axis=0))
        self.errs['cor'] = (numerator / denominator).reshape(-1) # shape = (lead_time,) numpy array

        return self.errs['cor']
    
   
    def rmse(self, pred_rmm1=None, pred_rmm2=None, obs_rmm1=None, obs_rmm2=None):
        if pred_rmm1 is None and pred_rmm2 is None:
            pred_rmm1 = self.observed_preds['RMM1']
            pred_rmm2 = self.observed_preds['RMM2']

        if obs_rmm1 is None and obs_rmm2 is None:
            obs_rmm1 = self.obs['RMM1']
            obs_rmm2 = self.obs['RMM2']
        
        lead_time = min(pred_rmm1.shape[1], obs_rmm1.shape[1])

        pred_rmm1 = pred_rmm1[:,:lead_time]
        pred_rmm2 = pred_rmm2[:,:lead_time]
        obs_rmm1 = obs_rmm1[:,:lead_time]
        obs_rmm2 = obs_rmm2[:,:lead_time]

        n_pred = pred_rmm1.shape[0]

        sum_rmm1 = np.sum((obs_rmm1-pred_rmm1)**2, axis=0)
        sum_rmm2 = np.sum((obs_rmm2-pred_rmm2)**2, axis=0)
        self.errs['rmse'] = ( np.sqrt( (sum_rmm1 + sum_rmm2) / n_pred ) ).reshape(-1)

        return self.errs['rmse']


    def phase_err(self, pred_rmm1=None, pred_rmm2=None, obs_rmm1=None, obs_rmm2=None):
        if pred_rmm1 is None and pred_rmm2 is None:
            pred_rmm1 = self.observed_preds['RMM1']
            pred_rmm2 = self.observed_preds['RMM2']

        if obs_rmm1 is None and obs_rmm2 is None:
            obs_rmm1 = self.obs['RMM1']
            obs_rmm2 = self.obs['RMM2']
        
        lead_time = min(pred_rmm1.shape[1], obs_rmm1.shape[1])

        pred_rmm1 = pred_rmm1[:,:lead_time]
        pred_rmm2 = pred_rmm2[:,:lead_time]
        obs_rmm1 = obs_rmm1[:,:lead_time]
        obs_rmm2 = obs_rmm2[:,:lead_time]

        n_pred = pred_rmm1.shape[0]

        num = obs_rmm1*pred_rmm2 - obs_rmm2*pred_rmm1
        den = obs_rmm1*pred_rmm1 + obs_rmm2*pred_rmm2

        temp = np.arctan2(pred_rmm2, pred_rmm1) * 180 / np.pi  - np.arctan2(obs_rmm2, obs_rmm1) * 180 / np.pi
        #temp = np.arctan(np.divide(num,den))
        self.errs['phase'] = ( np.sum(temp, axis=0) / n_pred ).reshape(-1)

        return self.errs['phase']

   
    def amplitude_err(self, pred_rmm1=None, pred_rmm2=None, obs_rmm1=None, obs_rmm2=None):
        if pred_rmm1 is None and pred_rmm2 is None:
            pred_rmm1 = self.observed_preds['RMM1']
            pred_rmm2 = self.observed_preds['RMM2']
        
        if obs_rmm1 is None and obs_rmm2 is None:
            obs_rmm1 = self.obs['RMM1']
            obs_rmm2 = self.obs['RMM2']
        
        lead_time = min(pred_rmm1.shape[1], obs_rmm1.shape[1])

        pred_rmm1 = pred_rmm1[:,:lead_time]
        pred_rmm2 = pred_rmm2[:,:lead_time]
        obs_rmm1 = obs_rmm1[:,:lead_time]
        obs_rmm2 = obs_rmm2[:,:lead_time]


        n_pred = pred_rmm1.shape[0]
        preds_amplitude = self.rmm_to_amplitude(pred_rmm1,pred_rmm2)
        
        obs_amplitude = self.rmm_to_amplitude(obs_rmm1,obs_rmm2)
        # obs_amplitude = self.obs['amplitude'][:,:lead_time]

        self.errs['amplitude'] = (np.sum((preds_amplitude - obs_amplitude),axis=0) / n_pred).reshape(-1)

        return self.errs['amplitude']

    def hss(self, critic_val=0.05):
        pred_rmm1 = self.observed_preds['RMM1']
        pred_rmm2 = self.observed_preds['RMM2']
        preds_phase = self.rmm_to_phase(pred_rmm1,pred_rmm2) # (n_pred,lead_time)
        preds_amplitude = self.rmm_to_amplitude(pred_rmm1,pred_rmm2) # (n_pred,lead_time)

        obs_phase = self.obs['phase'] # (n_pred,lead_time)
        obs_amplitude = self.obs['amplitude'] # (n_pred,lead_time)

        lead_time = pred_rmm1.shape[1]
        hss_phase = np.zeros((9,lead_time))
        hss_n = np.zeros((9,lead_time))
        hss_signif = np.zeros((9,lead_time), dtype=str)
        for phase_num in np.arange(9):
            if phase_num == 0: # whether the mjo is strong / whether it is a mjo event
                a = np.count_nonzero( (preds_amplitude < 1) & (obs_amplitude <1), axis=0) # (lead_time, ) array
                b = np.count_nonzero( (preds_amplitude < 1) & (obs_amplitude >= 1), axis=0) # (lead_time, ) array
                c = np.count_nonzero( (preds_amplitude >= 1) & (obs_amplitude < 1), axis=0) # (lead_time, ) array
                d = np.count_nonzero( (preds_amplitude >= 1) & (obs_amplitude >= 1), axis=0) # (lead_time, ) array
            else:
                a = np.count_nonzero( ( (preds_phase == phase_num) & (preds_amplitude >= 1) ) & ( (obs_phase == phase_num) & (obs_amplitude >= 1) ), axis=0)
                b = np.count_nonzero( ( (preds_phase == phase_num) & (preds_amplitude >= 1) ) & ( (obs_phase != phase_num) | (obs_amplitude < 1) ), axis=0)
                c = np.count_nonzero( ( (preds_phase != phase_num) | (preds_amplitude < 1) ) & ( (obs_phase == phase_num) & (obs_amplitude >= 1) ), axis=0)
                d = np.count_nonzero( ( (preds_phase != phase_num) | (preds_amplitude < 1) ) & ( (obs_phase != phase_num) | (obs_amplitude < 1) ), axis=0)
            
            n = a + b + c + d # (lead_time, ) array
            pc = (a[n != 0] + d[n != 0]) / n[n != 0] # (lead_time, ) array
            expect = ( (a[n != 0]+c[n != 0])*(a[n != 0]+b[n != 0]) + (b[n != 0]+d[n != 0])*(c[n != 0]+d[n != 0]) ) / (n[n != 0]**2)  # (lead_time, ) array or expect=0.5
            expect[expect==1] = 1 - 1e-6
            hss_phase[phase_num,n!=0] = (pc - expect) / (1 - expect) # (lead_time, ) array
            hss_phase[phase_num,n==0] = np.inf
            hss_n[phase_num,:] = n

            # test for confusion table (contigency table)
            for j in range(lead_time):
                confusion_tab = np.array([[a[j], b[j]], [c[j], d[j]]])
                
                # Fisher's exact test (https://en.wikipedia.org/wiki/Fisher%27s_exact_test)
                res = fisher_exact(confusion_tab, alternative='greater')

                if res.pvalue < critic_val: 
                    hss_signif[phase_num,j] = 'X' # significant bin
                else:
                    hss_signif[phase_num,j] = ''


        self.errs['hss'] = hss_phase
        self.errs['hss_n'] = hss_n
        self.errs['hss_signif'] = hss_signif
        return self.errs['hss']


    def crps(self, pred_mean_rmm1=None, pred_mean_rmm2=None, pred_std_rmm1=None, pred_std_rmm2=None, obs_rmm1=None, obs_rmm2=None):
        rmms = ['RMM1','RMM2']
        pred_means = {}
        pred_stds = {}
        obs = {}
        
        if pred_mean_rmm1 is None and pred_mean_rmm2 is None:
            pred_mean_rmm1 = self.observed_preds['RMM1']
            pred_mean_rmm2 = self.observed_preds['RMM2']
        
        if pred_std_rmm1 is None and pred_std_rmm2 is None:
            pred_std_rmm1 = self.std['RMM1']
            pred_std_rmm2 = self.std['RMM2']
        
        if obs_rmm1 is None and obs_rmm2 is None:
            obs_rmm1 = self.obs['RMM1']
            obs_rmm2 = self.obs['RMM2']
        
        lead_time = min(pred_mean_rmm1.shape[1], obs_rmm1.shape[1])

        pred_means['RMM1'] = pred_mean_rmm1[:,:lead_time]
        pred_means['RMM2'] = pred_mean_rmm2[:,:lead_time]

        pred_stds['RMM1'] = pred_std_rmm1[...,:lead_time]
        pred_stds['RMM2'] = pred_std_rmm2[...,:lead_time]

        obs['RMM1'] = obs_rmm1[:,:lead_time] # [n_pred, lead_time]-shape array
        obs['RMM2'] = obs_rmm2[:,:lead_time]
        
        out = 0
        sub_val = 1 / np.sqrt(np.pi).item()

        for rmm in rmms:    

            obs_scale = ( obs[rmm] - pred_means[rmm] ) / pred_stds[rmm] # [n_pred, lead_time]-shape array
            cdf_val = norm.cdf(obs_scale, loc=0, scale=1) # [n_pred, lead_time]-shape array
            pdf_val = norm.pdf(obs_scale, loc=0, scale=1) # [n_pred, lead_time]-shape array
            

            crps_unscale = np.multiply(obs[rmm], 2*cdf_val-1) + 2 * pdf_val - sub_val # [n_pred, lead_time]-shape array
            out += pred_stds[rmm] * crps_unscale # [n_pred, lead_time]-shape array

        self.errs['crps'] = np.mean(out, axis=0) # [lead_time, ]-shape array
        return self.errs['crps']

    def mll(self, pred_mean_rmm1=None, pred_mean_rmm2=None, 
            pred_std_rmm1=None, pred_std_rmm2=None, pred_crosscov=None, 
            obs_rmm1=None, obs_rmm2=None):
        
        add_val = np.log(2 * np.pi).item()
        
        if pred_mean_rmm1 is None and pred_mean_rmm2 is None:    
            pred_means = np.stack((self.observed_preds['RMM1'],self.observed_preds['RMM2']),axis=-1) # [n_pred, lead_time, 2]-shape array
        else:
            pred_means = np.stack((pred_mean_rmm1,pred_mean_rmm2),axis=-1) # (n_pred, lead_time, 2)-shape array
        
        if obs_rmm1 is None and obs_rmm2 is None:
            obs = np.stack((self.obs['RMM1'], self.obs['RMM2']), axis=-1) # [n_pred, lead_time, 2]-shape array
        else:
            obs = np.stack((obs_rmm1, obs_rmm2), axis=-1)
        
        lead_time = min(pred_means.shape[1], obs.shape[1])
        n_pred = pred_means.shape[0]
        pred_means = pred_means[:,:lead_time,:]
        obs = obs[:,:lead_time,:]
        

        diff = pred_means - obs # [n_pred, lead_time, 2]-shape array
        diff_right = diff[...,None] # [n_pred, lead_time, 2, 1]-shape
        diff_left = diff[...,None,:] # [n_pred, lead_time, 1, 2]-shape

        if pred_std_rmm1 is None and pred_std_rmm2 is None and pred_crosscov is None: # GP-model, use self.Kcond_correct
            pred_covs = np.zeros((n_pred, lead_time, 2, 2))
            for j in range(lead_time):
                pred_covs[:,j,:,:] = self.Kcond_correct[j][None,...].repeat(n_pred, axis=0) # (n_pred, 2, 2)-shape array
        else:
            pred_var_rmm1 = pred_std_rmm1[:,:lead_time] ** 2 # [n_pred, lead_time]-shape
            pred_var_rmm2 = pred_std_rmm2[:,:lead_time] ** 2 # [n_pred, lead_time]-shape

            # stack1 = np.insert(arr=pred_var_rmm1[...,None], obj=[1], values=0, axis=-1) # (n_pred, lead_time, 2)-shape array
            # stack2 = np.insert(arr=pred_var_rmm2[...,None], obj=[0], values=0, axis=-1) # (n_pred, lead_time, 2)-shape array
            # pred_covs = np.stack((stack1, stack2), axis=-1) # (n_pred, lead_time, 2, 2)-shape array

            pred_covs = np.zeros((n_pred, lead_time, 2, 2))
            pred_covs[:,:, 0, 0] = pred_var_rmm1
            pred_covs[:,:, 1, 1] = pred_var_rmm2
            pred_covs[:,:, 0, 1] = pred_covs[:,:, 1, 0] = pred_crosscov

        qua_term = np.matmul( np.matmul(diff_left, pred_covs), diff_right).squeeze() # [n_pred, lead_time]-shape
        det_term = np.linalg.det(pred_covs) # [n_pred, lead_time]-shape

        mll_total = (-0.5) * (det_term + qua_term + add_val) # [n_pred, lead_time]-shape
        self.errs['mll'] = mll_total.mean(axis=0) # (lead_time, )-shape

        return self.errs['mll']