import os
import numpy as np
import pandas as pd

data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dir_path = os.path.join(data_dir, 'data', 'obs')
data_path = os.path.join(dir_path, 'rmm.74toRealtime.txt')
save_path = os.path.join(dir_path, 'mjo_new_data.npz')


def main():
    ####################################################
    # Import dataset
    ####################################################
    ## Construct a dataframe from the csv dataset
    colnames=['year', 'month', 'day', 'RMM1', 'RMM2','phase','amplitude','Final_value']
    df = pd.read_csv(data_path, sep=r'[ \t]+', names=colnames, skiprows=[0], engine='python')
    total_days = len(df)

    ## Convert the columns of the dataframe to the numpy array
    year = df['year'].to_numpy()
    month = df['month'].to_numpy()
    day = df['day'].to_numpy()

    RMM1 = df['RMM1'].to_numpy(dtype='float64')
    RMM2 = df['RMM2'].to_numpy(dtype='float64')
    phase = df['phase'].to_numpy(dtype='float64')
    amplitude = df['amplitude'].to_numpy(dtype='float64')

    print(df)

    ####################################################
    # Delete missing values and their preceding values
    ####################################################
    raw_datas = [RMM1,RMM2,phase,amplitude]
    data_names = ['RMM1', 'RMM2', 'phase', 'amplitude']
    idx_miss = np.where(RMM1 > 1e34)[0] # index of missing value
    n_miss = len(idx_miss) # the number of missing value

    ## Delete the missing values and their predceding values
    new_datas = [0]*4
    for i in range(4):
        raw_data = raw_datas[i]
        data_name = data_names[i]
        dat = np.delete(raw_data, np.arange(max(idx_miss)+1)) # np.delete(raw_datas[i], idx_miss)
        new_datas[i] = dat
        if i == 0:
            RMM1_new = dat
        if i == 1:
            RMM2_new = dat
        if i == 2:
            phase_new = dat
        if i == 3:
            amplitude_new = dat

    raw_ymds = [year, month, day]
    ymd_names = ['year', 'month', 'day']
    new_ymds = [0]*3
    for i in range(3):
        raw_ymd = raw_ymds[i]
        tmd_name = ymd_names[i]
        dat_ymd = np.delete(raw_ymd, np.arange(max(idx_miss)+1)) # np.delete(raw_ymds[i], idx_miss)
        new_ymds[i] = dat_ymd
        if i == 0:
            year_new = dat_ymd
        if i == 1:
            month_new = dat_ymd
        if i == 2:
            day_new = dat_ymd
        

    print(new_datas)
    print(RMM1.shape)
    print(max(idx_miss)+1)
    print(RMM1_new.shape)

    print(new_ymds)
    print(year.shape)
    print(year_new.shape)

    ####################################################
    # Save and load the updated dataset
    ####################################################
    kwds = {'year':year_new, 'month':month_new, 'day':day_new, 
        'RMM1':RMM1_new, 'RMM2':RMM2_new,'phase':phase_new,'amplitude':amplitude_new}
    np.savez(save_path, **kwds)

    npzfile = np.load(save_path, allow_pickle=True)
    print(npzfile.files)
    print(npzfile['RMM1'])
    print(npzfile['RMM2'])
    print(npzfile['year'])
    print(npzfile['month'])
    print(npzfile['day'])

if __name__ == "__main__":
    main()