from typing import Union
import numpy as np
import torch

def dat_divide(new_data, n1, m, n, c, fixed_start,start_index, width):
    dic = {}
    dic_ids = {}

    dic['train1'] = new_data[:n1]
    dic_ids['train1'] = np.arange(0,n1)

    if fixed_start:
        dic['test'] = new_data[start_index-width:n1+m] # start_index should >= max(width) + n1 + 1
        dic_ids['test'] = np.arange(start_index-width,n1+m)
    else:
        dic['test'] = new_data[n1:n1+m]
        dic_ids['test'] = np.arange(n1,n1+m)
    
    dic['buffer'] = new_data[n1+m:n1+m+c]
    dic_ids['buffer'] = np.arange(n1+m, n1+m+c)
    
    dic['train2'] = new_data[n1+m+c:n+m+c]
    dic_ids['train2'] = np.arange(n1+m+c, n+m+c)
    return dic, dic_ids


def dics_divide(new_datas, data_names, n1, m, n, c, fixed_start=False, start_index=None, width=None):
    dics = {}
    dics_ids = {}
    for new_data, data_name in zip(new_datas, data_names):
        dics[data_name], dics_ids[data_name] = dat_divide(new_data, n1, m, n, c, fixed_start, start_index, width)
    return dics, dics_ids


# n1 = n1s[0]
# dics, dics_ids = dics_divide(new_datas, data_names, n1, m, n, c, fixed_start=False, start_index=None, width=None)
# print(dics['RMM1']['train1'])

# def dics_tot_divide(new_datas, data_names, n1s, m, n, c, fixed_start=False, start_index=None, width=None):
#     dics_total = {}
#     dics_temp = {}
#     for n1 in n1s:
#         for new_data, data_name in zip(new_datas, data_names):
#             dics_temp[data_name], ids_temp = dat_divide(new_data, n1, m, n, c, fixed_start, start_index, width)
#         dics_total[n1] = dics_temp
#     return dics_total

# dics_total = dics_tot_divide(new_datas, data_names, n1s, m, n, c, fixed_start=False, start_index=None, width=None)
# print(dics_total[n1s[0]]['RMM1']['train1'])



def rolling(a: Union[np.ndarray, torch.Tensor], width: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert an array or tensor into a matrix with a rolling window.
    
    ------------------------
    Parameters:
    a: 1D numpy array or torch tensor, a = [a_1,...,a_n]
    width: the width of the rolling window
    
    ------------------------
    Returns:
    a_roll: 2D numpy array or torch tensor with rolling windows
    a_roll =  [a_1,a_2,...,a_{wid};
            a_2,a_3,...,a_{wid+1};
            ...;
            a_{n-wid+1},a_{n-wid+2},...,a_n], is a (n-wid+1)*(wid) numpy arrray or tensor
    """
    if isinstance(a, np.ndarray):
        if a.size == 0:
            return a.reshape((-1, width))
        shape = (a.size - width + 1, width)
        strides = (a.itemsize, a.itemsize)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    elif isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return a.reshape(-1, width)
        return a.unfold(dimension=0, size=width, step=1) # [n-width+1, width]-size tensor

    else:
        raise TypeError('Invalid input type')