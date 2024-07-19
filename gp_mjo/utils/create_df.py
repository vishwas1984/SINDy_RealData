import pandas as pd
from datetime import datetime

def create_dataframe(npzfile):
    # Create a date_time column in YYYY-MM-DD format
    dates = [datetime(year, month, day).strftime('%Y-%m-%d') 
             for year, month, day in zip(npzfile['year'], npzfile['month'], npzfile['day'])]

    # Prepare the data for the DataFrame
    data = {
        'value': list(npzfile['RMM1']) + list(npzfile['RMM2']),
        'group': ['RMM1'] * len(npzfile['RMM1']) + ['RMM2'] * len(npzfile['RMM2']),
        'time_idx': list(npzfile['id']) + list(npzfile['id']),
        'date': dates + dates,
        'phase': list(npzfile['phase']) + list(npzfile['phase']),
        'amplitude': list(npzfile['amplitude']) + list(npzfile['amplitude'])
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    
    return df