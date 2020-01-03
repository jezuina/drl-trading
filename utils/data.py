import h5py
import numpy as np
import pandas as pd
import fxcmpy
import datetime as dt


def pandas_fill(arr):
    
    arr = arr.astype('float')
    arr[arr == 0] = 'nan'
    
    df = pd.DataFrame(arr)
    
    print(df)
    
    #   fillna(method='ffill').fillna(method='bfill')
    df.fillna(method='ffill', axis=0, inplace=True)
    df.fillna(method='bfill', axis=0, inplace=True)
    out = df.as_matrix()
    
    return out

    
def read_h5_history(filepath, replace_zeros=True):
    ''' Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        instruments:

    '''
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        if replace_zeros:
            np.nan_to_num(history, 0)
            mn = np.median(history[history > 0])
            history[history==0] = mn
        #   instruments = ['rand']

        instruments = f['instruments'][:].tolist()
        instruments = [instrument.decode('utf-8') for instrument in instruments]
        '''
        try:    
            instruments = f['instruments'][:].tolist()
            instruments = [instrument.decode('utf-8') for instrument in instruments]
        except:
            instruments = f['instruments']   
        '''
    print('history has NaNs: {}, has infs: {}'.format(np.any(np.isnan(history)), np.any(np.isinf(history))))
        
    return history, instruments


def read_h5_fx_history(filepath, replace_zeros=True):
    ''' Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        instruments:

    '''
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        bid = f['history'][:]
        ask = f['history'][:]
        
        if replace_zeros:
            np.nan_to_num(history, 0)
            mn = np.median(history[history > 0])
            history[history==0] = mn
        
            np.nan_to_num(bid, 0)
            mn = np.median(bid[bid > 0])
            history[bid==0] = mn
            
            np.nan_to_num(ask, 0)
            mn = np.median(ask[ask > 0])
            history[ask==0] = mn

        instruments = f['instruments'][:].tolist()
        instruments = [instrument.decode('utf-8') for instrument in instruments]
        '''
        try:    
            instruments = f['instruments'][:].tolist()
            instruments = [instrument.decode('utf-8') for instrument in instruments]
        except:
            instruments = f['instruments']   
        '''
    print('history has NaNs: {}, has infs: {}'.format(np.any(np.isnan(history)), np.any(np.isinf(history))))
    print('bid has NaNs: {}, has infs: {}'.format(np.any(np.isnan(bid)), np.any(np.isinf(bid))))
    print('ask has NaNs: {}, has infs: {}'.format(np.any(np.isnan(ask)), np.any(np.isinf(ask))))
        
    return history, bid, ask, instruments


def write_to_h5py(history, instruments, filepath):
    """ Write a numpy array history and a list of string to h5py
    Args:
        history: (N, timestamp, 5)
        instruments: a list of stock instruments
    Returns:
    """
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('history', data=history)
        instruments_array = np.array(instruments, dtype=object)
        # string_dt = h5py.special_dtype(vlen=str)
        string_dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset("instruments", data=instruments_array, dtype=string_dt)


def write_fx_to_h5py(history, bid, ask, instruments, filepath, fill_missing_values):
    """ Write a numpy array history and a list of string to h5py
    Args:
        history: (N, timestamp, 5)
        instruments: a list of stock instruments
    Returns:
    """
    
    if fill_missing_values:
        history = pandas_fill(history)    
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('history', data=history)
        f.create_dataset('bid', data=bid)
        f.create_dataset('ask', data=ask)
        instruments_array = np.array(instruments, dtype=object)
        # string_dt = h5py.special_dtype(vlen=str)
        string_dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset("instruments", data=instruments_array, dtype=string_dt)
        

def download_fxcm_data(data_file, instruments, period, start_date, end_date):
    conn = fxcmpy.fxcmpy(config_file='./config/fxcm.cfg', server='demo')
    
    history = np.empty(shape=(len(instruments), 0, 14), dtype=np.float)
    m = 0
    '''
    npdata = data.to_numpy()
    npdata = npdata [:,1:6]
    
    print(npdata)
    print(np.shape(npdata))
    '''

    for instrument in instruments:
        
        data = conn.get_candles(instrument, period=period, start=start_date, stop=end_date)
        
        df_data = pd.DataFrame(data) 
        
        Open = pd.DataFrame(data[['bidopen', 'askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh', 'askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow', 'asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose', 'askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
               
        df = pd.concat([Open, High, Low, Close, Volume, df_data], axis=1)
        #   export_csv = df.to_csv (r'C:\Users\hanna\Desktop\export_dataframe.csv', index = None, header=True) 

        npdata = df.to_numpy()
        
        print(npdata)

        history = np.resize(history, (history.shape[0], npdata.shape[0], history.shape[2]))

        for d in range(npdata.shape[0]):   
            #   print(np.shape(history))
            history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4],npdata[d,5], npdata[d,6], npdata[d,7], npdata[d,8], npdata[d,9], npdata[d,10], npdata[d,11], npdata[d,12], npdata[d,13]])
    
        m += 1
    
    write_to_h5py(history, instruments, data_file)
    

def read_csv_history(data_file, instruments, granularity, price, replace_zero=True):
    ''' Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        instruments:

    '''
    history = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    m = 0
    
    for instrument in instruments:
    
        data_file.replace('instrument', instrument)

        print("=========================")
        print(data_file)

        

        data_file.replace('granularity', granularity)
        
        print("=========================")
        print(data_file)

        
        
        data_file.replace('price', price)

        data = pd.read_csv(data_file)
        
        npdata = data.to_numpy()
        npdata = npdata [:,1:6]
    
        print(np.shape(npdata))
        history = np.resize(history, (history.shape[0], npdata.shape[0], history.shape[2]))
    
        for d in range(npdata.shape[0]):   
            #   print(np.shape(history))
            history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4]])
    
        m += 1
    
    return history