import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import datetime as dt
import h5py
import fxcmpy
from utils.globals import EPS, MAIN_DIR, CSV_DIR
from utils.data import write_fx_to_csv


def pandas_fill(arr):
    
    arr = arr.astype('float')
    arr[arr == 0] = 'nan'
    
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=1, inplace=True)
    out = df.as_matrix()
    return out


def create_fxcm_dataset(instruments, period, start_date, end_date):
    #   conn = fxcmpy.fxcmpy(config_file='config/fxcm.cfg', server='demo')
    conn = fxcmpy.fxcmpy(access_token = 'ceca822890e80080bcdb7128990b2f5312ab8654', server='demo', log_level = 'error')
    #   history = np.empty(shape=(len(instruments), 0, 14), dtype=np.float)
    history = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    bid = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    ask = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    
    m = 0
    '''
    npdata = data.to_numpy()
    npdata = npdata [:,1:6]
    
    print(npdata)
    print(np.shape(npdata))
    '''

    for instrument in instruments:
        
        data = conn.get_candles(instrument, period=period, start=start_date, stop=end_date)
        data.index = pd.to_datetime(data.index)
        
        #   Price 
        
        Open = pd.DataFrame(data[['bidopen', 'askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh', 'askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow', 'asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose', 'askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
        #   weekday = pd.DataFrame(data[['weekday']].mean(axis=1), columns=['weekday'])
        #   hour = pd.DataFrame(data[['hour']].mean(axis=1), columns=['hour'])
                
        df_data = pd.concat([Open, High, Low, Close, Volume], axis=1)

        npdata =  df_data.to_numpy()
        
        #   bid 
        
        Open = pd.DataFrame(data[['bidopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
        #   weekday = pd.DataFrame(data[['weekday']].mean(axis=1), columns=['weekday'])
        #   hour = pd.DataFrame(data[['hour']].mean(axis=1), columns=['hour'])
        
        df_bid = pd.concat([Open, High, Low, Close, Volume], axis=1)

        npbid = df_bid.to_numpy()
        
        #   ask 
        
        Open = pd.DataFrame(data[['askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
        #   weekday = pd.DataFrame(data[['weekday']].mean(axis=1), columns=['weekday'])
        #   hour = pd.DataFrame(data[['hour']].mean(axis=1), columns=['hour'])
        
        df_ask = pd.concat([Open, High, Low, Close, Volume], axis=1)

        npask = df_ask.to_numpy()
        
        history = np.resize(history, (history.shape[0], npdata.shape[0], history.shape[2]))
        bid = np.resize(history, (bid.shape[0], npbid.shape[0], bid.shape[2]))
        ask = np.resize(history, (ask.shape[0], npask.shape[0], ask.shape[2]))

        for d in range(npdata.shape[0]):   
            #   print(np.shape(history))
            #   history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4],npdata[d,5], npdata[d,6], npdata[d,7], npdata[d,8], npdata[d,9], npdata[d,10], npdata[d,11], npdata[d,12], npdata[d,13]])
            history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4]])
            bid[m][d] = np.array([npbid[d,0], npbid[d,1], npbid[d,2], npbid[d,3], npbid[d,4]])
            ask[m][d] = np.array([npask[d,0], npask[d,1], npask[d,2], npask[d,3], npask[d,4]])
    
        m += 1
        
        length = npdata.shape[0]
    
        print(length)
    
    return history, bid, ask, instruments, length
    
    #   write_fx_to_h5py(history, bid, ask, instruments, data_file, False)


if __name__ == '__main__':
    start_date =    dt.datetime(2015, 1, 1)
    end_date =    dt.datetime(2019, 12, 31)
    train_test_split = 0    #   0.75
    file_name_begin = 'fxcm_EURUSD_'
    
    delta = (end_date - start_date).days
    days_add =  int(delta * train_test_split) 
    
    start_date_train = start_date
    end_date_train = start_date + dt.timedelta(days=days_add)
    
    start_date_test = end_date_train + dt.timedelta(days=1)
    end_date_test = end_date
    
    print(start_date_train)
    print(end_date_train)
    print(start_date_test)
    print(end_date_test)    
    
    period = 'H4'   #   'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'D1', 'W1', 'M1'
    #   instruments = ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD', 'FRA40', 'GER30', 'JPN225', 'SPX500', 'UK100', 'US30', 'USOil']
    #   instruments = ['AUD/USD', 'AUS200', 'Bund', 'CHN50', 'Copper', 'ESP35', 'EUR/USD', 'EUSTX50', 'FRA40', 'GBP/USD', 'GER30', 'HKG33', 'JPN225', 'NAS100', 'NGAS', 'NZD/USD', 'SOYF', 'SPX500', 'UK100', 'UKOil', 'US30', 'USDOLLAR', 'USOil', 'XAG/USD', 'XAU/USD', 'US2000', 'WHEATF', 'CORNF', 'EMBasket', 'JPYBasket', 'USEquities']
    #   instruments = ['AUD/USD', 'AUS200', 'Bund', 'CHN50', 'Copper', 'ESP35', 'EUR/USD', 'EUSTX50', 'FRA40', 'GBP/USD', 'GER30', 'HKG33', 'JPN225', 'NAS100', 'NGAS', 'NZD/USD', 'SOYF', 'SPX500', 'UK100', 'UKOil', 'US30', 'USDOLLAR', 'USOil', 'XAG/USD', 'XAU/USD', 'US2000', 'WHEATF', 'CORNF', 'EMBasket', 'JPYBasket', 'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD', 'CryptoMajor', 'USEquities']
    #   instruments = ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD']
    instruments = ['EUR/USD']
    #   instruments = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',  'GBP/USD', 'NZD/USD', 'GBP/JPY', 'EUR/JPY', 'AUD/JPY', 'EUR/GBP', 'USD/CHF']
    #   instruments = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'GBP/USD', 'USD/CHF']
    
    # create train dataset
    history, bid, ask, instruments, length = create_fxcm_dataset(instruments, period, start_date_train, end_date_train)
    file_name_end = '_train_' + str(length) + '.csv'
    data_file = CSV_DIR + file_name_begin + period + '_' + str(start_date)[0:4] + '_' + str(end_date)[0:4] + file_name_end       
    write_fx_to_csv(history, bid, ask, instruments, data_file, False)
    
    # create test dataset
    history, bid, ask, instruments, length = create_fxcm_dataset(instruments, period, start_date_test, end_date_test)
    file_name_end = '_test_' + str(length) + '.csv'
    data_file = CSV_DIR + file_name_begin + period + '_' + str(start_date)[0:4] + '_' + str(end_date)[0:4] + file_name_end       
    write_fx_to_csv(history, bid, ask, instruments, data_file, False)