import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import datetime as dt
import h5py
import fxcmpy


CONN = fxcmpy.fxcmpy(config_file='config/fxcm.cfg', server='demo')


def download_history(instruments, period, number):

    history = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    bid = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    ask = np.empty(shape=(len(instruments), 0, 5), dtype=np.float)
    
    m = 0
    
    for instrument in instruments:
        
        data = CONN.get_candles(instrument, period=period, number=number)

        #   Price 
        
        Open = pd.DataFrame(data[['bidopen', 'askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh', 'askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow', 'asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose', 'askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
               
        df_data = pd.concat([Open, High, Low, Close, Volume], axis=1)
        #   print(df_data)
        npdata =  df_data.to_numpy()
        
        #   bid 
        
        Open = pd.DataFrame(data[['bidopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
               
        df_bid = pd.concat([Open, High, Low, Close, Volume], axis=1)

        npbid = df_bid.to_numpy()
        
        #   ask 
        
        Open = pd.DataFrame(data[['askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
               
        df_ask = pd.concat([Open, High, Low, Close, Volume], axis=1)

        npask = df_ask.to_numpy()
        
        history = np.resize(history, (history.shape[0], npdata.shape[0], history.shape[2]))
        bid = np.resize(bid, (bid.shape[0], npbid.shape[0], bid.shape[2]))
        ask = np.resize(ask, (ask.shape[0], npask.shape[0], ask.shape[2]))

        for d in range(npdata.shape[0]):   
            #   print(np.shape(history))
            #   history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4],npdata[d,5], npdata[d,6], npdata[d,7], npdata[d,8], npdata[d,9], npdata[d,10], npdata[d,11], npdata[d,12], npdata[d,13]])
            history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4]])
            bid[m][d] = np.array([npbid[d,0], npbid[d,1], npbid[d,2], npbid[d,3], npbid[d,4]])
            ask[m][d] = np.array([npask[d,0], npask[d,1], npask[d,2], npask[d,3], npask[d,4]])
    
        m += 1
    
    return history, bid, ask


def get_nargins(instrument, lot_size, leverage):
    None
    #   Not implemeneted


def rebalance(instruments, weights, lot_size, leverage):

    accounts = CONN.get_accounts()
    open_positions = CONN.get_open_positions(kind='dataframe')
     
    current_positions = []
    current_prices = []
        
    m = 0
    for instrument in instruments:
        try:
            current_position = open_positions[open_positions['currency'] == instrument]['amountK']
        except:
            current_position = 0
            
        #   if current_position.empty:
        #    current_position = 0
        if current_position != 0:
            print(current_position[m])
            current_position = current_position[m]
            m += 1    
            
        current_positions.append(current_position)
        
        CONN.subscribe_market_data(instrument)
        prices = CONN.get_prices(instrument)
        
        current_price = (prices['Bid'] + prices['Ask'])/2
        current_price = current_price[0]
        
        if instrument.find('JPY') > 0:
            current_price /= 100       
        current_prices.append(current_price)
        
    equity = accounts['equity'][0]
       
    target_values = weights * equity
      
    current_positions = np.asarray(current_positions)
    current_prices = np.asarray(current_prices)
    current_prices = current_prices.reshape(11,)
    
    current_margins = lot_size * leverage * current_prices

    target_positions = np.floor(target_values[:-1] / current_margins)
   
    print("-------------target_values---------------")
    print(target_values[:-1])
    print("-------------weights---------------")
    print(type(weights))
    print(weights)
    print("-------------current_prices---------------")
    print(type(current_prices))
    #   print(np.shape(current_prices))
    print(current_prices)
    print("-------------target_positions------------")
    print(target_positions)
    
    #   target_positions = target_values[:-1] / current_prices
    trade_amount = np.floor(target_positions - current_positions)
       
    print("-------------trade_amount------------")
    print(trade_amount)
    
    return target_positions, trade_amount


def trade(instruments, target_positions, trade_amount):

    CONN.close_all()

    m = 0
    for instrument in instruments:
        # find open trades and close
        if target_positions[m] > 0:
            print('Buy1 ' + instrument)
            order = CONN.create_market_buy_order(instrument, np.floor(target_positions[m]))
        '''
        elif target_positions[m] < 0:
            print('Sell1')
            order = CONN.create_market_sell_order(instrument, np.floor(target_positions[m]))       
        '''
        m += 1