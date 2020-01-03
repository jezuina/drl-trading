import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import fxcmpy
import time
from utils.globals import RISK
from utils.enums import account_currency


CONN = fxcmpy.fxcmpy(config_file='config/fxcm.cfg', server='demo')


def download_history(instruments, period, number):

    history = np.empty(shape=(len(instruments), 0, 7), dtype=np.float)
    bid = np.empty(shape=(len(instruments), 0, 7), dtype=np.float)
    ask = np.empty(shape=(len(instruments), 0, 7), dtype=np.float)
    
    m = 0
    
    for instrument in instruments:
        
        data = CONN.get_candles(instrument, period=period, number=number)
        data.index = pd.to_datetime(data.index)
        data['weekday'] = data.index.weekday
        data['hour'] = data.index.hour
        
        #   Price 
        
        Open = pd.DataFrame(data[['bidopen', 'askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh', 'askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow', 'asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose', 'askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
        weekday = pd.DataFrame(data[['weekday']].mean(axis=1), columns=['weekday'])
        hour = pd.DataFrame(data[['hour']].mean(axis=1), columns=['hour'])
        
        df_data = pd.concat([Open, High, Low, Close, Volume, weekday, hour], axis=1)
        #   print(df_data)
        npdata =  df_data.to_numpy()
        
        #   bid 
        
        Open = pd.DataFrame(data[['bidopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['bidhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['bidlow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['bidclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
        weekday = pd.DataFrame(data[['weekday']].mean(axis=1), columns=['weekday'])
        hour = pd.DataFrame(data[['hour']].mean(axis=1), columns=['hour'])
        
        df_bid = pd.concat([Open, High, Low, Close, Volume, weekday, hour], axis=1)

        npbid = df_bid.to_numpy()
        
        #   ask 
        
        Open = pd.DataFrame(data[['askopen']].mean(axis=1), columns=['Open'])
        High = pd.DataFrame(data[['askhigh']].mean(axis=1), columns=['High'])
        Low = pd.DataFrame(data[['asklow']].mean(axis=1), columns=['Low'])
        Close = pd.DataFrame(data[['askclose']].mean(axis=1), columns=['Close'])
        Volume = pd.DataFrame(data[['tickqty']].mean(axis=1), columns=['Volume'])
        weekday = pd.DataFrame(data[['weekday']].mean(axis=1), columns=['weekday'])
        hour = pd.DataFrame(data[['hour']].mean(axis=1), columns=['hour'])
        
        df_ask = pd.concat([Open, High, Low, Close, Volume, weekday, hour], axis=1)

        npask = df_ask.to_numpy()
        
        history = np.resize(history, (history.shape[0], npdata.shape[0], history.shape[2]))
        bid = np.resize(bid, (bid.shape[0], npbid.shape[0], bid.shape[2]))
        ask = np.resize(ask, (ask.shape[0], npask.shape[0], ask.shape[2]))

        for d in range(npdata.shape[0]):   
            #   print(np.shape(history))
            #   history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4],npdata[d,5], npdata[d,6], npdata[d,7], npdata[d,8], npdata[d,9], npdata[d,10], npdata[d,11], npdata[d,12], npdata[d,13]])
            history[m][d] = np.array([npdata[d,0], npdata[d,1], npdata[d,2], npdata[d,3], npdata[d,4], npdata[d,5], npdata[d,6]])
            bid[m][d] = np.array([npbid[d,0], npbid[d,1], npbid[d,2], npbid[d,3], npbid[d,4], npbid[d,5], npbid[d,6]])
            ask[m][d] = np.array([npask[d,0], npask[d,1], npask[d,2], npask[d,3], npask[d,4], npask[d,5], npask[d,6]])
    
        m += 1
    
    return history, bid, ask


def get_nargins(instrument, lot_size, leverage):
    None
    #   Not implemeneted


def get_open_positions_long(instrument, open_positions):
    return [element['amountK'] for element in open_positions if element['currency'] == instrument and element['isBuy']]

def get_open_positions_short(instrument, open_positions):
    return [element['amountK'] for element in open_positions if element['currency'] == instrument and not element['isBuy']]

def get_open_positions_net(instrument, open_positions):
    total_open_positions_long = 0
    total_open_positions_short = 0
    for element in get_open_positions_long(instrument, open_positions):
        total_open_positions_long += element
    for element in get_open_positions_short(instrument, open_positions):
        total_open_positions_short += element
    
    return (total_open_positions_long - total_open_positions_short)
    
def get_offers(instrument, offers):
    return [element['buy'] for element in offers if element['currency'] == instrument]

#   print(offer[['currency','buy','sell', 'spread','pipCost','volume', 'mmr', 'emr']])


def calculate_pip_value(currency, instrument):
    
    first_currency = instrument[0:3]
    second_currency = instrument[3:6]
    
    offers = CONN.get_offers(kind='list')
    
    print(first_currency)
    print(second_currency)
    
    if currency == account_currency.USD:
        if second_currency == 'USD':
            return get_offers(instrument, offers)
        elif first_currency == 'USD':
            if second_currency == 'JPY':
                return 100/get_offers(instrument, offers)
            else:
                return 1/get_offers(instrument, offers)
        else:
            # TODO
            return get_offers(instrument, offers)


def rebalance(instruments, weights, lot_size, leverage):

    current_prices = []
    
    accounts = CONN.get_accounts()
    offers = CONN.get_offers(kind='list')
            
    for instrument in instruments:
        buy = get_offers(instrument, offers)
        '''
        if (instrument[4:6] == 'JPY'):
            buy /= 100
        '''
        current_prices.append(buy)        
            
    current_prices = np.asarray(current_prices)
    current_prices = current_prices.reshape(11,)
    
    print('------------------current_prices')
    print(current_prices)
    
    equity = accounts['equity'][0]
    
    target_values = weights * equity
      
    print('------------------target_values')
    print(target_values)
    
    current_margins = lot_size * leverage * current_prices * RISK
    
    print('------------------current_margins')
    print(RISK)
    print(current_margins)
    
    #   target_positions = np.floor(target_values[:-1] / current_margins)   
    target_positions =  np.floor(target_values[:-1] / current_margins)
    
    return target_positions


def trade(instruments, target_positions):

    current_positions_net = []
    
    open_positions = CONN.get_open_positions(kind='list')
    if len(open_positions)==0:
        current_positions_net = np.zeros(len(instruments))
    else:
        for instrument in instruments:
            current_position_net = get_open_positions_net(instrument, open_positions)
            current_positions_net.append(current_position_net)
        
    current_positions_net = np.asarray(current_positions_net)
    trade_amount = np.round(target_positions - current_positions_net)
    
    #   trade_amount = np.floor(trade_amount*100)/100
    
    print('------------------current_positions_net----------z')
    print(current_positions_net)
    
    print("-------------trade_amount------------")
    print(trade_amount)
    
    #   Trading
    
    #   CONN.close_all()
    
    m = 0
    for instrument in instruments:
        if target_positions[m] == 0:
            CONN.close_all_for_symbol(instrument)
            print('Close all  ' + instrument) 
        elif target_positions[m] > 0  and trade_amount[m] == 0:
            pass
        elif target_positions[m] > 0  and trade_amount[m] > 0:
            order = None
            while order is None:
                order = CONN.create_market_buy_order(instrument, np.abs(trade_amount[m]))
            print('Buy ' + instrument + ' ' + str(np.abs(trade_amount[m])))  
        elif target_positions[m] > 0  and trade_amount[m] < 0:
            CONN.close_all_for_symbol(instrument)
            time.sleep(5)
            order = None
            while order is None:
                order = CONN.create_market_buy_order(instrument, np.abs(target_positions[m]))
            print('Adjust ' + instrument + ' ' + str(np.abs(target_positions[m]))) 
            '''
            order = None
            while order is None:
                order = CONN.create_market_sell_order(instrument, np.abs(trade_amount[m]))
            print('Sell ' + instrument + ' ' + str(target_positions[m])) 
            '''
        m += 1
        time.sleep(5)   # Wait for 5 seconds zulutrade