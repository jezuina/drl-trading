import sys
sys.path.append("..")
import talib._ta_lib as talib
import numpy as np
from utils.math import my_log


def get_indicators_returns(security, open_name, high_name, low_name, close_name, volume_name):
      
    #   LOG_RR    
    #   security['LOG_RR_0'] = my_log(security[close_name].shift(0).fillna(1) / security[open_name].shift(0).fillna(1))  
    security['LOG_RR_1'] = my_log(security[close_name].shift(1).fillna(1) / security[open_name].shift(1).fillna(1))
    security['LOG_RR_2'] = my_log(security[close_name].shift(2).fillna(1) / security[open_name].shift(2).fillna(1))
    security['LOG_RR_3'] = my_log(security[close_name].shift(3).fillna(1) / security[open_name].shift(3).fillna(1))  
    security['LOG_RR_4'] = my_log(security[close_name].shift(4).fillna(1) / security[open_name].shift(4).fillna(1))
    security['LOG_RR_5'] = my_log(security[close_name].shift(5).fillna(1) / security[open_name].shift(5).fillna(1))
    security['LOG_RR_6'] = my_log(security[close_name].shift(6).fillna(1) / security[open_name].shift(6).fillna(1))
    security['LOG_RR_7'] = my_log(security[close_name].shift(7).fillna(1) / security[open_name].shift(7).fillna(1))
    security['LOG_RR_8'] = my_log(security[close_name].shift(8).fillna(1) / security[open_name].shift(8).fillna(1))
    security['LOG_RR_9'] = my_log(security[close_name].shift(9).fillna(1) / security[open_name].shift(9).fillna(1))
    security['LOG_RR_10'] = my_log(security[close_name].shift(10).fillna(1) / security[open_name].shift(10).fillna(1))   
    
    security = security.dropna().astype(np.float32)
        
    return security