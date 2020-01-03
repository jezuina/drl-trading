import enum 
  
class compute_indicators(enum.Enum): 
    returns = 1
        
class compute_reward(enum.Enum): 
    profit = 1
    sharpe = 2
    sortino = 3
    max_drawdown = 4
    calmar = 5
    omega = 6
    downside_risk = 7

class compute_position(enum.Enum): 
    long_only = 1
    short_only = 2
    long_and_short = 3
    
class account_currency(enum.Enum):
    USD = 1
    EUR = 2
    GBP = 3      
      
class lot_size(enum.Enum):
    Standard = 100000
    Mini = 10000
    Micro = 1000
    Nano = 100