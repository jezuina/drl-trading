from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import SAC
from envs.pf_fx_env import PortfolioEnv
from fx.fxcm import download_history, rebalance, trade
from utils.globals import MODELS_DIR, DATASETS_DIR
from utils.data import write_fx_to_h5py
from utils.enums import compute_indicators, compute_reward, compute_position, lot_size
from utils.common import get_date_and_time


settings =	{
    # env
    'data_file':    'fxcm_XXXUSD_H4.h5',
    'output_file':     None,
    'strategy_name':    'Strategy',
    'total_steps':  1,
    'window_length': 1,
    'capital_base': 1e6,
    'lot_size': lot_size.Mini,
    'leverage':  100,
    'commission_percent': 0.0,
    'commission_fixed': 0.09,
    'max_slippage_percent': 0.0,
    'start_idx':  100,
    'compute_indicators':  compute_indicators.returns,  #   none  default  all   returns   ptr    atr    rsi
    'compute_reward':  compute_reward.profit,   #   profit   sharpe  sortino  max_drawdown   calmar   omega   downside risk
    'compute_position':  compute_position.long_and_short,  #   long_only    short_only  long_short  add long/short bias for stocks
    'debug':    False,
    # agnet
    'total_timestamp':  200000, #
    # vectorized normalized env
    'norm_obs': True, 
    'norm_reward':  True, 
    'clip_obs': 10.,
    'clip_reward':  10.,
    # model
    'model_name':   'fxcm_XXXUSD_H4_2015_2018_train_sac_100000_1_compute_indicators.returns_compute_position.long_only_0.992679041900889_d89448c0b42f4622',
}

   
def create_fxcm_dataset(data_file, instruments, period, number):
    
    history, bid, ask = download_history(instruments, period, number)
    
    write_fx_to_h5py(history, bid, ask, instruments, data_file, False)
    
    return history, bid, ask


def get_new_weights():

    v_env = PortfolioEnv(settings['data_file'],settings['output_file'],settings['strategy_name'],settings['total_steps'],settings['window_length'],settings['capital_base'],settings['lot_size'],settings['leverage'],settings['commission_percent'],settings['commission_fixed'],settings['max_slippage_percent'],settings['start_idx'],settings['compute_indicators'],settings['compute_reward'],settings['compute_position'],settings['debug'])   
    #   Create the vectorized environment
    #   v_env = DummyVecEnv([lambda: v_env])
    #   Normalize environment
    #   v_env = VecNormalize(v_env, norm_obs=settings['norm_obs'], norm_reward=settings['norm_reward'], clip_obs=settings['clip_obs'], clip_reward=settings['clip_reward'], gamma=p_gamma, epsilon=EPS)
  
    model = SAC.load(MODELS_DIR + settings['model_name']) 
     
    # Strategy
    
    obs = v_env.reset()
    dones = False
       
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = v_env.step(action)
        #   v_env.render(mode='ansi')
    
    weights = v_env.current_weights      
    
    return weights

            
def main():

    print("-------------date and time-------------")
    print(get_date_and_time())	
    
    data_file = DATASETS_DIR + settings['data_file']
    instruments = ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD']
    period = 'H4'   #   'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'D1', 'W1', 'M1'
    number = 400
    
    data, bid, ask = create_fxcm_dataset(data_file, instruments, period, number)

    #   print("-------------data---------------")
    #   print(data)

    weights = get_new_weights()
   
    print("---------------weights----------------")
    print(weights)
    
    target_positions = rebalance(instruments, weights, settings['lot_size'].value, settings['leverage'])      

    print("-------------instruments--------------")
    print(instruments)

    print("----------target_positions------------")
    print(target_positions)

    trade(instruments, target_positions)


# Trade trading system defined in current file.
if __name__ == '__main__':
    main()
    '''
    while True:
        now = dt.datetime.now()
        if (np.mod(now.hour,4) == 0 and now.minute == 0 and now.second == 0):
            main()
    '''