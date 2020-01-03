#   tensorboard --logdir=tensorboard   tensorboard --logdir=.
import numpy as np
import matplotlib.pyplot as plt
import pyfolio as pf
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1
from envs.pf_fx_env import PortfolioEnv
from utils.globals import EPS, MODELS_DIR
from utils.enums import compute_indicators, compute_reward, compute_position, lot_size


settings =	{
    # env
    'data_file':    'fxcm_XXXUSD_H4_2015_2019_test_1983.h5',
    #   'data_file':    'fxcm_EURUSD_H1_2019_2019_test_6232.h5',
    'output_file':     None,
    'strategy_name':    'Strategy',
    'total_steps':  1900,
    'window_length': 1,
    'capital_base': 1e4,
    'lot_size': lot_size.Mini,
    'leverage':  0.01,
    'commission_percent': 0.0,
    'commission_fixed': 0.0,
    'max_slippage_percent': 0.1,
    'start_idx':  1,
    'compute_indicators': compute_indicators.returns,  #   
    'compute_reward': compute_reward.profit,   #   profit   sharpe  sortino  max_drawdown   calmar   omega   downside risk
    'compute_position': compute_position.long_and_short, #   long_only    short_only  long_and_short  add long/short bias for stocks
    'add_noise':    False,
    'debug':    False,
    'model_name':   'fxcm_XXXUSD_H4_2015_2019_train_5990_ppo1_1000000_1_compute_indicators.returns_0.99_d955a70fac7449ec',
    #   'model_name':   'fxcm_EURUSD_H1_2018_2019_train_10558_ppo1_1000000_1_compute_indicators.returns_0.99_b00a387ebda14a1a',
    # agnet
    'norm_obs': True, 
    'norm_reward':  True, 
    'clip_obs': 10,
    'clip_reward':  10
}


p_gamma = 0.9


def ppo1_test():

    v_env = PortfolioEnv(settings['data_file'],settings['output_file'],settings['strategy_name'],settings['total_steps'],settings['window_length'],settings['capital_base'],settings['lot_size'],settings['leverage'],settings['commission_percent'],settings['commission_fixed'],settings['max_slippage_percent'],settings['start_idx'],settings['compute_indicators'],settings['compute_reward'],settings['compute_position'],settings['debug'])   
    #   Create the vectorized environment
    #   v_env = DummyVecEnv([lambda: v_env])
    #   Normalize environment
    #   v_env = VecNormalize(v_env, norm_obs=settings['norm_obs'], norm_reward=settings['norm_reward'], clip_obs=settings['clip_obs'], clip_reward=settings['clip_reward'], gamma=p_gamma, epsilon=EPS)
  
    model = PPO1.load(MODELS_DIR + settings['model_name']) 
     
    # Strategy
    
    obs = v_env.reset()
    dones = False
       
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = v_env.step(action)
        #   v_env.render(mode='ansi')
     
    v_env.strategy_name = 'fxcm_XXXUSD_H4 returns PPO1'       
    v_env.render(mode='human')     
    
    #   pv,pp,pw=v_env.get_summary()
    #   pr=pv.sum(axis=1).pct_change().fillna(0)
    
    pr = v_env.returns
    
    # Random agent
    
    obs = v_env.reset()
    dones = False
       
    while not dones:
        # action, _states = model.predict(obs, deterministic=True)
        action = v_env.action_sample
        obs, rewards, dones, info = v_env.step(action)
        #   v_env.render(mode='ansi')
     
    v_env.strategy_name = 'Random agent'       
    v_env.render(mode='human')     
    
    # Buy and hold
    
    obs = v_env.reset()
    dones = False
    
    weights=np.concatenate((np.ones(len(v_env.instruments))/len(v_env.instruments),[0]))
        
    #   print(weights)
    
    while not dones:
        obs, rewards, dones, info = v_env.step(action=weights)
        weights=v_env.current_weights
        v_env.render(mode='ansi')
    
    v_env.strategy_name = 'Buy and hold'        
    v_env.render(mode='human')    
    
    bpv,bpp,bpw=v_env.get_summary()
    bpr=bpv.sum(axis=1).pct_change().fillna(0)
    
    bpr = v_env.returns
    
    '''
    #   Extended
    pv,pp,pw=v_env.get_summary()
    pv.sum(axis=1).plot()
    plt.title('strategy')
    plt.show()

    
    bpv,bpp,bpw=v_env.get_summary()
    bpv.sum(axis=1).plot()
    plt.title('buy and hold')
    plt.show()

    pr=pv.sum(axis=1).pct_change().fillna(0)
    bpr=bpv.sum(axis=1).pct_change().fillna(0)
    '''
    #   pf.create_simple_tear_sheet(returns=pr,benchmark_rets=bpr)
    pf.create_full_tear_sheet(returns=pr,benchmark_rets=bpr)
         
    
# Evaluate trading system defined in current file.
if __name__ == '__main__':
    ppo1_test()