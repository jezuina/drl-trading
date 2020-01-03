#   tensorboard --logdir=tensorboard   tensorboard --logdir=.    tensorboard --logdir=. --host=0.0.0.0 --port=8080 
#   python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
#   python -m tensorboard.main --logdir=.
import gym
import numpy as np
import itertools as it
import uuid
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy,  ActorCriticPolicy, LstmPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1
from envs.pf_fx_env import PortfolioEnv
from utils.globals import EPS, MODELS_DIR
from utils.enums import compute_indicators, compute_reward, compute_position, lot_size

 
settings =	{
    # env
    #   'data_file':    'fxcm_11_H4_2015_2018_train_with_dates_6300.h5',
    'data_file':    'fxcm_XXXUSD_H4_2015_2019_train_5990.h5',
    'output_file':     None,
    'strategy_name':    'Strategy',
    'total_steps':  3000,
    'window_length': 1,
    'capital_base': 1e4,
    'lot_size': lot_size.Mini,
    'leverage':  0.01,
    'commission_percent': 0.0,
    'commission_fixed': 0.0,
    'max_slippage_percent': 0.1,
    'start_idx':  None,
    'compute_indicators': compute_indicators.returns,  #   
    'compute_reward': compute_reward.profit,   #   profit   sharpe  sortino  max_drawdown   calmar   omega   downside risk
    'compute_position': compute_position.long_and_short, #   long_only    short_only  long_and_short  add long/short bias for stocks
    'debug': False,
    # agnet
    'total_timestamp':  1000000, #
    # vectorized normalized env
    'norm_obs': True, 
    'norm_reward':  True, 
    'clip_obs': 10.,
    'clip_reward':  10.,
    # model
    'model_name':   'fxcm_XXXUSD_H4_2015_2019_train_5990_ppo1'
}


#   hyperparameters 
#   https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
#   https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
#   https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html


def ppo1_train():
    
    # best parames fxcm_11_H4_full_2015_2018_train_6300
         
    v_policy = MlpPolicy  #   policies = [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy]
    v_gamma = 0.99  #  default 0.99
    v_learning_rate = 0.0003   #  default 0.0003
    v_ent_coef = 'auto'   #  default 'auto'
         
    v_env = PortfolioEnv(settings['data_file'],settings['output_file'],settings['strategy_name'],settings['total_steps'],settings['window_length'],settings['capital_base'],settings['lot_size'],settings['leverage'],settings['commission_percent'],settings['commission_fixed'],settings['max_slippage_percent'],settings['start_idx'],settings['compute_indicators'],settings['compute_reward'],settings['compute_position'],settings['debug'])   
    #   Create the vectorized environment
    #   v_env = DummyVecEnv([lambda: v_env])
    #   Normalize environment
    #   v_env = VecNormalize(v_env, norm_obs=settings['norm_obs'], norm_reward=settings['norm_reward'], clip_obs=settings['clip_obs'], clip_reward=settings['clip_reward'], gamma=p_gamma, epsilon=EPS)
 
    #   n_actions = v_env.action_space.shape[-1]    
    #   v_action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    v_action_noise = None

    #   for v_policy, v_gamma, v_lam in it.product(p_policy, p_gamma, p_lam):
    #   print(str(v_policy) + '_' + str(v_gamma) + '_' + str(v_lam))
        
    model_name = settings['model_name'] + '_' + str(settings['total_timestamp']) + '_' + str(settings['window_length']) + '_'  +  str(settings['compute_indicators']) + '_' + str(v_gamma) + '_' +  (uuid.uuid4().hex)[:16]
    
    model = PPO1(env=v_env, policy=v_policy, gamma=v_gamma, verbose=0, tensorboard_log='log_' + model_name)
    model.learn(total_timesteps=(settings['total_timestamp']))
    model.save(MODELS_DIR + model_name)
    #   v_env.save_running_average(MODELS_DIR)

    del model


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    ppo1_train()