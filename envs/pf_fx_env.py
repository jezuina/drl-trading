# -*- coding:utf-8 -*-
from __future__ import absolute_import
import sys
sys.path.append("..")
import gym
import gym.spaces as spaces
import numpy as np
np.seterr(invalid='raise')
import pandas as pd
import datetime
import csv
import os
import uuid
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.signal import detrend
from gym.utils import seeding
from empyrical import max_drawdown, calmar_ratio, omega_ratio, sharpe_ratio, sortino_ratio, downside_risk
from utils.globals import EPS, DATASETS_DIR, OUTPUTS_DIR, CAPITAL_BASE_MULTIPLIER, MAX_WEIGHT, RISK
from utils.enums import compute_indicators, compute_reward, compute_position, lot_size, account_currency
from utils.data import read_h5_fx_history, read_csv_history
from utils.math import my_round, sum_abs, log_negative, log_negative_on_array 
from features.ta import get_indicators_returns


class PortfolioEnv(gym.Env):
    def __init__(self,
                 data_file='/datasets/price_history.h5',
                 output_file='/outputs/portfolio_management',
                 strategy_name='Strategy',
                 total_steps=1500,
                 window_length=7,
                 capital_base=1e6,
                 lot_size=lot_size.Standard,
                 leverage=0.01,
                 commission_percent=0.0,
                 commission_fixed=5,
                 max_slippage_percent=0.05,
                 start_idx=None,
                 compute_indicators=compute_indicators.returns,  #   none  default  all   returns   ptr  rsi
                 compute_reward=compute_reward.profit,   #   profit   sharpe  sortino  max_drawdown   calmar   omega   downside risk
                 compute_position=compute_position.long_only, #   long_only    short_only  long_short  add long/short bias for stocks
                 debug = False
                 ):

        self.datafile = DATASETS_DIR + data_file
        
        if output_file is not None:
            self.output_file = OUTPUTS_DIR + output_file + '_' + (uuid.uuid4().hex)[:16] + '.csv'
        else:
            self.output_file = None
        
        self.strategy_name=strategy_name
        self.total_steps = total_steps
        self.capital_base = capital_base
        self.lot_size = lot_size.value
        self.leverage = leverage
        self.commission_percent = commission_percent / 100
        self.commission_fixed = commission_fixed
        self.max_slippage_percent = max_slippage_percent
        self.start_idx = start_idx
        self.window_length = window_length
        self.compute_indicators = compute_indicators
        self.compute_reward = compute_reward
        self.compute_position = compute_position
        self.debug = debug

        self.instruments, self.price_history, self.tech_history = self._init_market_data()
        self.number_of_instruments = len(self.instruments)
        self.price_data, self.tech_data = self._get_episode_init_state()
        self.current_step = 0
        #   self.current_step = self.window_length - 1
        
        self.current_positions = np.zeros(len(self.instruments))
        self.current_portfolio_values = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))   #   starting with cash only
        self.current_weights = np.concatenate((np.zeros(len(self.instruments)), [1.]))   #   starting with cash only
        #   self.current_date = self.tech_data.major_axis[self.current_step]
        
        self.portfolio_values = []
        self.returns = []
        self.log_returns = []
        self.positions = []
        self.weights = []
        self.trade_dates = []
        self.trade_steps = []
        self.infos = []

        self.done = (self.current_step >= self.total_steps) or (np.sum(self.current_portfolio_values) < CAPITAL_BASE_MULTIPLIER * self.capital_base)
            
        # openai gym attributes
        self.action_space = spaces.Box(-1, 1, shape=(len(self.instruments) + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.instruments), window_length, self.tech_data.shape[-1]), dtype=np.float32)
        
        self.action_sample = self.action_space.sample()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        #   self.current_step = self.window_length - 1
        self.current_positions = np.zeros(len(self.instruments))
        self.current_portfolio_values = np.concatenate((np.zeros(len(self.instruments)), [self.capital_base]))   #   reset to cash only
        self.current_weights = np.concatenate((np.zeros(len(self.instruments)), [1.]))   #   reset to cash only
        self.price_data, self.tech_data = self._get_episode_init_state()
        self.done = (self.current_step >= self.total_steps) or (np.sum(self.current_portfolio_values) < CAPITAL_BASE_MULTIPLIER * self.capital_base)
        
        self.portfolio_values = []
        self.returns = []
        self.log_returns = []
        self.positions = []
        self.weights = []
        self.trade_dates = [] 
        self.trade_steps = []
        self.infos = []
                
        self.positions.append(self.current_positions)      
        self.portfolio_values.append(self.current_portfolio_values)
        self.weights.append(self.current_weights)
        self.trade_steps.append(self.current_step)
        
        return self._get_state()    #   , self.done
    
    def step(self, action):
        np.testing.assert_almost_equal(action.shape, (len(self.instruments) + 1,))

        # normalise just in case
        if self.compute_position == compute_position.long_only:
            weights = np.clip(action, 0, 1)
        elif self.compute_position == compute_position.short_only:
            weights = np.clip(action, -1, 0)
        elif self.compute_position == compute_position.long_and_short:
            weights = np.clip(action, -1, 1)

        '''
        print(action)
        print(weights)
        '''
        
        weights /= (sum_abs(weights) + EPS)
        weights[-1] += np.clip(1 - sum_abs(weights), 0, 1)  # if weights are all zeros we normalise to [0,0..1]
        weights[-1] = sum_abs(weights[-1])
        
        if np.all(np.isnan(weights)):
            weights[:] = 0
            weights[-1] = 1
        
        np.testing.assert_almost_equal(sum_abs(weights), 1.0, 3, 'absolute weights should sum to 1. weights=%s' %weights)
       
        if self.compute_position == compute_position.long_only:
            assert((weights >= 0) * (weights <= 1)).all(), 'all weights values should be between 0 and 1. Not %s' %weights
        elif self.compute_position == compute_position.short_only:
            assert((weights >= -1) * (weights <= 0)).all(), 'all weights values should be between -1 and 0. Not %s' %weights
        elif self.compute_position == compute_position.long_and_short:
            assert((weights >= -1) * (weights <= 1)).all(), 'all weights values should be between -1 and 1. Not %s' %weights
 
        self.weights.append(weights)
    
        self.current_step += 1

        current_prices = self.price_data[:, self.current_step + self.window_length - 1, 0]   #   open price
        slippage = np.random.uniform(0, self.max_slippage_percent/ 100, self.number_of_instruments) 
        #   slippage = self.max_slippage_percent/ 100   #   fixed slippage
        current_prices_with_slippage = np.multiply(current_prices, (1 + slippage)) 
        
        next_prices = self.price_data[:, self.current_step + self.window_length - 1, 3]   #   close price
        slippage = np.random.uniform(0, self.max_slippage_percent/ 100, self.number_of_instruments) 
        #   slippage = self.max_slippage_percent/ 100   #   fixed slippage
        next_prices_with_slippage = np.multiply(next_prices, (1 - slippage)) 

        self._rebalance(weights, current_prices_with_slippage)
        
        if weights[np.where(weights.any() > MAX_WEIGHT)] > 0:   #  or weights[-1] == 1:
            reward = -1
        else:
            reward = self._get_reward(current_prices_with_slippage, next_prices_with_slippage)

        pr, plr = self._get_portfolio_return()
        self.returns.append(pr)
        self.log_returns.append(plr)       
        #   reward = plr     
                   
        info = {
            'current step': self.current_step,
            'current prices': current_prices,
            'next prices': next_prices,  
            'weights': weights,        
            'portfolio value': np.sum(self.current_portfolio_values),
            'portfolio return':   pr,   
            'portfolio log return':   plr,   
            'reward': reward,
        }
        
        self.infos.append(info)
          
        self.done = (self.current_step >= self.total_steps) or (np.sum(self.current_portfolio_values) < CAPITAL_BASE_MULTIPLIER * self.capital_base)

        if self.done and self.output_file is not None:
            # Save infos to file
            keys = self.infos[0].keys()
            with open(self.output_file, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)
        
        if self.debug == True:
            print('current step: {}, portfolio value: {}, portfolio return: {}, portfolio log return: {}, reward: {}'.format(self.current_step, my_round(np.sum(self.current_portfolio_values)), my_round(pr), my_round(plr), my_round(reward)))
            
        return self._get_state(), reward, self.done, info
    
    def render(self, mode='ansi', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1], width=160, depth=2, compact=True)
        elif mode == 'human':
            self.plot()

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info.set_index('current step', inplace=True)
        #   df_info.set_index('date', inplace=True)
        rn = np.asarray(df_info['portfolio return'])
        
        try:
            spf=df_info['portfolio value'].iloc[1]  #   Start portfolio value
            epf=df_info['portfolio value'].iloc[-1] #   End portfolio value
            pr = (epf-spf)/spf
        except :
            pr = 0
            
        try:
            sr = sharpe_ratio(rn)
        except :
            sr = 0
            
        try:
            sor = sortino_ratio(rn)
        except :
            sor = 0
            
        try:
            mdd = max_drawdown(rn)
        except :
            mdd = 0
            
        try:
            cr = calmar_ratio(rn)
        except :
            cr = 0
        
        try:
            om = omega_ratio(rn)
        except :
            om = 0
                        
        try:
            dr = downside_risk(rn)
        except :
            dr = 0

        print("First portfolio value: ", np.round(df_info['portfolio value'].iloc[1]))
        print("Last portfolio value: ", np.round(df_info['portfolio value'].iloc[-1]))
        
        title = self.strategy_name + ': ' + 'profit={: 2.2%} sharpe={: 2.2f} sortino={: 2.2f} max drawdown={: 2.2%} calmar={: 2.2f} omega={: 2.2f} downside risk={: 2.2f}'.format(pr, sr, sor, mdd, cr, om, dr)
        #   df_info[['market value', 'portfolio value']].plot(title=title, fig=plt.gcf(), figsize=(15,10), rot=30)
        df_info[['portfolio value']].plot(title=title, fig=plt.gcf(), figsize=(15,10), rot=30)
               
    def get_meta_state(self):
        return self.tech_data[:, self.current_step, :]

    def get_summary(self):
        portfolio_value_df = pd.DataFrame(np.array(self.portfolio_values), index=np.array(self.trade_steps), columns=self.instruments + ['cash'])
        positions_df = pd.DataFrame(np.array(self.positions), index=np.array(self.trade_steps), columns=self.instruments)
        weights_df = pd.DataFrame(np.array(self.weights), index=np.array(self.trade_steps), columns=self.instruments + ['cash'])
        return portfolio_value_df, positions_df, weights_df


    def _rebalance(self, weights, current_prices):
        target_weights = weights
        target_values = np.sum(self.current_portfolio_values) * target_weights
        
        #   target_positions = (1/self.leverage) * np.floor(target_values[:-1] / current_prices)
        #   target_positions = np.floor(target_values[:-1] / current_prices)
        
        current_margins = self.lot_size * self.leverage * current_prices * RISK
        #   pip_value = self._calculate_pip_value_in_account_currency(account_currency.USD, current_prices)
        #   current_margins = np.multiply(current_margins, pip_value)
        
        target_positions = np.floor(target_values[:-1] / current_margins)   
        
        trade_amount = target_positions - self.current_positions
        
        commission_cost = 0
        commission_cost += np.sum(self.commission_percent * np.abs(trade_amount) * current_prices)
        commission_cost += np.sum(self.commission_fixed * np.abs(trade_amount))
        self.current_weights = target_weights
        self.current_portfolio_values = target_values - commission_cost
        self.current_positions = target_positions
        #   self.current_date = self.preprocessed_market_data.major_axis[self.current_step]
        
        self.positions.append(self.current_positions)
        self.portfolio_values.append(self.current_portfolio_values)
        self.weights.append(self.current_weights) 
        self.trade_steps.append(self.current_step)
        #   self.trade_dates.append(self.current_date)
        
        if self.debug == True:
            print("----------------------------------------------")
            print("current_step: ", self.current_step)
            print("current_prices: ", current_prices)
            print("target_weights: ", my_round(target_weights))
            print("target_values: ", my_round(target_values))
            print("target_positions: ", my_round(target_positions))
            print("trade_amount: ", my_round(trade_amount))
            print("commission_cost: ", my_round(commission_cost))
            print("current_portfolio_values: ", my_round(self.current_portfolio_values))
            print("----------------------------------------------")   
            
    def _get_portfolio_return(self):
        cpv = np.sum(self.portfolio_values[-1])     # current portfolio value
        ppv = np.sum(self.portfolio_values[-2])     # previous portfolio value

        try:
            if ppv > 0:
                pr = (cpv-ppv)/ppv
            else:
                pr = 0
        except:
            pr = 0

        try:
            if pr > 0:
                plr =  np.log(pr) 
            else:
                plr = 0
        except:
            plr = 0 

        return pr, plr
    
    def _get_reward(self, current_prices, next_prices):
        if self.compute_reward == compute_reward.profit:
            returns_rate = next_prices / current_prices
            #   pip_value = self._calculate_pip_value_in_account_currency(account_currency.USD, next_prices)
            #   returns_rate = np.multiply(returns_rate, pip_value)
            log_returns = np.log(returns_rate)
            last_weight = self.current_weights
            securities_value = self.current_portfolio_values[:-1] * returns_rate
            self.current_portfolio_values[:-1] = securities_value
            self.current_weights = self.current_portfolio_values / np.sum(self.current_portfolio_values)
            reward = last_weight[:-1] * log_returns
        elif self.compute_reward == compute_reward.sharpe:
            try:
                sr = sharpe_ratio(np.asarray(self.returns))
            except :
                sr = 0
            reward = sr 
        elif self.compute_reward == compute_reward.sortino:
            try:
                sr = sortino_ratio(np.asarray(self.returns))
            except :
                sr = 0
            reward = sr 
        elif self.compute_reward == compute_reward.max_drawdown:
            try:
                mdd = max_drawdown(np.asarray(self.returns))
            except :
                mdd = 0
            reward = mdd 
        elif self.compute_reward == compute_reward.calmar:
            try:
                cr = calmar_ratio(np.asarray(self.returns))
            except :
                cr = 0
            reward = cr 
        elif self.compute_reward == compute_reward.omega:
            try:
                om = omega_ratio(np.asarray(self.returns))
            except :
                om = 0
            reward = om 
        elif self.compute_reward == compute_reward.downside_risk:
            try:
                dr = downside_risk(np.asarray(self.returns))
            except :
                dr = 0
            reward = dr 
        
        try:
            reward = reward.mean()
        except:
            reward = reward
        
        return reward

    def _get_episode_init_state(self):
        # get data for this episode, each episode might be different.
        if self.start_idx is None:
            self.idx = np.random.randint(low=self.window_length, high=self.tech_history.shape[1] - self.total_steps)
        else:
            self.idx = self.start_idx
        
        assert self.idx >= self.window_length and self.idx <= self.tech_history.shape[1] - self.total_steps, 'Invalid start index'
        
        price_data = self.price_history[:, self.idx - self.window_length:self.idx + self.total_steps + 1, :]
        tech_data = self.tech_history[:, self.idx - self.window_length:self.idx + self.total_steps + 1, :]

        return price_data, tech_data

    def _get_state(self):
        tech_observation = self.tech_data[:, self.current_step:self.current_step + self.window_length, :]
        #   tech_observation = self.tech_data[:, self.current_step-self.window_length:self.current_step, :]
        cash_observation = np.ones((1, self.window_length, tech_observation.shape[2]))
        tech_observation_with_cash = np.concatenate((cash_observation, tech_observation), axis=0)

        price_observation = self.price_data[:, self.current_step:self.current_step + self.window_length, :]
        cash_observation = np.ones((1, self.window_length, price_observation.shape[2]))
        price_observation_with_cash = np.concatenate((cash_observation, price_observation), axis=0)

        return tech_observation

    def _get_normalized_state(self):
        data = self.tech_data.iloc[:, self.current_step + 1 - self.window_length:self.current_step + 1, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + EPS))[:, -1, :]
        return np.concatenate((state, self.current_weights[:-1][:, None]), axis=1)

    #   instruments = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',  'GBP/USD', 'NZD/USD', 'GBP/JPY', 'EUR/JPY', 'AUD/JPY', 'EUR/GBP', 'USD/CHF']
    def _calculate_pip_value_in_account_currency(self, currency, current_prices):        
        pip_values = []
        
        #dictionary to keep prices for each currency, assuming that current_prices has prices in the same order as instruments list has instument names
        prices_for_currency = {}
        
        instrument_index = 0
        for instrument in self.instruments:
            prices_for_currency[instrument] = current_prices[instrument_index]
            instrument_index += 1
            
            
        #account currency is USD
        if currency == account_currency.USD:
            m = 0            
            for instrument in self.instruments:                                               
                first_currency = instrument[0:3]
                second_currency = instrument[4:7]
                                
                #counter currency same as account currency
                if second_currency == 'USD':
                    pip_value = 0.0001
                #base currency same as account currency    
                elif first_currency == 'USD':
                    #counter currency is not JPY
                    if second_currency != 'JPY':
                        pip_value = 0.0001/current_prices[m]
                    #counter currency is JPY
                    else: pip_value = 0.01/current_prices[m] 
                #none of the currency pair is the same as account currency
                #is needed the currency rate for the base currency/account currency
                else:
                    ##base currency/account currency rate is retrieved from stored values in dictionary
                    base_account_rate = prices_for_currency[first_currency+"/USD"]
                    
                    if second_currency == 'JPY':
                        pip_value = base_account_rate * 0.01/current_prices[m]
                    else: pip_value = base_account_rate * 0.0001/current_prices[m] 
                        
                pip_values.append(pip_value)
                m += 1    

        
        return pip_values 

    def _init_market_data(self):
        data, bid, ask, instruments = read_h5_fx_history(filepath=self.datafile, replace_zeros=True)
          
        if self.compute_indicators is compute_indicators.returns:
            new_data = np.zeros((0,0,0),dtype=np.float32)
            for i in range(data.shape[0]):
                security = pd.DataFrame(data[i, :, 0:5]).fillna(method='ffill').fillna(method='bfill')
                security.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                tech_data = np.asarray(get_indicators_returns(security=security.astype(float), open_name='Open', high_name='High', low_name='Low', close_name='Close', volume_name='Volume'))
                new_data = np.resize(new_data, (new_data.shape[0]+1, tech_data.shape[0], tech_data.shape[1]))
                new_data[i] = tech_data    
            price_history = new_data[:,:,:5]
            tech_history = new_data[:,:,5:]
            
        #   print(price_history)    
        #   print(tech_history)    
            
        print('price_history has NaNs: {}, has infs: {}'.format(np.any(np.isnan(price_history)), np.any(np.isinf(price_history))))
        print('price_history shape: {}'.format(np.shape(price_history)))

        print('tech_history has NaNs: {}, has infs: {}'.format(np.any(np.isnan(tech_history)), np.any(np.isinf(tech_history))))
        print('tech_history shape: {}'.format(np.shape(tech_history)))
        
        assert np.sum(np.isnan(price_history)) == 0
        assert np.sum(np.isnan(tech_history)) == 0

        return instruments, price_history, tech_history