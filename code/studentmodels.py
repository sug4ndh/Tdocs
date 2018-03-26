from __future__ import division

import pickle
import os
import sys
import copy
import random
import types
from queue import Queue

import numpy as np
import gym
from gym import spaces
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.overrides import overrides


def sample_const_delay(d):
    return (lambda: d)


class StudentEnv(gym.Env):

    def __init__(self, n_items=10, n_steps=100, discount=1., sample_delay=None, reward_func='likelihood'):
        if sample_delay is None:
            self.sample_delay = sample_const_delay(1)
        else:
            self.sample_delay = sample_delay
        self.curr_step = None
        self.n_steps = n_steps
        self.n_items = n_items
        self.now = 0
        self.curr_item = 0
        self.curr_outcome = None
        self.curr_delay = None
        self.discount = discount
        self.reward_func = reward_func

        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Box(np.zeros(4), np.array([n_items - 1, 1, sys.maxsize, sys.maxsize]))

    def _recall_likelihoods(self):
        raise NotImplementedError

    def _recall_log_likelihoods(self, eps=1e-9):
        return np.log(eps + self._recall_likelihoods())

    def _update_model(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def _obs(self):
        timestamp = self.now - self.curr_delay
        return np.array([self.curr_item, self.curr_outcome, timestamp, self.curr_delay], dtype=int)

    def _rew(self):
        #rewards#
        if self.reward_func == 'likelihood':
            return self._recall_likelihoods().mean()
        elif self.reward_func == 'log_likelihood':
            return self._recall_log_likelihoods().mean()
        elif self.reward_func == 'average_sample_outcome_all':
            return (np.random.rand() < self._recall_likelihoods()).sum()
        elif self.reward_func == 'one_sample_outcome':
            return np.random.choice((np.random.rand() < self._recall_likelihoods()))
        else:
            raise ValueError

    def _step(self, action):
        if self.curr_step is None or self.curr_step >= self.n_steps:
            raise ValueError

        if action < 0 or action >= self.n_items:
            raise ValueError

        self.curr_item = action
        self.curr_outcome = 1 if np.random.random() < self._recall_likelihoods()[action] else 0

        self.curr_step += 1
        self.curr_delay = self.sample_delay()
        self.now += self.curr_delay

        self._update_model(self.curr_item, self.curr_outcome, self.now, self.curr_delay)

        obs = self._obs()
        r = self._rew()
        done = self.curr_step == self.n_steps
        info = {}

        return obs, r, done, info

    def _reset(self):
        self.curr_step = 0
        self.now = 0
        return self._step(np.random.choice(range(self.n_items)))[0]


item_difficulty_mean = 1
item_difficulty_std = 1

log_item_decay_exp_mean = 1
log_item_decay_exp_std = 1

log_delay_coef_mean = 0
log_delay_coef_std = 0.01


def sample_item_difficulties(n_items):
    return np.random.normal(item_difficulty_mean, item_difficulty_std, n_items)


def sample_student_ability():
    return 0


def sample_window_cw(n_windows):
    x = 1 / (np.arange(1, n_windows + 1, 1)) ** 2
    return x[::-1]


def sample_window_nw(n_windows):
    x = 1 / (np.arange(1, n_windows + 1, 1)) ** 2
    return x[::-1]


def sample_item_decay_exps(n_items):
    return np.exp(np.random.normal(log_item_decay_exp_mean, log_item_decay_exp_std, n_items))


def sample_student_decay_exp():
    return 0


def sample_delay_coef():
    return np.exp(np.random.normal(log_delay_coef_mean, log_delay_coef_std))


class DASHEnv(StudentEnv):

    def __init__(
            self, n_windows=5, item_difficulties=None, student_ability=None,
            window_cw=None, window_nw=None, item_decay_exps=None, student_decay_exp=None,
            delay_coef=None, **kwargs):
        super(DASHEnv, self).__init__(**kwargs)

        if item_difficulties is None:
            self.item_difficulties = sample_item_difficulties(self.n_items)
        else:
            if len(item_difficulties) != self.n_items:
                raise ValueError
            self.item_difficulties = item_difficulties

        if student_ability is None:
            self.student_ability = sample_student_ability()
        else:
            self.student_ability = student_ability

        if item_decay_exps is None:
            self.item_decay_exps = sample_item_decay_exps(self.n_items)
        else:
            if len(item_decay_exps) != self.n_items:
                raise ValueError
            self.item_decay_exps = item_decay_exps

        if student_decay_exp is None:
            self.student_decay_exp = sample_student_decay_exp()
        else:
            self.student_decay_exp = student_decay_exp

        if delay_coef is None:
            self.delay_coef = sample_delay_coef()
        else:
            self.delay_coef = delay_coef

        if self.n_steps % n_windows != 0:
            raise ValueError
        self.n_windows = n_windows
        self.window_size = self.n_steps // self.n_windows
        self.n_correct = None
        self.n_attempts = None

        if window_cw is None:
            window_cw = sample_window_cw(self.n_windows)
        if window_nw is None:
            window_nw = sample_window_nw(self.n_windows)
        if len(window_cw) != n_windows or len(window_nw) != n_windows:
            raise ValueError

        self.window_cw = np.tile(window_cw, self.n_items).reshape((self.n_items, self.n_windows))
        self.window_nw = np.tile(window_nw, self.n_items).reshape((self.n_items, self.n_windows))

        self.init_tlasts = np.exp(np.random.normal(0, 0.01, self.n_items))
        self._init_params()

    def _init_params(self):
        self.n_correct = np.zeros((self.n_items, self.n_windows))
        self.n_attempts = np.zeros((self.n_items, self.n_windows))
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)

    def _current_window(self):
        return min(self.n_windows - 1, self.curr_step // self.window_size)

    def _recall_likelihoods(self):
        curr_window = self._current_window()
        study_histories = (self.window_cw[:, :curr_window] * np.log(
            1 + self.n_correct[:, :curr_window]) + self.window_nw[:, :curr_window] * np.log(
            1 + self.n_attempts[:, :curr_window])).sum(axis=1)
        m = 1 / (1 + np.exp(-(self.student_ability - self.item_difficulties + study_histories)))
        f = np.exp(self.student_decay_exp - self.item_decay_exps)
        delays = self.now - self.tlasts
        return m / (1 + self.delay_coef * delays) ** f

    def _update_model(self, item, outcome, timestamp, delay):
        curr_window = self._current_window()
        if outcome == 1:
            self.n_correct[item, curr_window] += 1
        self.n_attempts[item, curr_window] += 1
        self.tlasts[item] = timestamp

    def _reset(self):
        self._init_params()
        return super(DASHEnv, self)._reset()


def sample_item_decay_rates(n_items):
    return np.exp(np.random.normal(np.log(0.077), 1, n_items))


class EFCEnv(StudentEnv):
    '''exponential forgetting curve'''

    def __init__(self, item_decay_rates=None, **kwargs):
        super(EFCEnv, self).__init__(**kwargs)

        if item_decay_rates is None:
            self.item_decay_rates = sample_item_decay_rates(self.n_items)
        else:
            self.item_decay_rates = item_decay_rates

        self.tlasts = None
        self.strengths = None
        self.init_tlasts = np.exp(np.random.normal(0, 1, self.n_items))
        self._init_params()

    def _init_params(self):
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)
        self.strengths = np.ones(self.n_items)

    def _recall_likelihoods(self):
        return np.exp(-self.item_decay_rates * (self.now - self.tlasts) / self.strengths)

    def _update_model(self, item, outcome, timestamp, delay):
        # self.strengths[item] = max(1, self.strengths[item] + 2 * outcome - 1) # fictional Leitner system
        self.strengths[item] += 1  # num attempts
        self.tlasts[item] = timestamp

    def _reset(self):
        self._init_params()
        return super(EFCEnv, self)._reset()


def sample_loglinear_coeffs(n_items):
    coeffs = np.array([1, 1, 0])
    coeffs = np.concatenate((coeffs, np.random.normal(0, 1, n_items)))
    return coeffs


class HLREnv(StudentEnv):
    '''exponential forgetting curve with log-linear memory strength'''

    def __init__(self, loglinear_coeffs=None, **kwargs):
        super(HLREnv, self).__init__(**kwargs)

        if loglinear_coeffs is None:
            self.loglinear_coeffs = sample_loglinear_coeffs(self.n_items)
        else:
            self.loglinear_coeffs = loglinear_coeffs
        assert self.loglinear_coeffs.size == 3 + self.n_items

        self.tlasts = None
        self.loglinear_feats = None
        self.init_tlasts = np.exp(np.random.normal(0, 1, self.n_items))
        self._init_params()

    def _init_params(self):
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)
        self.loglinear_feats = np.zeros((self.n_items, 3))  # n_attempts, n_correct, n_incorrect
        self.loglinear_feats = np.concatenate((self.loglinear_feats, np.eye(self.n_items)), axis=1)

    def _strengths(self):
        return np.exp(np.einsum('j,ij->i', self.loglinear_coeffs, self.loglinear_feats))

    def _recall_likelihoods(self):
        return np.exp(-(self.now - self.tlasts) / self._strengths())

    def _update_model(self, item, outcome, timestamp, delay):
        self.loglinear_feats[item, 0] += 1
        self.loglinear_feats[item, 1 if outcome == 1 else 2] += 1
        self.tlasts[item] = timestamp

    def _reset(self):
        self._init_params()
        return super(HLREnv, self)._reset()
