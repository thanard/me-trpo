from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import tensorflow as tf

def get_xy_coordinate(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def get_dxy_by_dt(theta,theta_dot):
    return np.array([-np.sin(theta)*theta_dot, np.cos(theta)*theta_dot])

def get_original_representation(states):
    '''
    The first two are com and the rest are angles.
    '''
    if states is None:
        return None
    assert len(states) == 10
    out = np.array(states)
    out[:2] -= 2/3*get_xy_coordinate(states[2])
    out[:2] -= 1/2*get_xy_coordinate(np.pi + states[2] + states[3])
    out[:2] -= 1/6*get_xy_coordinate(np.pi + states[2] + states[3] + states[4])
    out[5:7] -= 2/3*get_dxy_by_dt(states[2], states[7])
    out[5:7] -= 1/2*get_dxy_by_dt(np.pi + states[2] + states[3],
                                  states[7] + states[8])
    out[5:7] -= 1/6*get_dxy_by_dt(np.pi + states[2] + states[3] + states[4],
                                  states[7] + states[8] + states[9])
    return out

class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_original_states(self):
        qpos = np.squeeze(self.model.data.qpos)
        qvel = np.squeeze(self.model.data.qvel)
        return np.concatenate([qpos, qvel])
    def get_current_obs(self):
        qpos = np.squeeze(self.model.data.qpos)
        qvel = np.squeeze(self.model.data.qvel)
        return np.concatenate([
            self.get_body_com("torso")[:2],
            qpos[2:5],
            self.get_body_comvel("torso")[:2],
            qvel[2:5]
        ]).reshape(-1)

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        # ctrl_cost = - 0.5 * self.ctrl_cost_coeff * np.sum(
        # np.log(1+1e-8 -(action/scaling)**2))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        # Modified reward here
        return Step(next_obs, reward, done)

    def reset(self, init_state=None):
        self.reset_mujoco(get_original_representation(init_state))
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    @overrides
    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)

    def cost_np(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        return -np.mean(x_next[:, 5] - self.ctrl_cost_coeff * np.mean(np.square(u), axis=1))

    def cost_tf(self, x, u, x_next):
        return -tf.reduce_mean(x_next[:, 5] - self.ctrl_cost_coeff * tf.reduce_mean(tf.square(u), axis=1))

    def cost_np_vec(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        return -(x_next[:, 5] - self.ctrl_cost_coeff * np.mean(np.square(u), axis=1))