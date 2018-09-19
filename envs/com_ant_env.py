from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math
import tensorflow as tf

class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.get_body_com("torso").flat,
            self.model.data.qpos.flat[3:],
            self.get_body_comvel("torso").flat,
            self.model.data.qvel.flat[3:],
            # np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            # self.get_body_xmat("torso").flat,
            # self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        com = self.get_body_com("torso")
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and com[2] >= 0.2 and com[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    def cost_tf(self, x, u, x_next, dones):
        vel = x_next[:, 15]
        return -tf.reduce_mean((vel -
                               1e-2 * 0.5 * tf.reduce_sum(tf.square(u), axis=1) +
                               0.05) * (1-dones)
                               )

    def cost_np_vec(self, x, u, x_next):
        vel = x_next[:, 15]
        assert np.amax(np.abs(u)) <= 1.0
        return -(vel -
                 1e-2 * 0.5 * np.sum(np.square(u), axis=1) +
                 0.05
                 )

    def cost_np(self, x, u, x_next):
        return np.mean(self.cost_np_vec(x, u, x_next))

    def is_done(self, x, x_next):
        '''
        :param x: vector of obs
        :param x_next: vector of next obs
        :return: boolean array
        '''
        notdone = np.logical_and(
            np.logical_and(
                x_next[:, 2] >= 0.2,
                x_next[:, 2] <= 1.0
            ),
            np.amin(np.isfinite(x_next), axis=1)
        )
        return np.invert(notdone)

    def is_done_tf(self, x, x_next):
        '''
        :param x:
        :param x_next:
        :return: float array 1.0 = True, 0.0 = False
        '''
        notdone = tf.logical_and(
            tf.logical_and(
                x_next[:, 2] >= 0.2,
                x_next[:, 2] <= 1.0
            ),
            tf.reduce_all(tf.is_finite(x_next), axis=1)
        )
        return tf.cast(tf.logical_not(notdone), tf.float32)