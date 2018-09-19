import numpy as np
import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperEnv(MujocoEnv, Serializable):

    FILE = 'hopper.xml'

    @autoargs.arg('alive_coeff', type=float,
                  help='reward coefficient for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            alive_coeff=1,
            ctrl_cost_coeff=0.01,
            *args, **kwargs):
        self.alive_coeff = alive_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(HopperEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    # Consist of 11 dimensions
    # 0: z-com
    # 1: forward pitch along y-axis
    # 5: x-comvel
    # 6: z-comvel
    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.get_body_com("torso")[2].flat,
            self.model.data.qpos[2:].flat,
            self.get_body_comvel("torso")[[0, 2]].flat,
            self.model.data.qvel[2:].flat
        ])

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = next_obs[5]
        height = next_obs[0]
        ang = next_obs[1]
        reward = vel - \
            0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling)) -\
            np.sum(np.maximum(np.abs(next_obs[2:]) - 100, 0)) -\
            10*np.maximum(0.45 - height, 0) - \
            10*np.maximum(abs(ang) - .2, 0)
        # notdone = np.isfinite(state).all() and \
        #     (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
        #     (abs(state[2]) < .2)
        # done = not notdone
        done = False
        return Step(next_obs, reward, done)

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

    def cost_tf(self, x, u, x_next):
        vel = x_next[:, 5]
        height = x_next[:, 0]
        ang = x_next[:, 1]
        return -tf.reduce_mean(vel -
                               self.ctrl_cost_coeff * 0.5 * tf.reduce_sum(tf.square(u), axis=1) -
                               10*tf.maximum(0.45 - height, 0) -
                               10*tf.maximum(tf.abs(ang) - .2, 0) -
                               tf.reduce_sum(tf.maximum(tf.abs(x_next[:, 2:]) - 100, 0), axis=1)
                               )

    def cost_np_vec(self, x, u, x_next):
        vel = x_next[:, 5]
        height = x_next[:, 0]
        ang = x_next[:, 1]
        assert np.amax(np.abs(u)) <= 1.0
        return -(vel -
                 self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1) -
                 10 * np.maximum(0.45 - height, 0) -
                 10 * np.maximum(np.abs(ang) - .2, 0) -
                 np.sum(np.maximum(np.abs(x_next[:, 2:]) - 100, 0), axis=1)
                 )

    def cost_np(self, x, u, x_next):
        return np.mean(self.cost_np_vec(x, u, x_next))