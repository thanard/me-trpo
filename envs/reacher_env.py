import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import tensorflow as tf

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    # def _step(self, a):
    #     # x = self._get_obs()[None]
    #     # assert np.allclose(self.get_body_com("fingertip")[:2], get_fingertips(x)),\
    #     # str(self.get_body_com("fingertip")) + " "+ str(get_fingertips(x))
    #     vec = self.get_body_com("fingertip")-self.get_body_com("target")
    #     reward_dist = - np.linalg.norm(vec[:2])
    #     reward_ctrl = - np.square(a).sum()*0.01
    #     reward = reward_dist + reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _step(self, a):
        a = np.reshape(np.clip(a, -1, 1), -1)
        obs = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        obs_next = self._get_obs()
        reward = - self.cost_np(obs[None], a[None], obs_next[None])
        done = False
        return obs_next, reward, done, dict(ctrl_cost=np.sum(np.square(a)),
                                            comment='this_is_local_step')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < .2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, obs=None):
        if obs is not None:
            self.set_state(np.concatenate([obs[:2], obs[4:6]]),
                           np.concatenate([obs[2:4], np.zeros(2)]))
            return self._get_obs()
        else:
            return self._reset()

    # def _get_obs(self):
    #     theta = self.model.data.qpos.flat[:2]
    #     return np.concatenate([
    #         np.cos(theta),
    #         np.sin(theta),
    #         self.model.data.qpos.flat[2:],
    #         self.model.data.qvel.flat[:2],
    #         self.get_body_com("fingertip") - self.get_body_com("target")
    #     ])
    # def _get_obs(self):
    #     vec = self.get_body_com("fingertip") - self.get_body_com("target")
    #     return np.concatenate([self.model.data.qpos.flat[:2],
    #                            self.model.data.qvel.flat[:2],
    #                            vec[:2]])

    def _get_obs(self):
        '''
        state representations: theta1, theta2, theta1_dot, theta2_dot, x_goal, y_goal
        :return:
        '''
        assert np.allclose(self.model.data.qpos.flat[2:], self.get_body_com("target")[:2])
        return np.concatenate([self.model.data.qpos.flat[:2],
                               self.model.data.qvel.flat[:2],
                               self.get_body_com("target")[:2]])
    @property
    def n_goals(self):
        '''
        :return: goal dimensions
        '''
        return 2
    @property
    def n_states(self):
        '''
        :return: state dimensions
        '''
        return 4

    def cost_np(self, x, u, x_next, ctrl_cost_coeff=2):
        assert np.amax(np.abs(u)) <= 1.0
        return np.mean(np.linalg.norm(x[:, -2:]-get_fingertips(x), axis=1) +\
                       ctrl_cost_coeff*0.5*np.sum(np.square(u), axis=1))

    def cost_tf(self, x, u, x_next, ctrl_cost_coeff=2):
        return tf.reduce_mean(tf.norm(x[:, -2:]-get_fingertips_tf(x), axis=1) +\
                       ctrl_cost_coeff*0.5*tf.reduce_sum(tf.square(u), axis=1))

    def cost_np_vec(self, x, u, x_next, ctrl_cost_coeff=2):
        assert np.amax(np.abs(u)) <= 1.0
        return (np.linalg.norm(x[:, -2:]-get_fingertips(x), axis=1) +\
                       ctrl_cost_coeff*0.5*np.sum(np.square(u), axis=1))

def get_fingertips(x):
    x_cord = np.reshape(0.1 * np.cos(x[:, 0]) + 0.11 * np.cos(x[:, 0] + x[:, 1]), (-1, 1))
    y_cord = np.reshape(0.1 * np.sin(x[:, 0]) + 0.11 * np.sin(x[:, 0] + x[:, 1]), (-1, 1))
    return np.concatenate([x_cord, y_cord], axis=1)

def get_fingertips_tf(x):
    x_cord = tf.reshape(0.1 * tf.cos(x[:, 0]) + 0.11 * tf.cos(x[:, 0] + x[:, 1]), (-1, 1))
    y_cord = tf.reshape(0.1 * tf.sin(x[:, 0]) + 0.11 * tf.sin(x[:, 0] + x[:, 1]), (-1, 1))
    return tf.concat([x_cord, y_cord], axis=1)

'''
Call this after loading reacher gym version.
'''
def gym_to_local():
    import gym
    from sandbox.rocky.tf.spaces.box import Box
    import envs.base as base
    gym.envs.mujoco.reacher.ReacherEnv._get_obs = ReacherEnv._get_obs
    gym.envs.mujoco.reacher.ReacherEnv._step = ReacherEnv._step
    gym.envs.mujoco.reacher.ReacherEnv.observation_space = property(lambda self: Box(
        low=ReacherEnv().observation_space.low,
        high=ReacherEnv().observation_space.high
    ))
    gym.envs.mujoco.reacher.ReacherEnv.reset = ReacherEnv.reset
    gym.envs.mujoco.reacher.ReacherEnv.reset_model = ReacherEnv.reset_model
    gym.envs.mujoco.reacher.ReacherEnv.n_goals = ReacherEnv.n_goals
    gym.envs.mujoco.reacher.ReacherEnv.n_states = ReacherEnv.n_states
    gym.envs.mujoco.reacher.ReacherEnv.cost_np = ReacherEnv.cost_np
    gym.envs.mujoco.reacher.ReacherEnv.cost_tf = ReacherEnv.cost_tf
    gym.envs.mujoco.reacher.ReacherEnv.cost_np_vec = ReacherEnv.cost_np_vec
    base.TfEnv.observation_space = property(lambda self: Box(
        low=ReacherEnv().observation_space.low,
        high=ReacherEnv().observation_space.high
    ))