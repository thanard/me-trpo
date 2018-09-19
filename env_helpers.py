import numpy as np
import tensorflow as tf
import os.path
import rllab.misc.logger as rllab_logger
from rllab.envs.normalized_env import normalize
from envs import  *
from sandbox.rocky.tf.envs.base import TfEnv

####################
#### Environment ###
####################

def get_env(env_name):
    if env_name == 'snake':
        return TfEnv(normalize(SnakeEnv()))
    elif env_name == 'swimmer':
        return TfEnv(normalize(SwimmerEnv()))
    elif env_name == 'half_cheetah':
        return TfEnv(normalize(HalfCheetahEnv()))
    elif env_name == 'hopper':
        return TfEnv(normalize(HopperEnv()))
    elif env_name == 'ant':
        return TfEnv(normalize(AntEnv()))
    # elif env_name == 'humanoidstandup':
    #     return TfEnv(GymEnv('HumanoidStandup-v1',
    #                         record_video=False,
    #                         record_log=False))
    elif env_name == 'humanoid':
        return TfEnv(normalize(HumanoidEnv()))
    # elif env_name == 'simple_humanoid':
    #     return TfEnv(normalize(SimpleHumanoidEnv()))
    else:
        assert False, "Define the env from env_name."

policy_scope = 'training_policy'
clip_action = ''
def get_action(observation,
               policy_in,
               policy_out,
               sess,
               action_noise,
               **kwargs):
    # TODO think about what to do in first iteration when diff weights is None.
    action = sess.run(policy_out, feed_dict={policy_in: np.array([observation])})
    # More noisy as t increases, max_var = 1.0
    n_actions = len(action)
    action += action_noise * np.random.randn(n_actions)
    return np.clip(action, *kwargs['action_bounds'])

def prepare_policy(sess, param_noise, diff_weights, initial_param_std):
    if diff_weights is not None:
        num_weight_vars = diff_weights.shape[0]
        flat_weight_update = param_noise * diff_weights * np.random.randn(num_weight_vars)
        fwu_ph = tf.get_collection('perturb_policy')[0]
        opts = tf.get_collection('perturb_policy')[1:]
        sess.run(opts, feed_dict={fwu_ph: flat_weight_update})
        return np.mean(np.abs(flat_weight_update))
    assert initial_param_std == 0.0
    return 0.0

def write_stats(dict, data):
    for key, value in dict.items():
        if '%' in key:
            value.append(np.percentile(data, int(key[:-1]), axis=0))
        elif key == 'avg':
            value.append(np.mean(data, axis=0))
        elif key == 'batch_size':
            value.append(len(data))
        else:
            assert False
def write_to_csv(data, timesteps, path):
    # Make it 2D np array
    make_values_np_array(data)
    # Save to csv
    import csv
    header = sorted(data.keys())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timesteps'] + header)
        for i, timestep in enumerate(timesteps):
            writer.writerow([str(timestep)] + [str(data[h][i]) for h in header])
            # with open(os.path.join(log_dir,'errors_state_cost_%d.csv'%count), 'w', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['timesteps'] + header)
            #     for i, timestep in enumerate(timesteps):
            #         writer.writerow([str(timestep)] + ["%f"%errors['state_diff'][h][i][-1] if
            #                                            h =='batch_size' else
            #                                            errors['state_diff'][h][i] for h in header])

def make_values_np_array(dict):
    for key, value in dict.items():
        dict[key] = np.array(value)

# Under the current policy and dynamics models, we sample roll-outs from a set of fixed initial states.
# We also perturb the policy on its parameters and compute the average.
def evaluate_model_predictions(env,
                               policy_in,
                               policy_out,
                               dynamics_in,
                               dynamics_out,
                               reset_initial_states,
                               sess,
                               log_dir,
                               count,  # For logging csv
                               max_timestep,
                               cost_np_vec,
                               timesteps=(1, 3, 5, 7, 10, 12, 15, 18, 20, 100)):
    errors = {'timesteps': timesteps,
              'l2_sum': [],
              'l1_sum': [],
              'l1_state_cost': [],
              'state_diff':{
                  '100%':[],
                  '0%':[],
                  '75%':[],
                  '25%':[],
                  '50%':[],
                  'avg':[],
                  'batch_size':[]
              },
              'cost_diff': {
                  '100%': [],
                  '0%': [],
                  '75%': [],
                  '25%': [],
                  '50%': [],
                  'avg': [],
                  'batch_size': []
              }
            }
    # Get numpy arrays Os, As, Rs
    Os, As, Rs = sample_fixed_init_trajectories(env,
                                                policy_in,
                                                policy_out,
                                                reset_initial_states,
                                                sess,
                                                max_timestep)

    assert max_timestep == Rs.shape[1]
    # Compute the errors
    for timestep in timesteps:
        n_states = Os.shape[2]
        Xs = np.reshape(Os[:, :-timestep, :], (-1, n_states))
        Ys = np.reshape(Os[:, timestep:, :], (-1, n_states))
        costs = np.zeros(len(Xs))
        rewards = np.zeros(len(Xs))
        observations = Xs
        for t in range(timestep):
            actions = sess.run(policy_out, feed_dict={policy_in: observations})
            actions = np.clip(actions, *env.action_space.bounds)
            next_observations = sess.run(dynamics_out,
                                         feed_dict={dynamics_in: np.concatenate([observations, actions], axis=1)})
            costs += cost_np_vec(observations, actions, next_observations)
            rewards += np.reshape(Rs[:, t:t+max_timestep+1-timestep], -1)
            # Update observations
            observations = next_observations

        # Get the different after t steps
        state_diff = np.abs(Ys - observations)
        cost_diff = np.abs(costs + rewards)
        # Add errors
        errors['l1_sum'].append(np.mean(np.sum(state_diff, axis=1)))
        errors['l2_sum'].append(np.mean(np.sum(state_diff, axis=1)))
        errors['l1_state_cost'].append(np.mean(state_diff[:, -1]))
        write_stats(errors['state_diff'], state_diff)
        write_stats(errors['cost_diff'], cost_diff)

    write_to_csv(errors['state_diff'], timesteps,
                 os.path.join(log_dir, 'state_diff_%d.csv' % count))
    write_to_csv(errors['cost_diff'], timesteps,
                 os.path.join(log_dir, 'cost_diff_%d.csv' % count))
    return errors

# TODO: fix when early stop
def get_error_distribution(policy_in,
                           policy_out,
                           dynamics_in,
                           dynamics_out,
                           env,
                           cost_np_vec,
                           sess,
                           logger,
                           log_dir,
                           count,
                           horizon=100,
                           sample_size=100,
                           known_actions=False,
                           is_plot=False
                           ):
    real_costs = []
    initial_states = []
    actions = []
    real_final_states = []
    # Compute real costs
    for i in range(sample_size):
        x = env.reset()
        initial_states.append(x)
        real_cost = 0
        _action = []
        for t in range(horizon):
            action = sess.run(policy_out,
                              feed_dict={policy_in: x[None]})[0]
            x, r, done, _ = env.step(action)
            _action.append(action)
            real_cost -= r
            if done:
                break
        actions.append(_action)
        real_costs.append(real_cost)
        real_final_states.append(x)
    real_costs = np.array(real_costs)
    real_final_states = np.array(real_final_states)

    # Compute estimated costs
    o = np.array(initial_states)
    actions = np.clip(actions, -1, 1)
    estimated_costs = np.zeros_like(real_costs)
    for t in range(horizon):
        # Sim step
        if known_actions:
            a = actions[:, t, :]
        else:
            a = np.clip(sess.run(
                policy_out, feed_dict={policy_in: o}
            ), *env.action_space.bounds)
        o_next = sess.run(dynamics_out,
                          feed_dict={dynamics_in: np.concatenate([o, a], axis=1)})
        estimated_costs += cost_np_vec(o, a, o_next)
        # update
        o = o_next
    # Plot
    e_cost = estimated_costs - real_costs
    e_state = o - real_final_states
    loss = np.sum(np.square(e_state), axis=1)

    logger.info('### Real cost ###')
    logger.info('mean: {}'.format(np.mean(real_costs)))
    logger.info('std: {}'.format(np.std(real_costs)))
    logger.info('median: {}'.format(np.median(real_costs)))

    logger.info("### Total cost difference ###")
    logger.info('mean: {}'.format(np.mean(e_cost)))
    logger.info('std: {}'.format(np.std(e_cost)))
    logger.info('median: {}'.format(np.median(e_cost)))

    logger.info("### Final state error ###")
    logger.info('mean: {}'.format(np.mean(loss)))
    logger.info('std: {}'.format(np.std(loss)))
    logger.info('median: {}'.format(np.median(loss)))

    logger.info("### Dimension mean ###")
    logger.info(np.mean(np.square(e_state), axis=0))
    if is_plot:
        import matplotlib as mpl
        mpl.use('Agg')
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        x = pd.Series(e_cost, name="total cost difference")
        sns.distplot(x)
        plt.savefig(os.path.join(log_dir, 'cost_diff_dist_%d.png' % count))
        plt.close()

        plt.figure()
        x = pd.Series(loss, name="final state prediction error (L2)")
        sns.distplot(x, color='g')
        plt.savefig(os.path.join(log_dir, 'state_diff_dist_%d.png' % count))
        plt.close()
    return (e_cost, e_state)

def test_policy_cost(policy_init,
                     policy_cost,
                     policy_in,
                     policy_out,
                     dynamics_in,
                     dynamics_out,
                     env,
                     cost_np_vec,
                     sess,
                     horizon=100,
                     sample_size=10,
                     is_done=None
                     ):
    '''
    :return: verify if the trajectory costs computed in tensorflow and numpy.
    We check the average over sample_size trajectories.
    '''
    initial_states = np.array([env.reset() for i in range(sample_size)])
    estimated_policy_cost = sess.run(policy_cost, feed_dict={policy_init: initial_states})
    o = initial_states
    estimated_cost = 0
    dones = np.array([False for i in range(sample_size)])
    for t in range(horizon):
        # Sim step
        a = np.clip(sess.run(
                policy_out, feed_dict={policy_in: o}
            ), *env.action_space.bounds)
        o_next = sess.run(dynamics_out,
                          feed_dict={dynamics_in: np.concatenate([o, a], axis=1)})
        estimated_cost += np.mean((1-dones)*cost_np_vec(o, a, o_next))
        o = o_next
        if is_done is not None and is_done(o, o_next).any():
            dones[is_done(o, o_next)] = True
    print(estimated_cost, estimated_policy_cost)
    assert np.allclose(estimated_cost, estimated_policy_cost)

# Make sure that this can be found and matched to the policy learning curve.

def sample_fixed_init_trajectories(env,
                                   policy_in,
                                   policy_out,
                                   reset_initial_states,
                                   sess,
                                   max_timestep):
    Os = []
    As = []
    Rs = []
    for x in reset_initial_states:
        _os = []
        _as = []
        _rs = []
        if hasattr(env.wrapped_env, 'wrapped_env'):
            observation = env.wrapped_env.wrapped_env.reset(x)
        else:
            env.reset()
            half = int(len(x)/2)
            inner_env = env.wrapped_env.env.unwrapped
            inner_env.set_state(x[:half], x[half:])
            observation = inner_env._get_obs()
        _os.append(observation)

        for t in range(max_timestep):
            action = sess.run(policy_out, feed_dict={policy_in: observation[None]})
            next_observation, reward, done, info = env.step(action[0])

            _os.append(next_observation)
            _as.append(action[0])
            _rs.append(reward)
            # Update observation
            observation = next_observation
            if done:
                break
        Os.append(_os)
        As.append(_as)
        Rs.append(_rs)
    return np.array(Os), np.array(As), np.array(Rs)


# Sample a batch of trajectories from an environment
# Use tensorflow policy given as (in and out).
# Batch size is the total number of transitions (not trajectories).
def sample_trajectories(env,
                        policy_in,
                        policy_out,
                        exploration,
                        batch_size,
                        saver,
                        diff_weights,
                        log_dir,
                        logger,
                        is_monitored,
                        monitorpath,
                        sess,
                        max_timestep,
                        render_every=None,
                        cost_np=None,
                        is_done=None):

    saver.save(sess,
               os.path.join(log_dir, 'policy.ckpt'),
               write_meta_graph=False)

    if is_monitored:
        from gym import wrappers
        env = wrappers.Monitor(env, monitorpath)
    Os = []
    As = []
    Rs = []
    max_eps_reward = -np.inf
    min_eps_reward = np.inf
    avg_eps_reward = 0.0
    _counter = 1
    while _counter <= batch_size:
        o = []
        a = []
        r = []
        if is_monitored:
            env.stats_recorder.done = True
        observation = env.reset()
        o.append(observation)
        episode_reward = 0.0
        avg_weight_change = prepare_policy(sess,
                                           exploration['param_noise'],
                                           diff_weights,
                                           exploration['initial_param_std'])
        for t in range(max_timestep):
            # Perturb policy.
            if exploration['vary_trajectory_noise']:
                action_noise = exploration['action_noise']*np.random.uniform()
            else:
                action_noise = exploration['action_noise']
            action = get_action(observation,
                                policy_in,
                                policy_out,
                                sess,
                                action_noise=action_noise,
                                action_bounds=env.action_space.bounds)
            observation, reward, done, info = env.step(action)
            # Debug is_done
            if is_done is not None:
                assert done == is_done(o[-1][None], observation[None])[0]
            o.append(observation)
            a.append(action[0])
            r.append(reward)
            episode_reward += reward
            _counter += 1
            if render_every is not None and len(Os) % render_every == 0:
                env.render()
            if done:
                break
        # debugging cost function
        if cost_np is not None:
            episode_cost = len(a) * cost_np(np.array(o[:-1]),
                                            np.array(a),
                                            np.array(o[1:]))
            # Check if cost_np + env_reward == 0
            logger.info('%d steps, cost %.2f, verify_cost %.3f, avg_weight_change %.3f'
                        % (_counter - 1,
                           episode_cost,
                           episode_reward + episode_cost,
                           avg_weight_change))
        else:
            logger.info('%d steps, reward %.2f, avg_weight_change %.3f'
                        % (_counter - 1, episode_reward, avg_weight_change))
        # Recover policy
        saver.restore(sess, os.path.join(log_dir, 'policy.ckpt'))
        logger.debug("Restored the policy back to %s" % os.path.join(log_dir, 'policy.ckpt'))

        Os.append(o)
        As.append(a)
        Rs.append(r)
        # Update stats
        avg_eps_reward += episode_reward
        if episode_reward > max_eps_reward:
            max_eps_reward = episode_reward
        if episode_reward < min_eps_reward:
            min_eps_reward = episode_reward

    avg_eps_reward /= len(Os)
    rllab_logger.record_tabular('EpisodesSoFar', len(Os))
    rllab_logger.record_tabular('TimeStepsSoFar', _counter - 1)
    return Os, As, Rs, {'avg_eps_reward': avg_eps_reward,
                        'min_eps_reward': min_eps_reward,
                        'max_eps_reward': max_eps_reward}

def reset_batch(envs, reset_initial_states):
    obs = []
    for env, x in zip(envs, reset_initial_states):
        if hasattr(env.wrapped_env, 'wrapped_env'):
            obs.append(env.wrapped_env.wrapped_env.reset(x))
        else:
            env.reset()
            half = int(len(x) / 2)
            inner_env = env.wrapped_env.env.unwrapped
            inner_env.set_state(x[:half], x[half:])
            obs.append(inner_env._get_obs())
    return np.array(obs)

def step_batch(envs, actions):
    next_steps = [env.step(action) for (env, action) in zip(envs, actions)]
    next_obs, rs, ds, infos = list(zip(*next_steps))
    return np.array(next_obs), np.array(rs), np.array(ds), infos

# Given a batch of initial states and a policy, do deterministic rollout on real env.
# Don't render. Add cost function evaluation.
def evaluate_fixed_init_trajectories(env,
                                     policy_in,
                                     policy_out,
                                     reset_initial_states,
                                     cost_np_vec, sess,
                                     max_timestep=100,
                                     gamma=1.0):
    import pickle
    n_envs = len(reset_initial_states)
    envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_envs)]
    observations = reset_batch(envs, reset_initial_states)
    dones = [False for _ in range(n_envs)]
    cost = 0.0
    reward = 0.0
    for t in range(max_timestep):
        actions = sess.run(policy_out, feed_dict={policy_in: observations})
        # clipping
        actions = np.clip(actions, *env.action_space.bounds)
        next_observations, _rewards, _dones, _ = step_batch(envs, actions)
        dones = np.logical_or(dones, _dones)
        # Update rewards and costs
        rewards = (1.0 - dones) * _rewards * gamma**t
        costs = (1.0-dones)*cost_np_vec(observations, actions, next_observations) * gamma**t
        # Update observation
        observations = next_observations
        cost += np.mean(costs)
        reward += np.mean(rewards)
    assert cost + reward < 1e-2
    return cost

# def evaluate_learned_dynamics_trajectories(dynamics_in,
#                                            dynamics_out,
#                                            policy_in,
#                                            policy_out,
#                                            initial_states,
#                                            cost_np, sess,
#                                            max_timestep=100):
#     batch_size, n_states = initial_states.shape
#     avg_eps_cost = 0.0
#
#     observations = initial_states
#     for t in range(max_timestep):
#         actions = sess.run(policy_out, feed_dict={policy_in: observations})
#         actions = np.clip(actions, -1.0, 1.0)
#         # Only using model 0.
#         next_observations = sess.run(dynamics_out,
#                                      feed_dict={dynamics_in: np.concatenate([observations, actions], axis=1)})
#         avg_cost = cost_np(observations, actions, next_observations)
#         # Update observations
#         observations = next_observations
#         # Update cost
#         avg_eps_cost += avg_cost
#     return avg_eps_cost

from rllab.envs.base import Env
from rllab.envs.base import Step
class NeuralNetEnv(Env):
    def __init__(self, env, inner_env, cost_np, dynamics_in, dynamics_outs, sam_mode):
        self.vectorized = True
        self.env = env
        self.cost_np = cost_np
        self.is_done = getattr(inner_env, 'is_done', lambda x, y: np.asarray([False] * len(x)))
        self.dynamics_in = dynamics_in
        self.dynamics_outs = dynamics_outs
        self.n_models = len(dynamics_outs)
        self.sam_mode = sam_mode
        super(NeuralNetEnv, self).__init__()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self._state = self.env.reset()
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        sess = tf.get_default_session()
        action = np.clip(action, *self.action_space.bounds)
        index = np.random.randint(self.n_models)
        next_observation = sess.run(self.dynamics_outs[index],
                                    feed_dict={self.dynamics_in: np.concatenate([self._state, action])[None]})
        reward = - self.cost_np(self._state[None], action[None], next_observation)
        done = self.is_done(self._state[None], next_observation)[0]
        self._state = np.reshape(next_observation, -1)
        return Step(observation=self._state, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

    def vec_env_executor(self, n_envs, max_path_length):
        return VecSimpleEnv(env=self, n_envs=n_envs, max_path_length=max_path_length)


class VecSimpleEnv(object):
    def __init__(self, env, n_envs, max_path_length):
        self.env = env
        self.n_envs = n_envs
        self.num_envs = n_envs
        self.states = np.zeros((self.n_envs, env.observation_space.shape[0]))
        self.ts = np.zeros((self.n_envs,))
        self.max_path_length = max_path_length
        self.cur_model_idx = np.random.randint(len(self.env.dynamics_outs), size=(n_envs, ))

    def reset(self, dones=None):
        if dones is None:
            dones = np.asarray([True] * self.n_envs)
        else:
            dones = np.cast['bool'](dones)
        for i, done in enumerate(dones):
            if done:
                self.states[i] = self.env.reset()
                self.cur_model_idx[i] = np.random.randint(len(self.env.dynamics_outs))
        self.ts[dones] = 0
        return self.states[dones]

    def step(self, actions):
        self.ts += 1
        actions = np.clip(actions, *self.env.action_space.bounds)
        next_observations = self.get_next_observation(actions)
        rewards = - self.env.cost_np(self.states, actions, next_observations)
        self.states = next_observations
        dones = self.env.is_done(self.states, next_observations)
        dones[self.ts >= self.max_path_length] = True
        if np.any(dones):
            self.reset(dones)
        return self.states, rewards, dones, dict()

    def get_next_observation(self, actions):
        sess = tf.get_default_session()
        sam_mode = self.env.sam_mode
        next_possible_observations = sess.run(self.env.dynamics_outs,
                                              feed_dict={self.env.dynamics_in:
                                                             np.concatenate([self.states, actions],
                                                                            axis=1)})
        next_possible_observations = np.array(next_possible_observations)
        if sam_mode == 'step_rand':
            # Choose a random model for each batch.
            indices = np.random.randint(self.env.n_models, size=self.n_envs)
            next_observations = next_possible_observations[indices, range(self.n_envs)]
        elif sam_mode == 'eps_rand':
            indices = self.cur_model_idx
            next_observations = next_possible_observations[indices, range(self.n_envs)]
        elif sam_mode == 'model_mean_std':
            std = np.std(next_possible_observations, axis=0)
            next_observations = np.mean(next_possible_observations, axis=0) + np.random.normal(size=std.shape)*std
        elif sam_mode == 'model_mean':
            next_observations = np.mean(next_possible_observations, axis=0)
        elif sam_mode == 'model_med':
            next_observations = np.median(next_possible_observations, axis=0)
        elif sam_mode == 'one_model':
            next_observations = next_possible_observations[0]
        else:
            assert False, "sam mode %s is not defined." % sam_mode
        return next_observations