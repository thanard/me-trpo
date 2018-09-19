import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from samplers.batch_sampler import BatchSampler
from samplers.vectorized_sampler import VectorizedSampler
import numpy as np


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.kwargs = kwargs
        if 'reset_init_path' in self.kwargs:
            assert 'horizon' in self.kwargs
            import pickle
            with open(kwargs['reset_init_path'], 'rb') as f:
                self.reset_initial_states = pickle.load(f)
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, determ=False):
        return self.sampler.obtain_samples(itr, determ)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def evaluate_fixed_init_trajectories(self,
                                         reset_initial_states,
                                         horizon):
        def f(x):
            if hasattr(self.env.wrapped_env, 'wrapped_env'):
                inner_env = self.env.wrapped_env.wrapped_env
                observation = inner_env.reset(x)
            else:
                self.env.reset()
                half = int(len(x) / 2)
                inner_env = self.env.wrapped_env.env.unwrapped
                inner_env.set_state(x[:half], x[half:])
                observation = inner_env._get_obs()
            episode_reward = 0.0
            episode_cost = 0.0
            for t in range(horizon):
                action = self.policy.get_action(observation)[1]['mean'][None]
                # clipping
                action = np.clip(action, *self.env.action_space.bounds)
                next_observation, reward, done, info = self.env.step(action[0])
                cost = inner_env.cost_np(observation[None], action, next_observation[None])
                # Update observation
                observation = next_observation
                # Update cost
                episode_cost += cost
                # Update reward
                episode_reward += reward
            # assert episode_cost + episode_reward < 1e-2
            return episode_cost

        # Run evaluation in parallel
        outs = np.array(list(map(f, reset_initial_states)))
        # Return avg_eps_reward and avg_eps_cost accordingly
        return np.mean(outs)

    def train(self):
        if 'initialized_path' in self.kwargs:
            import joblib
            from utils import get_session
            sess = get_session(interactive=True, mem_frac=0.1)
            data = joblib.load(self.kwargs['initialized_path'])
            self.policy = data['policy']
            self.env = data['env']
            self.baseline = data['baseline']
            self.init_opt()
            sess.run(tf.assign(self.policy._l_std_param.param, np.zeros(
                self.env.action_space.shape[0]
            )))
            uninitialized_vars=[]
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            init_new_vars_op = tf.variables_initializer(uninitialized_vars)
            sess.run(init_new_vars_op)
            self.start_worker()
            start_time = time.time()
            avg_eps_costs= []
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)
                    if 'horizon' in self.kwargs:
                        # paths = self.obtain_samples(itr, True)
                        avg_cost = self.evaluate_fixed_init_trajectories(
                            self.reset_initial_states,
                            self.kwargs['horizon'])
                        logger.record_tabular('validation_cost', avg_cost)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.start_worker()
                start_time = time.time()
                for itr in range(self.start_itr, self.n_itr):
                    itr_start_time = time.time()
                    with logger.prefix('itr #%d | ' % itr):
                        logger.log("Obtaining samples...")
                        paths = self.obtain_samples(itr)
                        logger.log("Processing samples...")
                        samples_data = self.process_samples(itr, paths)
                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(paths)
                        logger.log("Optimizing policy...")
                        self.optimize_policy(itr, samples_data)
                        logger.log("Saving snapshot...")
                        params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                        if self.store_paths:
                            params["paths"] = samples_data["paths"]
                        logger.save_itr_params(itr, params)
                        logger.log("Saved")
                        logger.record_tabular('Time', time.time() - start_time)
                        logger.record_tabular('ItrTime', time.time() - itr_start_time)
                        if 'horizon' in self.kwargs:
                            avg_cost = self.evaluate_fixed_init_trajectories(
                                self.reset_initial_states,
                                self.kwargs['horizon'])
                            logger.record_tabular('validation_cost', avg_cost)
                        logger.dump_tabular(with_prefix=False)
                        if self.plot:
                            self.update_plot()
                            if self.pause_for_plot:
                                input("Plotting evaluation run: Press Enter to "
                                      "continue...")
        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
