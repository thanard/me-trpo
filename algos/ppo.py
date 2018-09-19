


from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class PPO(BatchPolopt):
    """
    Proximal Policy Optimization.
    """

    def __init__(
            self,
            clip_lr=0.3,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            min_penalty=1e-3,
            max_penalty=1e6,
            entropy_bonus_coeff=0.,
            gradient_clipping=40.,
            log_loss_kl_before=True,
            log_loss_kl_after=True,
            use_kl_penalty=False,
            initial_kl_penalty=1.,
            use_line_search=True,
            max_backtracks=10,
            backtrack_ratio=0.5,
            optimizer=None,
            step_size=0.01,
            min_n_epochs=2,
            adaptive_learning_rate=False,
            max_learning_rate=1e-3,
            min_learning_rate=1e-5,
            **kwargs
    ):
        self.clip_lr = clip_lr
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.gradient_clipping = gradient_clipping
        self.log_loss_kl_before = log_loss_kl_before
        self.log_loss_kl_after = log_loss_kl_after
        self.use_kl_penalty = use_kl_penalty
        self.initial_kl_penalty = initial_kl_penalty
        self.use_line_search = use_line_search
        self.max_backtracks = max_backtracks
        self.backtrack_ratio = backtrack_ratio
        self.step_size = step_size
        self.min_n_epochs = min_n_epochs
        self.adaptive_learning_rate = adaptive_learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        policy = kwargs['policy']
        if optimizer is None:
            optimizer = AdamOptimizer()
        self.optimizer = optimizer
        super().__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None,) * (1 + is_recurrent) + shape, name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None,) * (1 + is_recurrent) + shape, name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        kl_penalty_var = tf.Variable(
            initial_value=self.initial_kl_penalty,
            dtype=tf.float32,
            name="kl_penalty"
        )

        # TODO: The code below only works for FF policy.
        assert is_recurrent == 0

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        ent = tf.reduce_mean(dist.entropy_sym(dist_info_vars))
        mean_kl = tf.reduce_mean(kl)

        clipped_lr = tf.clip_by_value(lr, 1. - self.clip_lr, 1. + self.clip_lr)

        surr_loss = - tf.reduce_mean(lr * advantage_var)
        clipped_surr_loss = - tf.reduce_mean(
            tf.minimum(lr * advantage_var, clipped_lr * advantage_var)
        )

        clipped_surr_pen_loss = clipped_surr_loss - self.entropy_bonus_coeff * ent
        if self.use_kl_penalty:
            clipped_surr_pen_loss += kl_penalty_var * tf.maximum(0., mean_kl - self.step_size)

        self.optimizer.update_opt(
            loss=clipped_surr_pen_loss,
            target=self.policy,
            inputs=[obs_var, action_var, advantage_var] + state_info_vars_list + old_dist_info_vars_list,
            diagnostic_vars=OrderedDict([
                ("UnclippedSurrLoss", surr_loss),
                ("MeanKL", mean_kl),
            ])
        )
        self.kl_penalty_var = kl_penalty_var
        self.f_increase_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                tf.minimum(kl_penalty_var * self.increase_penalty_factor, self.max_penalty)
            )
        )
        self.f_decrease_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                tf.maximum(kl_penalty_var * self.decrease_penalty_factor, self.min_penalty)
            )
        )
        self.f_reset_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                self.initial_kl_penalty
            )
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        # logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        # logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        # logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        # logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        # logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        # logger.record_tabular('LossBefore', loss_before)
        # logger.record_tabular('LossAfter', loss_after)
        # logger.record_tabular('MeanKLBefore', mean_kl_before)
        # logger.record_tabular('MeanKL', mean_kl)
        # logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
