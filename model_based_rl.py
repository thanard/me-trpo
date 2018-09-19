import tensorflow as tf
import numpy as np
import pickle
import os
import logging
from env_helpers import sample_trajectories, \
    evaluate_fixed_init_trajectories, evaluate_model_predictions, \
    get_error_distribution, test_policy_cost, NeuralNetEnv
from utils import *
from svg_utils import setup_gradients, svg_update
import rllab.misc.logger as rllab_logger
import time
import copy
import joblib

np.set_printoptions(
    formatter={
        'float_kind': lambda x: "%.2f" % x
    }
)
TF_SUMMARY = True

def build_dynamics_graph(scope,
                         dynamics_model,
                         dynamics_in,
                         dynamics_in_full,
                         y_training_full,
                         n_dynamics_input,
                         n_models,
                         get_regularizer_loss,
                         n_states,
                         logger):
    '''
    Build dynamics tensorflow graph at training and test times.
    :param scope:
    :param dynamics_model:
    :param dynamics_in:
    :param y_training:
    :param dynamics_in_full:
    :param y_training_full:
    :param n_dynamics_input:
    :param n_models:
    :param get_regularizer_loss:
    :param n_states:
    :return:
    '''
    # For training
    _dynamics_outs = [
        dynamics_model(
            get_ith_tensor(dynamics_in_full, i, n_dynamics_input),
            scope,
            'model%d' % i,
            collect_summary=TF_SUMMARY
        ) for i in range(n_models)
        ]
    _regularizer_losses = [get_regularizer_loss(scope, 'model%d' % i) for i in range(n_models)]
    _prediction_losses = [
        tf.reduce_mean(
            tf.reduce_sum(
                tf.square(
                    y_predicted - get_ith_tensor(
                        y_training_full,
                        i,
                        n_states
                    )
                ),
                axis=[1]
            )
        )
        for i, y_predicted in enumerate(_dynamics_outs)
        ]
    prediction_loss = tf.reduce_sum(_prediction_losses,
                                    name='total_prediction_loss')
    regularizer_loss = tf.reduce_sum(_regularizer_losses,
                                     name='total_regularizer_loss')
    # Add summaries
    with tf.name_scope('%s/prediction_loss' % scope):
        tf.summary.histogram('dist_over_models', _prediction_losses)
        tf.summary.scalar('summary', prediction_loss)

    assert len(_prediction_losses) == len(_regularizer_losses)
    dynamics_losses = [
        _prediction_losses[i] + _regularizer_losses[i]
        for i in range(len(_prediction_losses))
        ]
    dynamics_loss = tf.add(prediction_loss, regularizer_loss,
                           name='total_dynamics_loss')
    logger.info("Defined %d models in scope %s" % (n_models, scope))

    # At test time
    _dynamics_outs = [
        dynamics_model(
            dynamics_in,
            scope,
            'model%d' % i
        ) for i in range(n_models)
        ]
    # TODO: change this hack back.
    # avg_prediction = tf.reduce_mean(tf.stack(_dynamics_outs, axis=0),
    #                             axis=0,
    #                             name='avg_prediction')
    logger.info("Built prediction network for scope %s" % (scope))
    return dynamics_loss, prediction_loss, regularizer_loss, _dynamics_outs, dynamics_losses


def build_policy_graph(policy_scope,
                       scope,
                       policy_training_init,
                       n_models,
                       policy_opt_params,
                       policy_model,
                       dynamics_model,
                       env,
                       cost_tf,
                       logger,
                       is_env_done_tf=None,
                       stochastic=None):
    # TODO: Think about using avg model in each prediction step.
    _policy_costs = []
    n_saturates = 0 # Debug
    with tf.name_scope(scope):
        for i in range(n_models):
            # Initial states
            x = policy_training_init
            _policy_cost = 0
            dones = 0.0
            for t in range(policy_opt_params.T):
                u = tf.clip_by_value(policy_model(x, stochastic), *env.action_space.bounds)
                n_saturates += tf.cast(tf.equal(tf.abs(u), 1.0), tf.int32)
                x_next = dynamics_model(tf.concat([x, u], axis=1),
                                        scope,
                                        'model%d' % i)
                # Update dones after computing cost.
                if is_env_done_tf is not None:
                    _policy_cost += (policy_opt_params.gamma ** t) * cost_tf(x, u, x_next,
                                                                             dones=dones)
                    dones = tf.maximum(dones, is_env_done_tf(x, x_next))
                else:
                    _policy_cost += (policy_opt_params.gamma ** t) * cost_tf(x, u, x_next)
                # Move forward 1 step.
                x = x_next
            _policy_costs.append(_policy_cost)
            # Average over cost from all dynamics models

    # Collecting summary
    with tf.name_scope('%s/policy_cost' % policy_scope):
        tf.summary.histogram('dist_over_models', _policy_costs)
        tf.summary.scalar('cost_on_model0', _policy_costs[0])
    policy_model(policy_training_init, collect_summary=TF_SUMMARY)
    logger.info("Built %d policy graphs for %s model" % (n_models, scope))
    return _policy_costs, n_saturates


def get_dynamics_optimizer(scope, prediction_loss, reg_loss, dynamics_opt_params, logger):
    with tf.variable_scope('adam_' + scope):
        # Allow learning rate decay schedule.
        if type(dynamics_opt_params.learning_rate) == dict:
            adaptive_lr = tf.Variable(dynamics_opt_params.learning_rate["scratch"], trainable=False)
        else:
            adaptive_lr = dynamics_opt_params.learning_rate
        _prediction_opt = tf.train.AdamOptimizer(learning_rate=adaptive_lr)
        prediction_opt_op = minimize_and_clip(_prediction_opt,
                                            prediction_loss[scope],
                                            var_list=tf.get_collection(
                                                     tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=scope),
                                            collect_summary=TF_SUMMARY
                                            )
        _reg_opt = tf.train.GradientDescentOptimizer(learning_rate=adaptive_lr)
        reg_opt_op = minimize_and_clip(_reg_opt,
                                       reg_loss[scope],
                                       var_list=tf.get_collection(
                                           tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=scope),
                                       collect_summary=TF_SUMMARY
                                       )
        dynamics_opt_op = [prediction_opt_op, reg_opt_op]
        # Get variables and re-initializer.
        _dynamics_adam_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='adam_' + scope)
        dynamics_adam_init = tf.variables_initializer(_dynamics_adam_vars)
        logger.debug('num_%s_adam_variables %d' % (scope, len(_dynamics_adam_vars)))
        return dynamics_opt_op, dynamics_adam_init, adaptive_lr


def get_policy_optimizer(scope, policy_cost, policy_opt_params, logger):
    with tf.variable_scope('adam_' + scope):
        policy_opt = tf.train.AdamOptimizer(learning_rate=policy_opt_params.learning_rate)
        policy_opt_op = minimize_and_clip(
            policy_opt,
            policy_cost,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope),
            clip_val=policy_opt_params.grad_norm_clipping,
            collect_summary=TF_SUMMARY
        )
    # Debugging
    policy_grads_and_vars = policy_opt.compute_gradients(
        policy_cost,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    )
    # Get variables and re-initializer.
    policy_adam_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adam_' + scope)
    policy_adam_init = tf.variables_initializer(policy_adam_vars)
    logger.debug('num_policy_adam_variables %d' % len(policy_adam_vars))
    logger.info("Created policy opt operator.")
    return policy_opt_op, policy_adam_init, policy_grads_and_vars

def create_perturb_policy_opts(policy_scope, shape):
    '''
    :return: flat weight update placeholder, and
    collection of perturb weight operators
    '''
    flat_weight_update_ph = tf.placeholder(tf.float32, shape=shape)
    weights = get_variables(scope=policy_scope, filter='/b:')
    weights.extend(get_variables(scope=policy_scope, filter='/W:'))
    weight_updates = unflatten_tensors(flat_weight_update_ph, weights)
    opts = get_update_variable_opt(weight_updates, weights)
    tf.add_to_collection('perturb_policy', flat_weight_update_ph)
    for opt in opts:
        tf.add_to_collection('perturb_policy', opt)

'''
This method is used for training dynamics and policy models.
The oracle option can be made in policy_opt_params.
The oracle mode will allow an access to true dynamics (oracle)
during policy optimization step (know when to stop).
So, this could be an upper bound on how stable could bootstrapping achieve.
'''


def train_models(env,
                 dynamics_model,
                 dynamics_opt_params,
                 get_regularizer_loss,
                 policy_model,
                 policy_opt_params,
                 rollout_params,
                 cost_np,
                 cost_np_vec,
                 cost_tf,
                 snapshot_dir,
                 working_dir,
                 n_models=1,
                 sweep_iters=10,
                 sample_size=1000,
                 verbose=False,
                 careful_init=False,
                 variant={},
                 **kwargs
                 ):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    u_max = env.action_space.high[0]

    sess = tf.get_default_session()
    assert (sess is not None)

    logger = get_logger(__name__, snapshot_dir)

    is_env_done = getattr(kwargs['inner_env'], 'is_done', None)
    is_env_done_tf = getattr(kwargs['inner_env'], 'is_done_tf', None)
    ###############
    # Build Graph #
    ###############
    '''
    Rollouts
    '''
    policy_scope = 'training_policy'
    policy_in = tf.placeholder(tf.float32, shape=(None, n_states), name='policy_in')
    policy_out = policy_model(policy_in)
    tf.add_to_collection("policy_in", policy_in)
    tf.add_to_collection("policy_out", policy_out)

    '''
    Dynamics Optimization
    '''
    n_dynamics_input = n_states + n_actions
    dynamics_in = tf.placeholder(tf.float32,
                                 shape=(None, n_dynamics_input),
                                 name='dynamics_in')
    dynamics_in_full = tf.placeholder(tf.float32,
                                      shape=(None, n_models * n_dynamics_input),
                                      name='dyanmics_in_full')
    y_training_full = tf.placeholder(tf.float32,
                                     shape=(None, n_models * n_states),
                                     name='y_training_full')
    # Ground-truth next states.
    if policy_opt_params.mode == 'fourth_estimated':
        model_scopes = ["training_dynamics", "validation_dynamics",
                        "second_validation_dynamics", "third_validation_dynamics"]
    elif policy_opt_params.mode == 'third_estimated':
        model_scopes = ["training_dynamics", "validation_dynamics", "second_validation_dynamics"]
    elif policy_opt_params.mode == 'second_estimated':
        model_scopes = ["training_dynamics", "validation_dynamics"]
    elif policy_opt_params.mode == 'estimated' or policy_opt_params.mode == 'trpo_mean': #TODO: bad hacking
        model_scopes = ["training_dynamics"]
    else:
        # assert 'real' == policy_opt_params.mode
        model_scopes = ["training_dynamics"]
        # model_scopes = ["training_dynamics", "validation_dynamics"]
    dynamics_loss = {}
    prediction_loss = {}
    reg_loss = {}
    dynamics_outs = {}
    dynamics_losses = {}
    for scope in model_scopes:
        dynamics_loss[scope], prediction_loss[scope], reg_loss[scope], \
        dynamics_outs[scope], dynamics_losses[scope] = \
            build_dynamics_graph(scope,
                                 dynamics_model,
                                 dynamics_in,
                                 dynamics_in_full,
                                 y_training_full,
                                 n_dynamics_input,
                                 n_models,
                                 get_regularizer_loss,
                                 n_states,
                                 logger)
        for i in range(n_models):
            tf.add_to_collection('%s_out' % scope, dynamics_outs[scope][i])
    tf.add_to_collection('dynamics_in', dynamics_in)

    # Adam optimizers
    dynamics_opt_op = {}
    dynamics_adam_init = []
    lr_up = []
    lr_ph = tf.placeholder(tf.float32, shape=())
    for scope in model_scopes:
        dynamics_opt_op[scope], _dynamics_adam_init, model_lr = \
            get_dynamics_optimizer(scope,
                                   prediction_loss,
                                   reg_loss,
                                   dynamics_opt_params,
                                   logger)
        lr_up.append(model_lr.assign(lr_ph))
        dynamics_adam_init.append(_dynamics_adam_init)
    logger.info("Created dynamics opt operator.")

    # File writers
    train_writer = tf.summary.FileWriter(os.path.join(snapshot_dir, 'tf_logs/train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(snapshot_dir, 'tf_logs/val'), sess.graph)

    '''
    Policy Optimization
    '''
    policy_training_init = tf.placeholder(tf.float32, shape=(None, n_states), name='x_optimizing')
    policy_costs = {}
    stochastic = tf.Variable(0.0, trainable=False)
    set_stochastic = [stochastic.assign(0.0), stochastic.assign(1.0)]
    for scope in model_scopes:
        policy_costs[scope], n_saturates = build_policy_graph(policy_scope,
                                                 scope,
                                                 policy_training_init,
                                                 n_models,
                                                 policy_opt_params,
                                                 policy_model,
                                                 dynamics_model,
                                                 env,
                                                 cost_tf,
                                                 logger,
                                                 is_env_done_tf,
                                                 stochastic)

    # Setting up for BPTT, TRPO, SVG, and L-BFGS
    # TODO play with different training dynamics model.
    training_policy_cost = tf.reduce_mean(policy_costs['training_dynamics'])
    training_models = dynamics_outs['training_dynamics']
    # mid = int(n_models/2)
    # topk_values = tf.nn.top_k(policy_costs['training_dynamics'], mid+1)[0]
    # if 2*mid==n_models:
    #     training_policy_cost = (topk_values[-1] + topk_values[-2])/2
    # else:
    #     training_policy_cost = topk_values[-1]
    if kwargs['algo_name'] == 'trpo' or kwargs['algo_name'] == 'vpg':
        from envs.base import TfEnv
        kwargs["rllab_algo"].env = TfEnv(NeuralNetEnv(env=env,
                                                      inner_env=kwargs['inner_env'],
                                                      cost_np=cost_np_vec,
                                                      dynamics_in=dynamics_in,
                                                      dynamics_outs=training_models,
                                                      sam_mode=policy_opt_params.sam_mode))
    elif kwargs['algo_name'] == 'svg':
        dynamics_configs = dict(scope_name='training_dynamics', variable_name='0')
        cost_gradient, policy_gradient, model_gradient, theta_vars = \
            setup_gradients(policy_model,
                            dynamics_model,
                            cost_tf,
                            sess,
                            kwargs['inner_env'],
                            dynamics_configs)
        logger.info('Build SVG update graph.')
    elif kwargs['algo_name'] == 'l-bfgs':
        train_step = tf.contrib.opt.ScipyOptimizerInterface(
            training_policy_cost,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='training_policy'),
            method='L-BFGS-B'
        )
        tf.add_to_collection("l-bfgs_step", train_step)

    if "bptt" in kwargs['algo_name']:
        policy_opt_op, policy_adam_init, policy_grads_and_vars = \
            get_policy_optimizer(policy_scope, training_policy_cost, policy_opt_params, logger)
    else:
        policy_opt_op = []
        policy_adam_init = []
        policy_grads_and_vars = []

    '''
    Prepare variables and data for learning
    '''
    if careful_init:
        all_vars = []
        policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy_scope)
        try:
            sess.run(policy_vars)
            logger.info('Do not re-initialize policy.')
        except tf.errors.FailedPreconditionError:
            # We don't want to reinitialize the policy if it has been initialized.
            all_vars.extend(policy_vars)
            logger.info('Re-initialize policy.')
        for scope in model_scopes:
            all_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        network_init = tf.variables_initializer(all_vars)
        sess.run([policy_adam_init, network_init] + dynamics_adam_init)
    else:
        # Initialize all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        logger.info('Re-initialize policy.')

    '''
    Policy weights
    '''
    policy_weights = get_variables(scope=policy_scope, filter='/b:')
    policy_weights.extend(get_variables(scope=policy_scope, filter='/W:'))
    flat_policy_weights = flatten_tensors(policy_weights)
    create_perturb_policy_opts(policy_scope, flat_policy_weights.shape)

    ############# Dynamics validation data ###############
    dynamics_data = {}
    dynamics_validation = {}
    datapath = os.path.join(working_dir, rollout_params.datapath)
    for scope in model_scopes:
        dynamics_data[scope] = data_collection(max_size=rollout_params.training_data_size)
        dynamics_validation[scope] = data_collection(max_size=rollout_params.validation_data_size)
    if os.path.isfile(datapath) and rollout_params.load_rollout_data:
        for scope in model_scopes:
            with open(datapath, 'rb') as f:
                training_data = pickle.load(f)
                split_ratio = rollout_params.split_ratio
                validation_size = round(training_data['dc'].n_data * split_ratio/(1.-split_ratio))
            dynamics_data[scope].clone(training_data['dc'])
            dynamics_validation[scope].clone(training_data['dc_valid'], validation_size)
        logger.warning('Load data collections from %s.' % rollout_params.datapath)
    else:
        logger.warning('Start dynamics data and validation collection from scratch.')

    ############# Policy validation data (fixed) ###############
    vip = os.path.join(working_dir, policy_opt_params.validation_init_path)
    vrip = os.path.join(working_dir, policy_opt_params.validation_reset_init_path)
    if os.path.isfile(vip) and os.path.isfile(vrip):
        with open(vip, 'rb') as f:
            policy_validation_init = pickle.load(f)
        logger.info('Loaded policy val init state data from %s.' % vip)
        with open(vrip, 'rb') as f:
            policy_validation_reset_init = pickle.load(f)
        logger.info('Loaded policy val reset init state data from %s.' % vrip)
    elif vip == vrip:
        # We know that reset is correct, e.g., swimmer.
        policy_validation_init = [env.reset() for i in range(policy_opt_params.batch_size)]
        policy_validation_reset_init = np.array(policy_validation_init)
        with open(vip, 'wb') as f:
            pickle.dump(policy_validation_init, f)
        logger.info('Created %s contains policy validation initial state data.' % vip)
    else:
        # Make sure that the reset works with the representation.
        # If not generate this manually.
        policy_validation_init = []
        policy_validation_reset_init = []
        for i in range(policy_opt_params.batch_size):
            init = env.reset()
            if hasattr(env._wrapped_env, '_wrapped_env'):
                inner_env = env._wrapped_env._wrapped_env
            else:
                inner_env = env._wrapped_env.env.unwrapped
            reset_init = np.concatenate(
                [inner_env.model.data.qpos[:, 0],
                 inner_env.model.data.qvel[:, 0]])
            if hasattr(env._wrapped_env, '_wrapped_env'):
                assert np.allclose(init, inner_env.reset(reset_init))
            policy_validation_init.append(init)
            policy_validation_reset_init.append(reset_init)
        policy_validation_init = np.array(policy_validation_init)
        policy_validation_reset_init = np.array(policy_validation_reset_init)
        with open(vip, 'wb') as f:
            pickle.dump(policy_validation_init, f)
        logger.info('Created %s contains policy validation initial state data.' % vip)
        with open(vrip, 'wb') as f:
            pickle.dump(policy_validation_reset_init, f)
        logger.info('Created %s contains policy validation reset initial state data.' % vrip)

    '''
	Saver
	'''
    log_dir = os.path.join(snapshot_dir, 'training_logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess, os.path.join(log_dir, 'policy-and-models-0.ckpt'))

    # Model savers
    dynamics_savers = {}
    dynamics_vars = []
    for scope in dynamics_data.keys():
        dynamics_savers[scope] = []
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        dynamics_vars.extend(var_list)
        # TODO:here assuming #validation models = #training models = n_models
        for i in range(n_models):
            vars_i = list(filter(lambda x: '%s/model%d' % (scope, i) in x.name, var_list))
            dynamics_savers[scope].append(tf.train.Saver(vars_i))
            logger.info('Dynamics saver %d in scope %s has %d variables.' % (i, scope, len(vars_i)))
    logger.info('Total dynamics var %d' % len(dynamics_vars))

    # Model initializers
    dynamics_initializer = tf.variables_initializer(dynamics_vars)

    # Model summaries
    dynamics_summaries = {}
    for scope in dynamics_data.keys():
        dynamics_summaries[scope] = []
        for i in range(n_models):
            # TODO: Hack to save global model stats
            # TODO: Hack adam_ + namescope is supposed to be global.
            name_scope = "%s" % scope
            #name_scope = "%s/model%d" % (scope, i)
            merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=name_scope)+\
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='adam_'+name_scope)
                                      )
            dynamics_summaries[scope].append(merged)
    logger.info('Summaries merged.')

    # Policy saver
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_scope)
    policy_saver = tf.train.Saver(var_list)
    logger.info('Policy saver has %d variables.' % (len(var_list)))
    logger.debug(''.join([var.name for var in var_list]))
    # Policy summaries
    policy_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=policy_scope) +\
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='adam_'+policy_scope)
                                      )

    ###############
    ## Learning ###
    ###############
    start_time = time.time()
    count = 1
    diff_weights = None
    while True:
        itr_start_time = time.time()
        # Save params every iteration
        joblib.dump(kwargs["saved_policy"], os.path.join(snapshot_dir, 'training_logs/params_%d.pkl' % count))
        reinit_every = int(dynamics_opt_params.reinitialize)
        if reinit_every <= 0 or count % reinit_every != 1:
            reinitialize = False
        else:
            reinitialize = True
        if count == 1:
            reinitialize = True
        # # Check if policy cost is computed consistently using tf and np.
        # for i in range(n_models):
        #     test_policy_cost(policy_training_init,
        #                      policy_costs["training_dynamics"][i],
        #                      policy_in,
        #                      policy_out,
        #                      dynamics_in,
        #                      dynamics_outs["training_dynamics"][i],
        #                      env,
        #                      cost_np_vec,
        #                      sess,
        #                      horizon=policy_opt_params.T,
        #                      is_done=is_env_done)

        # Policy Rollout
        logger.info('\n\nPolicy Rollout and Cost Estimation: %d' % (count))
        rollout_data = collect_data(env,
                                    sample_size,
                                    dynamics_data,
                                    dynamics_validation,
                                    policy_in,
                                    policy_out,
                                    cost_np,  # for debugging
                                    sess,
                                    diff_weights,  # for param_noise exploration
                                    policy_saver,  # for param_noise exploration
                                    log_dir,  # for param_noise exploration
                                    rollout_params,
                                    count,
                                    logger,
                                    is_env_done,
                                    kwargs['input_rms'],
                                    kwargs['diff_rms'])
        rllab_logger.record_tabular('collect_data_time', time.time() - itr_start_time)
        current_time = time.time()

        # Dynamics Optimization
        logger.info('\n\nDynamics Optimization: %d' % (count))
        dynamics_learning_logs = optimize_models(dynamics_data,
                                                 dynamics_validation,
                                                 dynamics_opt_op,
                                                 dynamics_opt_params,
                                                 dynamics_adam_init,
                                                 dynamics_loss,  # for debugging
                                                 dynamics_losses,
                                                 prediction_loss,  # for debugging
                                                 dynamics_in_full,
                                                 y_training_full,
                                                 sess,
                                                 verbose,
                                                 dynamics_savers, log_dir, logger,
                                                 n_models,
                                                 dynamics_initializer,
                                                 dynamics_summaries,
                                                 train_writer,
                                                 val_writer,
                                                 lr_up,
                                                 lr_ph,
                                                 reinitialize)
        rllab_logger.record_tabular('model_opt_time', time.time() - current_time)
        current_time = time.time()

        # # Evaluate training dynamics model
        # logger.info('\n\nEvaluate training dynamics model: %d' % (count))
        # # Compute model prediction errors T steps ahead.
        # errors = evaluate_model_predictions(env,
        #                                     policy_in,
        #                                     policy_out,
        #                                     dynamics_in,
        #                                     avg_prediction['training_dynamics'],
        #                                     policy_validation_reset_init,
        #                                     sess,
        #                                     log_dir,
        #                                     count,
        #                                     max_timestep=policy_opt_params.T,
        #                                     cost_np_vec=cost_np_vec)

        # # Compute an error distribution.
        # errors = get_error_distribution(policy_in,
        #                                 policy_out,
        #                                 dynamics_in,
        #                                 training_models[0],
        #                                 env,
        #                                 cost_np_vec,
        #                                 sess,
        #                                 logger,
        #                                 log_dir,
        #                                 count,
        #                                 horizon=policy_opt_params.T,
        #                                 sample_size=50,
        #                                 known_actions=False,
        #                                 is_plot=True #(kwargs['mode'] == 'local')
        #                                 )
        # rllab_logger.record_tabular('eval_model_time', time.time() - current_time)
        # current_time = time.time()

        # 1-SVG update
        if kwargs["algo_name"] == "svg":
            svg_args = dict(rollout_data=rollout_data,
                            policy_gradient=policy_gradient,
                            model_gradient=model_gradient,
                            cost_gradient=cost_gradient,
                            lr=policy_opt_params.learning_rate,
                            theta_vars=theta_vars)
        else:
            svg_args = {}

        # Policy Optimization
        logger.info('\n\nPolicy Optimization:  %d' % (count))
        # Get weights before update
        assert len(get_variables(scope=policy_scope, filter='weights')) == 0
        old_policy_weights = sess.run(flat_policy_weights)
        policy_learning_logs = optimize_policy(env,
                                               policy_opt_op,
                                               policy_opt_params,
                                               policy_costs,
                                               set_stochastic,
                                               n_saturates,
                                               training_policy_cost,  # for training
                                               policy_adam_init,
                                               policy_in,  # for oracle
                                               policy_out,  # for oracle
                                               cost_np_vec,  # for oracle
                                               sess,
                                               policy_training_init,
                                               policy_validation_init,
                                               policy_validation_reset_init,
                                               policy_grads_and_vars,
                                               verbose,
                                               policy_saver, log_dir, logger,
                                               dynamics_data['training_dynamics'],
                                               svg_args,
                                               policy_summary,
                                               train_writer,
                                               val_writer,
                                               **kwargs)
        new_policy_weights = sess.run(flat_policy_weights)
        rllab_logger.record_tabular('policy_opt_time', time.time() - current_time)
        current_time = time.time()

        ######################
        ## Save every sweep ##
        ######################
        if (np.abs(new_policy_weights - old_policy_weights) > 0).any():
            diff_weights = np.abs(new_policy_weights - old_policy_weights)
        if not diff_weights is None:
            rllab_logger.record_tabular('MaxPolicyWeightDiff', np.amax(diff_weights))
            rllab_logger.record_tabular('MinPolicyWeightDiff', np.amin(diff_weights))
            rllab_logger.record_tabular('AvgPolicyWeightDiff', np.mean(diff_weights))
        else:
            rllab_logger.record_tabular('MaxPolicyWeightDiff', 0)
            rllab_logger.record_tabular('MinPolicyWeightDiff', 0)
            rllab_logger.record_tabular('AvgPolicyWeightDiff', 0)

        # # Save errors
        # error_log_path = os.path.join(log_dir, 'errors_%d' % count)
        # with open(error_log_path, 'wb') as f:
        #     pickle.dump(errors, f)
        #     logger.info('\tSave errors to %s.' % error_log_path)

        policy_learning_log_path = os.path.join(log_dir, 'policy_learning_sweep_%d.pkl' % (count))
        with open(policy_learning_log_path, 'wb') as f:
            pickle.dump(policy_learning_logs, f)
            logger.info('\tSave policy training and test results to %s.' % policy_learning_log_path)

        dynamics_learning_log_path = os.path.join(log_dir, 'dynamics_learning_sweep_%d.pkl' % (count))
        with open(dynamics_learning_log_path, 'wb') as f:
            pickle.dump(dynamics_learning_logs, f)
            logger.info('\tSave dynamics training and test results to %s.' % dynamics_learning_log_path)

        # Save meta for the final model.
        ckpt_path = os.path.join(log_dir, 'policy-and-models-%d.ckpt' % count)
        saver.save(sess, ckpt_path)

        rllab_logger.record_tabular('save_and_log_time', time.time() - current_time)
        rllab_logger.record_tabular('Time', time.time() - start_time)
        rllab_logger.record_tabular('ItrTime', time.time() - itr_start_time)
        rllab_logger.dump_tabular()

        ######################
        # Prepare next sweep #
        ######################
        ask_to_run_more = 'mode' in variant and variant['mode'] == 'local'
        count += 1
        if count > sweep_iters:
            if ask_to_run_more:
                response = input('Do you want to run 5 more?\n')
                if response in {'yes', 'y', 'Y', 'Yes'}:
                    sweep_iters += 5
                elif str.isdigit(response):
                    sweep_iters += int(response)
                else:
                    break
            else:
                break
    # Save meta for the final model.
    ckpt_path = os.path.join(log_dir, 'policy-and-models-final.ckpt')
    saver.save(sess, ckpt_path)
    return sess


def collect_data(env,
                 sample_size,
                 dynamics_data,
                 dynamics_validation,
                 policy_in,
                 policy_out,
                 cost_np,  # for debugging
                 sess,
                 diff_weights,
                 saver,
                 log_dir,
                 rollout_params,
                 count,
                 logger,
                 is_env_done,
                 input_rms,
                 output_rms):
    if sample_size == 0:
        return []
    Os, As, Rs, info = sample_trajectories(env,
                                           policy_in,
                                           policy_out,
                                           rollout_params.exploration,
                                           sample_size,
                                           saver,
                                           diff_weights,
                                           log_dir,
                                           logger,
                                           is_monitored=rollout_params.is_monitored,
                                           monitorpath=rollout_params.monitorpath + '/iter_%d' % count,
                                           sess=sess,
                                           max_timestep=rollout_params.max_timestep,
                                           render_every=rollout_params.render_every,
                                           cost_np=cost_np,
                                           is_done=is_env_done)
    # Get all x and y pairs.
    x_all = []
    y_all = []
    # TODO: for svg only can remove if no longer needed.
    rollout_data = []
    for i, o in enumerate(Os):
        a = As[i]
        triplets = []
        for t in range(len(o) - 1):
            x_all.append(np.concatenate([o[t], a[t]]))
            y_all.append(o[t + 1])
            triplets.append((o[t], a[t], o[t + 1]))
        rollout_data.append(triplets)
    x_all = np.array(x_all)
    y_all = np.array(y_all)

    # Save data
    with open(os.path.join(log_dir, 'new_rollouts_%d.pkl' % count), 'wb') as f:
        pickle.dump((x_all, y_all), f)
        logger.info('Saved new rollouts data')
    indices = list(range(len(x_all)))
    # Splitting data into collections.
    if rollout_params.splitting_mode == "trajectory":
        pass
    elif rollout_params.splitting_mode == "triplet":
        # Randomly permute data before added
        np.random.shuffle(indices)
    else:
        assert (False)
    cur_i = 0
    assert len(x_all) >= sample_size
    total_sample_size = len(x_all)
    for scope in dynamics_data.keys():
        if rollout_params.use_same_dataset:
            _n = round(rollout_params.split_ratio * total_sample_size)
            dynamics_validation[scope].add_data(x_all[indices[:_n], :],
                                                y_all[indices[:_n], :])
            dynamics_data[scope].add_data(x_all[indices[_n:], :],
                                          y_all[indices[_n:], :])
            cur_i = len(indices)
            # Update running mean and std.
            input_rms.update(x_all[indices[_n:], :])
            output_rms.update(y_all[indices[_n:], :] - x_all[indices[_n:], :y_all.shape[1]])
        else:
            _n = int(rollout_params.split_ratio * total_sample_size /
                     len(dynamics_data.keys()))
            dynamics_validation[scope].add_data(x_all[indices[cur_i:cur_i + _n], :],
                                                y_all[indices[cur_i:cur_i + _n], :])
            cur_i += _n
            _m = int(total_sample_size / len(dynamics_data.keys()) - _n)
            dynamics_data[scope].add_data(
                x_all[indices[cur_i:cur_i + _m], :],
                y_all[indices[cur_i:cur_i + _m], :]
            )
            cur_i += _m
        logger.info('current %s dynamics_data size : %d' %
                    (scope, dynamics_data[scope].get_num_data()))
        logger.info('current %s dynamics_validation size : %d' %
                    (scope, dynamics_validation[scope].get_num_data()))
    assert (cur_i == total_sample_size)
    # if rollout_params.splitting_mode == "trajectory":
    #     for i in range(total_sample_size):
    #         if i % rollout_params.max_timestep != rollout_params.max_timestep - 1:
    #             assert (x_all[indices[i] + 1, 0] == y_all[indices[i], 0])
    return rollout_data


# Baseline - predict using previous state.
def compute_baseline_loss(x_batch, y_batch, n_models):
    return n_models * np.sum(
        np.square(
            y_batch - x_batch[:, :y_batch.shape[1]]
        )
    ) / y_batch.shape[0]

def assign_lr(lr_up, lr_ph, value_to_assign, sess):
    sess.run(lr_up, feed_dict={lr_ph: value_to_assign})

def recover_weights(savers, recover_indices, scope, logger, log_dir, sess):
    logger.info(
        'Recover back best weights for each model.'
    )
    n_models = len(savers[scope])
    for i in range(n_models):
        savers[scope][i].restore(sess, os.path.join(log_dir, '%s_%d.ckpt' % (scope, i)))
        logger.debug('Restore %s %d from iter %d' % (scope, i, recover_indices[i]))


def optimize_models(dynamics_data,
                    dynamics_validation,
                    dynamics_opt_op,
                    dynamics_opt_params,
                    dynamics_adam_init,
                    dynamics_loss,  # for debugging
                    dynamics_losses,
                    prediction_loss,
                    dynamics_in_full,
                    y_training_full,
                    sess,
                    verbose,
                    savers,
                    log_dir,
                    logger,
                    n_models,
                    dynamics_initializer,
                    dynamics_summaries,
                    train_writer,
                    val_writer,
                    lr_up,
                    lr_ph,
                    reinitialize):
    ## Re-initialize Adam parameters.
    lr = dynamics_opt_params.learning_rate
    if reinitialize:
        assign_lr(lr_up, lr_ph, lr["scratch"], sess)
        sess.run([dynamics_adam_init, dynamics_initializer])
        logger.info('Reinitialize dynamics models & '
                    'Adam Optimizer & '
                    'update the learning rate to %f.' %
                    lr["scratch"])
    else:
        assign_lr(lr_up, lr_ph, lr["refine"], sess)
        sess.run(dynamics_adam_init)
        logger.info('Reinitialize Adam Optimizer & '
                    'update the learning rate to %f.' %
                    lr["refine"])


    batch_size = dynamics_opt_params.batch_size
    training_losses = dict([(scope, []) for scope in dynamics_data.keys()])
    validation_losses = dict([(scope, []) for scope in dynamics_data.keys()])
    best_js = dict([(scope, 0) for scope in dynamics_data.keys()])
    for scope in dynamics_data.keys():
        ## Save all weights before training
        for i in range(n_models):
            savers[scope][i].save(sess,
                                  os.path.join(log_dir, '%s_%d.ckpt' % (scope, i)),
                                  write_meta_graph=False)
        logger.info('Saved all initial weights in %s.' % scope)

        ## Don't recompute validation input
        x_batch_val = np.tile(dynamics_validation[scope].x, n_models)
        y_batch_val = np.tile(dynamics_validation[scope].y, n_models)

        logger.info('Model %s' % scope)
        ## Initialize min validation loss
        min_sum_validation_loss, min_validation_losses = sess.run(
            [dynamics_loss[scope], dynamics_losses[scope]],
            feed_dict={
                dynamics_in_full: x_batch_val,
                y_training_full: y_batch_val
            }
        )
        min_validation_losses = np.array(min_validation_losses)
        log_losses(logger,
                   0,
                   min_sum_validation_loss,
                   min_validation_losses)
        recover_indices = np.zeros(n_models)
        refine_idx = -1

        iter_const = dynamics_data[scope].n_data / batch_size
        max_iters = int(dynamics_opt_params.max_passes * iter_const)
        log_every = int(dynamics_opt_params.log_every * iter_const)
        num_iters_threshold = int(dynamics_opt_params.num_passes_threshold * iter_const)
        for j in range(1, max_iters + 1):
            # Training
            if dynamics_opt_params.sample_mode == 'next_batch':
                x_batch, y_batch = dynamics_data[scope].get_next_batch(batch_size * n_models, is_shuffled=False)
            else:
                assert dynamics_opt_params.sample_mode == 'random'
                x_batch, y_batch = dynamics_data[scope].sample(batch_size * n_models)

            _, training_loss = sess.run([dynamics_opt_op[scope], dynamics_loss[scope]],
                                        feed_dict={
                                            dynamics_in_full: np.reshape(x_batch, (batch_size, -1)),
                                            y_training_full: np.reshape(y_batch, (batch_size, -1))
                                        })

            # Validation and Logging
            if j % log_every == 0:
                training_losses[scope].append(training_loss)
                ### TODO: Now only get the summary of the first model in each scope.
                validation_loss, _validation_losses, _prediction_loss = sess.run(
                    [dynamics_loss[scope],
                     dynamics_losses[scope],
                     prediction_loss[scope]],
                    feed_dict={dynamics_in_full: x_batch_val,
                               y_training_full: y_batch_val}
                )
                _validation_losses = np.array(_validation_losses)
                validation_losses[scope].append(validation_loss)

                log_losses(logger,
                           j,
                           validation_loss,
                           _validation_losses,
                           training_loss=training_loss,
                           prediction_loss=_prediction_loss)

                # Save the model when each validation cost reaches minimum.
                # Save best_j for best total loss.
                if min_sum_validation_loss > validation_loss:
                    min_sum_validation_loss = validation_loss
                    best_js[scope] = j
                to_update = min_validation_losses > _validation_losses
                min_validation_losses[to_update] = _validation_losses[to_update]
                for i, is_updated in enumerate(to_update):
                    if is_updated:
                        savers[scope][i].save(sess,
                                              os.path.join(log_dir, '%s_%d.ckpt' % (scope, i)),
                                              write_meta_graph=False)
                        if verbose:
                            logger.debug('Saved %s %d' % (scope, i))
                        recover_indices[i] = j
                logger.info('Saved %d models.' % np.sum(to_update))
                # # Break if the cost is going up, and recover to the ckpt.
                # if dynamics_opt_params.stop_critereon(min_validation_loss, validation_loss) or \
                #     j == dynamics_opt_params.max_iters:
                #     logger.info('\tStop at iter %d.' % (j))
                #     logger.info('\t\tAfter update validation loss increased from min %.3f to %.3f' % (
                #         min_validation_loss, validation_loss))
                #
                #     logger.info(
                #         '\t\tRecover back to iter %d from %s.' %
                #         (best_j[scope], tf.train.latest_checkpoint(log_dir)) <-- TO FIX
                #     )
                #     saver.restore(sess, tf.train.latest_checkpoint(log_dir)) <-- TO FIX
                #     break
                if j - max(np.amax(recover_indices), refine_idx) \
                        >= num_iters_threshold:
                    if reinitialize and refine_idx < 0 and lr["scratch"] > lr["refine"]:
                        recover_weights(savers, recover_indices, scope, logger, log_dir, sess)
                        logger.info('\nFinished training with lr = %f. Now, reduce to %f\n' %
                                    (lr["scratch"], lr["refine"]))
                        assign_lr(lr_up, lr_ph, lr["refine"], sess)
                        refine_idx = j
                        continue
                    break

        ## After updates go to minimum
        recover_weights(savers, recover_indices, scope, logger, log_dir, sess)

        assert ('dynamics' in scope)
        rllab_logger.record_tabular('# model updates', j)
        rllab_logger.record_tabular('%s_min_sum_validation_loss' % scope,
                                    min_sum_validation_loss)
        # Save summary
        if TF_SUMMARY:
            summary = sess.run(
                dynamics_summaries[scope][0],
                feed_dict={dynamics_in_full: x_batch_val[:5],
                           y_training_full: y_batch_val[:5]}
            )
            val_writer.add_summary(summary, j)
    return {'dynamics_opt_params': get_pickeable(dynamics_opt_params),
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'best_index': best_js}


def log_losses(logger,
               index,
               validation_loss,
               validation_losses,
               first_n=5,
               training_loss=None,
               compute_baseline=False,
               **kwargs):
    assert isinstance(validation_losses, np.ndarray)
    msg = np.array_str(validation_losses[:first_n], max_line_width=50, precision=2)
    if index == 0:
        logger.info('iter 0 (no update yet)')

    else:
        logger.info('iter %d' % index)
        logger.info('\ttraining_loss \t: %f' % training_loss)
    logger.info('\tvalidation_loss \t: %f' % validation_loss)
    logger.info('\tvalidation_losses \t: %s' % msg)
    if 'prediction_loss' in kwargs:
        logger.info('\tprediction_loss \t: %s' % kwargs['prediction_loss'])
    if compute_baseline:
        logger.info(
            '\tbaseline_validation_loss : %.3f' %
            compute_baseline_loss(kwargs['x_batch'],
                                  kwargs['y_batch'],
                                  kwargs['n_models']))


def optimize_policy(env,
                    policy_opt_op,
                    policy_opt_params,
                    policy_costs,
                    set_stochastic,
                    sym_n_saturates,
                    training_policy_cost,
                    policy_adam_init,
                    policy_in,  # for oracle
                    policy_out,  # for oracle
                    cost_np_vec,  # for oracle
                    sess,
                    policy_training_init,
                    policy_validation_init,
                    policy_validation_reset_init,
                    policy_grads_and_vars,  # for debugging
                    verbose,
                    saver, log_dir, logger,
                    training_dynamics_data,
                    svg_args,
                    policy_summary,
                    train_writer,
                    val_writer,
                    **kwargs):
    mode_order = ['real',
                  'estimated',
                  'second_estimated',
                  'third_estimated',
                  'fourth_estimated',
                  ]
    scope2mode = {'training_dynamics': 'estimated',
                  'validation_dynamics': 'second_estimated',
                  'second_validation_dynamics': 'third_estimated',
                  'third_validation_dynamics': 'fourth_estimated'}
    batch_size = policy_opt_params.batch_size

    ### Re-initialize Adam parameters.
    if 'reset_opt' in kwargs:
        sess.run([policy_adam_init, kwargs['reset_opt']])
        logger.info('Reinitialize Adam Optimizer and init_std.')
    else:
        sess.run(policy_adam_init)
        logger.info('Reinitialize Adam Optimizer')

    ### Save all weights before updates
    saver.save(sess,
               os.path.join(log_dir, 'policy.ckpt'),
               write_meta_graph=False)
    logger.info('Saved policy weights before update.')

    ### Estimated cost for T steps prediction cost.
    training_costs = []
    estimated_validation_costs = {}
    real_validation_costs = []
    trpo_mean_costs = []
    ### Current minimums contain real and estimated validation costs
    '''
    Here is an example of min_validation costs
    {
        'real':1.0,
        'estimated':4.4,
        'second_estimated':[1.1, 3.2, 5., 0.3, 1.4 ], (length = n_models)
        'sum': 2.0 (avg of second_estimated)
    }
    '''
    min_validation_costs = {}
    min_validation_costs['real'] = evaluate_fixed_init_trajectories(
        env,
        policy_in,
        policy_out,
        policy_validation_reset_init,
        cost_np_vec, sess,
        max_timestep=policy_opt_params.oracle_maxtimestep,
        gamma=policy_opt_params.gamma
    )
    min_validation_costs['trpo_mean'] = np.inf
    for scope in policy_costs.keys():
        mode = scope2mode[scope]
        min_validation_costs[mode] = np.array(sess.run(
            policy_costs[scope],
            feed_dict={policy_training_init: policy_validation_init}
        ))

    best_index = 0
    real_current_validation_cost = min_validation_costs['real']
    logger.info('iter 0 (no update yet)')
    log_dictionary(mode_order, min_validation_costs, min_validation_costs, logger)

    candidates = {}
    for j in range(1, policy_opt_params.max_iters + 1):
        ### Train policy
        if kwargs['algo_name'] == 'trpo' or kwargs['algo_name']=='vpg':
            algo = kwargs['rllab_algo']
            algo.start_worker()
            with rllab_logger.prefix('itr #%d | ' % j):
                paths = algo.obtain_samples(j)
                samples_data = algo.process_samples(j, paths)
                algo.optimize_policy(j, samples_data)
            training_cost = 0
        elif kwargs['algo_name'] == 'bptt':
            # Sample new initial states for computing gradients.
            # x_batch = training_dynamics_data.sample(batch_size)[1]
            x_batch = np.array([env.reset() for i in range(batch_size)])
            _, training_cost = sess.run([policy_opt_op, training_policy_cost],
                                        feed_dict={policy_training_init: x_batch})
            training_cost = np.squeeze(training_cost)
        elif kwargs['algo_name'] == 'bptt-stochastic':
            x_batch = np.array([env.reset() for i in range(batch_size)])
            # TODO: remove sym_n_saturates.
            _, _, training_cost, n_saturates = sess.run([set_stochastic[1],
                                                         policy_opt_op,
                                                         training_policy_cost,
                                                         sym_n_saturates],
                                        feed_dict={policy_training_init: x_batch})
            training_cost = np.squeeze(training_cost)
        elif kwargs['algo_name'] == 'l-bfgs':
            train_step = tf.get_collection("l-bfgs_step")[0]
            x_batch = np.array([env.reset() for i in range(batch_size)])
            train_step.minimize(sess, feed_dict={policy_training_init: x_batch})
            training_cost = sess.run(training_policy_cost,
                                     feed_dict={policy_training_init: x_batch})
        else:
            assert kwargs['algo_name'] == 'svg'
            grads = svg_update(**svg_args, sess=sess)
            training_cost = 0

        ### Evaluate in learned and real dynamics
        if j % policy_opt_params.log_every == 0:
            # Set stochasticity back to 0.0.
            if kwargs['algo_name'] == 'bptt-stochastic':
                logger.debug('n_saturates: {}'.format(n_saturates[:5]))
                entropy = 1./2.*np.sum(np.log(2*np.pi*np.e) + sess.run(kwargs['logstd']))
                logger.debug('Entropy: %.3f' % entropy)
                sess.run(set_stochastic[0])

            # Compute TRPO mean if neccessary.
            if kwargs["algo_name"] == "trpo":
                algo = kwargs['rllab_algo']
                if policy_opt_params.mode == 'trpo_mean':
                    determ_paths = algo.obtain_samples(j, determ=True)
                    traj_costs = []
                    for determ_path in determ_paths:
                        traj_costs.append(- np.sum(determ_path["rewards"]))
                    candidates['trpo_mean'] = np.mean(traj_costs)
                    if 'trpo_mean' != mode_order[1]:
                        mode_order.insert(1, 'trpo_mean')
                else:
                    candidates['trpo_mean'] = 0.0
                trpo_mean_costs.append(candidates['trpo_mean'])
            else:
                candidates['trpo_mean'] = 0.0

            ## Training cost
            training_costs.append(training_cost)
            ## Estimated cost
            for scope in policy_costs.keys():
                mode = scope2mode[scope]
                estimated_valid_cost = sess.run(policy_costs[scope],
                    feed_dict={policy_training_init: policy_validation_init}
                )
                estimated_valid_cost = np.array(estimated_valid_cost)

                if mode in estimated_validation_costs:
                    estimated_validation_costs[mode].append(np.mean(estimated_valid_cost))
                else:
                    estimated_validation_costs[mode] = [np.mean(estimated_valid_cost)]
                candidates[mode] = estimated_valid_cost

            ## Real cost
            real_validation_cost = evaluate_fixed_init_trajectories(
                env,
                policy_in,
                policy_out,
                policy_validation_reset_init,
                cost_np_vec, sess,
                max_timestep=policy_opt_params.oracle_maxtimestep,
                gamma=policy_opt_params.gamma
            )
            real_validation_costs.append(real_validation_cost)
            candidates['real'] = real_validation_cost

            ## Logging
            logger.info('iter %d' % j)
            logger.info('\ttraining_cost:\t%.3f' % training_cost)
            log_dictionary(mode_order, candidates, min_validation_costs, logger)

            if False: #verbose and kwargs['algo_name'] == 'bptt':  # TODO: debug this
                _policy_grads = sess.run([gv[0] for gv in policy_grads_and_vars],
                                         feed_dict={policy_training_init: x_batch})
                logger.debug('\t, policy_grads_max: {}'.format(
                    np.array([np.max(np.abs(grad)) for grad in _policy_grads])))
                logger.debug('\t, policy_grads_norm: {}'.format(
                    np.array([np.linalg.norm(grad) for grad in _policy_grads])))
                logger.debug('\t, policy_grads_avg: {}'.format(
                    np.array([np.mean(np.abs(grad)) for grad in _policy_grads])))
                logger.debug('\t, policy_grads_min: {}'.format(
                    np.array([np.min(np.abs(grad)) for grad in _policy_grads])))

            if kwargs['algo_name'] == 'svg':
                is_broken = True
                break

            ## Not done - we update.
            ## Done - we go back and reduce std.
            if not is_done(policy_opt_params, min_validation_costs, candidates, logger):
                best_index = j
                real_current_validation_cost = candidates['real']
                # Save
                logger.info('\tSaving policy')
                saver.save(sess,
                           os.path.join(log_dir, 'policy.ckpt'),
                           write_meta_graph=False)
                ## Update - only when we save the update.
                update_stats(min_validation_costs, candidates, policy_opt_params.whole)
            ## Stop
            # If the number of consecutive dones is greater than the threshold
            if j - best_index >= policy_opt_params.num_iters_threshold:
                break

    log_and_restore(sess,
                    log_dir,
                    j,
                    min_validation_costs,
                    candidates,
                    logger,
                    saver,
                    mode_order,
                    best_index,
                    policy_opt_params.mode)
    if policy_opt_params.mode == 'one_model' or policy_opt_params.mode == 'no_early':
        min_val_cost = min_validation_costs['estimated'][0]
    else:
        min_val_cost = np.mean(min_validation_costs[policy_opt_params.mode])

    for key in min_validation_costs.keys():
        rllab_logger.record_tabular('%s_policy_mean_min_validation_cost' % key,
                                    np.mean(min_validation_costs[key]))
    rllab_logger.record_tabular('real_current_validation_cost', real_current_validation_cost)
    rllab_logger.record_tabular('# policy updates', best_index)

    # Save summary
    if TF_SUMMARY:
        summary = sess.run(policy_summary,
                           feed_dict={policy_training_init: policy_validation_init}
                           )
        val_writer.add_summary(summary, best_index)

    return {'real_validation_costs': real_validation_costs,
            'training_costs': training_costs,
            'estimated_validation_costs': estimated_validation_costs,
            'policy_opt_params': get_pickeable(policy_opt_params),
            'best_index': best_index,
            'best_cost': min_val_cost,
            'trpo_mean_costs': trpo_mean_costs
            }


def is_done(policy_opt_params, min_validation_costs, candidates, logger):
    '''
    When mode == 'real', we stop immediately if the cost increases.
    When mode has 'estimated', we stop if one of the bucket has
    the ratio of increasing costs exceeds a certain threshold

    If one of the candidates are worst than min_validation_costs we stop.
    '''
    mode = policy_opt_params.mode
    if mode == 'real':
        # TODO:relax the constraint.
        # return policy_opt_params.stop_critereon(min_validation_costs[mode],
        #                                         candidates[mode])
        return min_validation_costs['real'] < candidates['real']
    elif mode == 'trpo_mean':
        assert 'trpo_mean' in min_validation_costs.keys()
        return min_validation_costs['trpo_mean'] < candidates['trpo_mean']
    elif mode == 'one_model':
        return min_validation_costs['estimated'][0] < candidates['estimated'][0]
    elif mode == 'no_early':
        return False
    else:
        assert 'estimated' in mode
        for _mode in min_validation_costs.keys():
            if 'estimated' in _mode and \
                    policy_opt_params.stop_critereon(
                        min_validation_costs[_mode],  # Input an array
                        candidates[_mode],
                        mode='vector'
                    ):
                logger.info('\t### %s tells us to stop.' % _mode)
                return True
        return False


def log_and_restore(sess,
                    log_dir,
                    index,
                    min_validation_costs,
                    candidates,
                    logger,
                    saver,
                    mode_order,
                    best_index,
                    mode):
    '''
    Recover back to the last saved checkpoint.
    '''
    logger.info('Stop at iter %d. Recover to iter %d.' % (index, best_index))
    for _mode in mode_order:
        if _mode in min_validation_costs:
            _msg = '\t%.5s validation cost \t %.3f  -->  %.3f' % (
                _mode,
                np.mean(candidates[_mode]),
                np.mean(min_validation_costs[_mode])
            )
            if _mode == mode:
                _msg += ' ***'
            logger.info(
                _msg
            )
    saver.restore(sess, os.path.join(log_dir, 'policy.ckpt'))


def update_stats(min_validation_costs, candidates, whole=False):
    '''
    :param whole: if this is true, the whole candidates are set to
    min_validation_costs. This means information in
    the min validation costs are consistent (from the same iteration).
    If this is false, we keep the min of each individual we have seen so far.
    '''
    for _mode in min_validation_costs.keys():
        costs = min_validation_costs[_mode]
        if hasattr(costs, '__iter__') and len(costs) != 1:
            if whole:
                min_validation_costs[_mode][:] = candidates[_mode][:]
            else:
                to_update = costs > candidates[_mode]
                min_validation_costs[_mode][to_update] = candidates[_mode][to_update]
        elif whole or costs > candidates[_mode]:
            min_validation_costs[_mode] = candidates[_mode]

def log_dictionary(mode_order, validation_costs, min_validation_costs, logger, first_n=5):
    for mode in mode_order:
        if mode in validation_costs:
            costs = validation_costs[mode]
            if hasattr(costs, '__iter__'):
                assert 'estimated' in mode
                msg = np.array_str(costs[:first_n], max_line_width=50, precision=2)
                logger.info('\t%.5s_validation_cost:\t%s' %
                            (mode, msg))
                logger.info('\t\tavg=%.2f, increase_ratio=%.2f' % (
                    np.mean(costs),
                    np.mean(costs > min_validation_costs[mode])
                ))
                logger.info('\t\tmode=%.2f, std=%.2f, min=%.2f, max=%.2f' %
                            (np.median(costs),
                             np.std(costs),
                             np.min(costs),
                             np.max(costs)))
            else:
                logger.info('\t%.5s_validation_cost:\t%.3f' %
                            (mode, costs))
