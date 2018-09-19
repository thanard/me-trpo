import tensorflow as tf
import numpy as np
import joblib
import json
import os
import tensorflow.contrib.layers as layers

from model_based_rl import train_models
from env_helpers import get_env
from utils import get_scope_variable, get_session, data_summaries,\
    variable_summaries, stop_critereon, set_global_seeds
from shutil import copyfile, rmtree
from namedtuples import Dynamics_opt_params, Policy_opt_params, Rollout_params
import rllab.config as config
import rllab.misc.logger as logger

def train(variant):
    set_global_seeds(variant['seed'])

    if variant['mode'] == 'local':
        import colored_traceback.always
    '''
    Set-up folder and files
    '''
    snapshot_dir = logger.get_snapshot_dir()
    working_dir = config.PROJECT_PATH
    param_path = os.path.join(working_dir,'params/params.json')
    # copyfile(param_path, os.path.join(snapshot_dir,'params.json'))

    try:
        '''
        Save parameters
        '''
        if 'params' in variant:
            logger.log('Load params from variant.')
            params = variant['params']
        else:
            logger.log('Load params from file.')
            with open(param_path ,'r') as f:
                params = json.load(f)

        # Save to snapshot dir
        new_param_path = os.path.join(snapshot_dir,'params.json')
        with open(new_param_path,'w') as f:
            json.dump(params, f, sort_keys=True, indent=4, separators=(',', ': '))

        # TODO: can use variant to modify here.
        dynamics_opt_params = params['dynamics_opt_params']
        dynamics_opt_params['stop_critereon'] = stop_critereon(
            threshold=dynamics_opt_params['stop_critereon']['threshold'],
            offset=dynamics_opt_params['stop_critereon']['offset']
            )
        dynamics_opt_params = Dynamics_opt_params(**dynamics_opt_params)

        policy_opt_params = params['policy_opt_params']
        policy_opt_params['stop_critereon'] = stop_critereon(
            threshold=policy_opt_params['stop_critereon']['threshold'],
            offset=policy_opt_params['stop_critereon']['offset'],
            percent_models_threshold=policy_opt_params['stop_critereon']['percent_models_threshold']
            )
        policy_opt_params = Policy_opt_params(**policy_opt_params)

        rollout_params = params['rollout_params']
        rollout_params['monitorpath'] = os.path.join(snapshot_dir, 'videos')
        rollout_params = Rollout_params(**rollout_params)

        assert params['rollout_params']['max_timestep'] == \
               params['policy_opt_params']['oracle_maxtimestep'] == \
               params['policy_opt_params']['T']

        '''
        Policy model
        '''
        def build_policy_from_rllab(scope_name='training_policy'):
            '''
            Return both rllab policy and policy model function.
            '''
            sess = tf.get_default_session()

            ### Initialize training_policy to copy from policy
            from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
            output_nonlinearity = eval(params['policy']['output_nonlinearity'])

            training_policy = GaussianMLPPolicy(
                name=scope_name,
                env_spec=env.spec,
                hidden_sizes=params['policy']['hidden_layers'],
                init_std=policy_opt_params.trpo['init_std'],
                output_nonlinearity=output_nonlinearity
            )
            training_policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_policy')
            sess.run([tf.variables_initializer(training_policy_vars)])

            ### Compute policy model function using the same weights.
            training_layers = training_policy._mean_network.layers
            def policy_model(x, stochastic=0.0, collect_summary=False):
                assert (training_layers[0].shape[1] == x.shape[1])
                h = x
                for i, layer in enumerate(training_layers[1:]):
                    w = layer.W
                    b = layer.b
                    pre_h = tf.matmul(h, w) + b
                    h = layer.nonlinearity(pre_h, name='policy_out')
                    if collect_summary:
                        with tf.name_scope(scope_name + '/observation'):
                            variable_summaries(x)
                        with tf.name_scope(scope_name + '/layer%d' % i):
                            with tf.name_scope('weights'):
                                variable_summaries(w)
                            with tf.name_scope('biases'):
                                variable_summaries(b)
                            with tf.name_scope('Wx_plus_b'):
                                tf.summary.histogram('pre_activations', pre_h)
                            tf.summary.histogram('activations', h)
                std = training_policy._l_std_param.param
                h += stochastic * tf.random_normal(shape=(tf.shape(x)[0], n_actions)) * tf.exp(std)
                return h
            return training_policy, policy_model

        '''
        Dynamics model
        '''
        def get_value(key, dict):
            return key in dict and dict[key]
        def prepare_input(xgu,
                          xgu_norm,
                          scope_name,
                          variable_name,
                          collect_summary,
                          prediction_type):
            name_scope = '%s/%s' % (scope_name, variable_name)
            assert n_states > 1 and n_actions > 1 \
                   and xgu.shape[1] == n_states + n_actions + n_goals
            xu = tf.concat([xgu[:, :n_states], xgu[:, n_states + n_goals:]], axis=1)
            xu_norm = tf.concat([xgu_norm[:, :n_states], xgu_norm[:, n_states+n_goals:]], axis=1)
            # Collect data summaries
            if collect_summary:
                with tf.name_scope(name_scope + '/inputs'):
                    with tf.name_scope('states'):
                        data_summaries(xgu[:, :n_states])
                    with tf.name_scope('goals'):
                        data_summaries(xgu[:, n_states:n_states + n_goals])
                    with tf.name_scope('actions'):
                        data_summaries(xgu[:, n_states + n_goals:])
            # Ignore xy in the current state.
            if get_value('ignore_xy_input', params['dynamics_model']):
                n_inputs = n_states + n_actions - 2
                nn_input = xu_norm[:, 2:]
            elif get_value('ignore_x_input', params['dynamics_model']):
                n_inputs = n_states + n_actions - 1
                nn_input = xu_norm[:, 1:]
            else:
                n_inputs = n_states + n_actions
                nn_input = xu_norm
            hidden_layers = list(params['dynamics_model']['hidden_layers'])
            nonlinearity = [eval(_x) for _x in params['dynamics_model']['nonlinearity']]
            assert (len(nonlinearity) == len(hidden_layers))
            # Verify if the input type is valid.
            if prediction_type == 'state_change' or \
                            prediction_type == 'state_change_goal':
                n_outputs = n_states
            else:
                assert prediction_type == 'second_derivative' or \
                       prediction_type == 'second_derivative_goal'
                n_outputs = int(n_states / 2)
            nonlinearity.append(tf.identity)
            hidden_layers.append(n_outputs)
            return xu, nn_input, n_inputs, n_outputs, \
                   nonlinearity, hidden_layers

        def build_ff_neural_net(nn_input,
                                n_inputs,
                                hidden_layers,
                                nonlinearity,
                                scope_name,
                                variable_name,
                                collect_summary,
                                logit_weights=None,
                                initializer=layers.xavier_initializer()
                                ):
            assert len(hidden_layers) == len(nonlinearity)
            name_scope = '%s/%s' % (scope_name, variable_name)
            h = nn_input
            n_hiddens = n_inputs
            n_hiddens_next = hidden_layers[0]
            for i in range(len(hidden_layers)):
                w = get_scope_variable(scope_name,
                                       "%s/layer%d/weights" % (variable_name, i),
                                       shape=(n_hiddens, n_hiddens_next),
                                       initializer=initializer)
                b = get_scope_variable(scope_name,
                                       "%s/layer%d/biases" % (variable_name, i),
                                       shape=(n_hiddens_next),
                                       initializer=initializer)
                if collect_summary:
                    with tf.name_scope(name_scope + '/layer%d' % i):
                        with tf.name_scope('weights'):
                            variable_summaries(w)
                        with tf.name_scope('biases'):
                            variable_summaries(b)
                        with tf.name_scope('Wx_plus_b'):
                            pre_h = tf.matmul(h, w) + b
                            tf.summary.histogram('pre_activations', pre_h)
                        h = nonlinearity[i](pre_h, name='activation')
                        tf.summary.histogram('activations', h)
                else:
                    pre_h = tf.matmul(h, w) + b
                    h = nonlinearity[i](pre_h, name='activation')
                n_hiddens = hidden_layers[i]
                if i+1 < len(hidden_layers):
                    n_hiddens_next = hidden_layers[i+1]
                if logit_weights is not None and i == len(hidden_layers)-2:
                    h *= logit_weights
            return h

        def build_dynamics_model(n_states, n_actions, n_goals, dt=None, input_rms=None, diff_rms=None):
            prediction_type = params['dynamics_model']['prediction_type']
            def dynamics_model(xgu, scope_name, variable_name, collect_summary=False):
                '''
                :param xu: contains states, goals, actions
                :param scope_name:
                :param variable_name:
                :param dt:
                :return:
                '''
                xu, nn_input, n_inputs, n_outputs, nonlinearity, hidden_layers = \
                    prepare_input(xgu,
                                  (xgu - input_rms.mean)/input_rms.std,
                                  scope_name,
                                  variable_name,
                                  collect_summary,
                                  prediction_type)

                if "use_logit_weights" in params["dynamics_model"] and params["dynamics_model"]["use_logit_weights"]:
                    logit_weights = build_ff_neural_net(nn_input,
                                                        n_inputs,
                                                        hidden_layers[:-1],
                                                        nonlinearity[:-2] + [tf.nn.sigmoid],
                                                        scope_name,
                                                        variable_name + '_sig',
                                                        collect_summary
                                                        )
                else:
                    logit_weights = None
                nn_output = build_ff_neural_net(nn_input,
                                                n_inputs,
                                                hidden_layers,
                                                nonlinearity,
                                                scope_name,
                                                variable_name,
                                                collect_summary,
                                                logit_weights=logit_weights
                                                )

                # predict the delta instead (x_next-x_current)
                if 'state_change' in prediction_type:
                    next_state = tf.add(diff_rms.mean[:n_states] + diff_rms.std[:n_outputs] * nn_output, xu[:, :n_states])
                else:
                    assert 'second_derivative' in prediction_type
                    # We train 'out' to match state_dot_dot
                    # Currently only works for swimmer.
                    qpos = xu[:, :n_outputs] + dt * xu[:, n_outputs:n_states]
                    qvel = xu[:, n_outputs:n_states] + dt * nn_output
                    next_state = tf.concat([qpos, qvel], axis=1)
                if '_goal' in prediction_type:
                    assert n_goals > 1
                    g = xgu[:, n_states:n_states + n_goals]
                    next_state = tf.concat([next_state, g], axis=1)
                return tf.identity(next_state, name='%s/%s/dynamics_out' % (scope_name, variable_name))
            return dynamics_model

        def get_regularizer_loss(scope_name, variable_name):
            if params['dynamics_model']['regularization']['method'] in [None,'']:
                return tf.constant(0.0, dtype=tf.float32)
            constant = params['dynamics_model']['regularization']['constant']
            regularizer = eval(params['dynamics_model']['regularization']['method'])
            hidden_layers = params['dynamics_model']['hidden_layers']
            reg_loss = 0.0
            for i in range(len(hidden_layers)+1):
                w = get_scope_variable(scope_name, "%s/layer%d/weights" % (variable_name, i))
                b = get_scope_variable(scope_name, "%s/layer%d/biases" % (variable_name, i))
                reg_loss += regularizer(w) + regularizer(b)
            return constant * reg_loss

        '''
        Main
        '''
        # with get_session() as sess:
        if variant['mode']=='local':
            sess = get_session(interactive=True, mem_frac=0.1)
        else:
            sess = get_session(interactive=True, mem_frac=1.0, use_gpu=variant['use_gpu'])

        # data = joblib.load(os.path.join(working_dir, params['trpo_path']))
        env = get_env(variant['params']['env'])

        # policy = data['policy']
        training_policy, policy_model = build_policy_from_rllab()
        if hasattr(env._wrapped_env, '_wrapped_env'):
            inner_env = env._wrapped_env._wrapped_env
        else:
            inner_env = env._wrapped_env.env.unwrapped
        n_obs = inner_env.observation_space.shape[0]
        n_actions = inner_env.action_space.shape[0]
        cost_np = inner_env.cost_np
        cost_tf = inner_env.cost_tf
        cost_np_vec = inner_env.cost_np_vec
        if hasattr(inner_env, 'n_goals'):
            n_goals = inner_env.n_goals
            n_states = inner_env.n_states
            assert n_goals+n_states == n_obs
        else:
            n_goals =0
            n_states = n_obs
        dt = None
        # Only necessary for second_derivative
        if hasattr(inner_env, 'model') and hasattr(inner_env, 'frame_skip'):
            dt = inner_env.model.opt.timestep*inner_env.frame_skip
        from running_mean_std import RunningMeanStd
        with tf.variable_scope('input_rms'):
            input_rms = RunningMeanStd(epsilon=0.0, shape=(n_states + n_goals + n_actions))
        with tf.variable_scope('diff_rms'):
            diff_rms = RunningMeanStd(epsilon=0.0, shape=(n_states + n_goals))
        dynamics_model = build_dynamics_model(n_states=n_states,
                                              n_actions=n_actions,
                                              n_goals=n_goals,
                                              dt=dt,
                                              input_rms=input_rms,
                                              diff_rms=diff_rms
                                              )

        kwargs = {}
        kwargs['input_rms'] = input_rms
        kwargs['diff_rms'] = diff_rms
        kwargs['mode'] = variant['mode']
        if params['trpo_init_ratio'] == 0.0 or params['trpo_init_ratio'] is None:
            careful_init = False
        else:
            careful_init = True

        if params['algo'] == 'vpg':
            from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
            from algos.vpg import VPG
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = VPG(
                env=env,
                policy=training_policy,
                baseline=baseline,
                batch_size=policy_opt_params.vpg['batch_size'],
                max_path_length=policy_opt_params.T,
                discount=policy_opt_params.vpg['discount'],
            )
            kwargs['rllab_algo']=algo
            if params["policy_opt_params"]["vpg"]["reset"]:
                kwargs['reset_opt'] = tf.assign(training_policy._l_std_param.param,
                np.log(params["policy_opt_params"]["vpg"]["init_std"])*np.ones(n_actions))
        elif params['algo'] == 'trpo':
            ### Write down baseline and algo
            from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
            from algos.trpo import TRPO
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=training_policy,
                baseline=baseline,
                batch_size=policy_opt_params.trpo['batch_size'],
                max_path_length=policy_opt_params.T,
                discount=policy_opt_params.trpo['discount'],
                step_size=policy_opt_params.trpo['step_size'],
            )
            kwargs['rllab_algo']=algo
            if params["policy_opt_params"]["trpo"]["reset"]:
                kwargs['reset_opt'] = tf.assign(training_policy._l_std_param.param,
                np.log(params["policy_opt_params"]["trpo"]["init_std"])*np.ones(n_actions))
            # if "decay_rate" in params["policy_opt_params"]["trpo"]:
            #     kwargs['trpo_std_decay'] = tf.assign_sub(training_policy._l_std_param.param,
            #     np.log(params["policy_opt_params"]["trpo"]["decay_rate"])*np.ones(n_actions))
        kwargs['inner_env'] = inner_env
        kwargs['algo_name'] = params['algo']
        kwargs['logstd'] = training_policy._l_std_param.param

        policy_in = tf.placeholder(tf.float32, shape=(None, n_obs), name='policy_in_test')
        policy_out = policy_model(policy_in)
        for obs in [np.zeros(n_obs), np.ones(n_obs), np.random.uniform(size=n_obs)]:
            print(sess.run(policy_out, feed_dict={policy_in: obs[None]}))
            print(training_policy.get_action(obs)[1]['mean'])
            # print(policy.get_action(obs)[1]['mean'])

        # data['policy'] = training_policy
        joblib.dump(training_policy, os.path.join(snapshot_dir, 'params-initial-%.1f.pkl'%params['trpo_init_ratio']))

        train_models(env=env,
                     dynamics_model=dynamics_model,
                     dynamics_opt_params=dynamics_opt_params,
                     get_regularizer_loss=get_regularizer_loss,
                     policy_model=policy_model,
                     policy_opt_params=policy_opt_params,
                     rollout_params=rollout_params,
                     cost_np=cost_np,
                     cost_np_vec=cost_np_vec,
                     cost_tf=cost_tf,
                     snapshot_dir=snapshot_dir,
                     working_dir=working_dir,
                     n_models=params['n_models'],
                     sweep_iters=params['sweep_iters'],
                     sample_size=params['sample_size'],
                     verbose=False,
                     careful_init=careful_init,
                     variant=variant,
                     saved_policy=training_policy,
                     **kwargs) # Make sure not to reinitialize TRPO policy.

        # Save the final policy
        # data['policy'] = training_policy
        joblib.dump(training_policy, os.path.join(snapshot_dir, 'params.pkl'))

    except Exception as e:
        rmtree(snapshot_dir)
        import sys, traceback
        # traceback.print_exception(*sys.exc_info())
        from IPython.core.ultratb import ColorTB
        c = ColorTB()
        exc = sys.exc_info()
        print(''.join(c.structured_traceback(*exc)))
        print('Removed the experiment folder %s.' % snapshot_dir)
