import tensorflow as tf
import numpy as np
from utils import get_variables, flatten_tensors, unflatten_tensors, \
    update_variables


def get_policy_parameters(sess):
    weights = get_variables(scope="training_policy", filter='/b:')
    weights.extend(get_variables(scope="training_policy", filter='/W:'))
    return flatten_tensors(weights), weights

def svg_update(rollout_data,
               policy_gradient,
               model_gradient,
               cost_gradient,
               theta_vars,
               lr,
               sess):
    grads = svg_gradient(rollout_data,
                         policy_gradient,
                         model_gradient,
                         cost_gradient)
    update = unflatten_tensors(-lr * np.squeeze(grads['theta']).astype(np.float32), theta_vars)
    update_variables(sess, update, theta_vars)
    return grads['theta']

def svg_gradient(rollouts,
                 policy_gradient,
                 model_gradient,
                 cost_gradient,
                 gamma=1.0):

    # TODO: can make this faster.
    avg_grads = {'theta': None, 'state': None}
    for rollout in rollouts:
        # Compute grads
        grads = {'theta': None, 'state': None}
        for s, a, s_next in reversed(rollout):
            # Get all required gradients
            cost_grads = cost_gradient(s[None], a[None])
            policy_grads = policy_gradient(s[None])
            model_grads = model_gradient(s[None], a[None])

            if grads['theta'] is None and grads['state'] is None:
                grads['state'] = np.zeros((1, len(s)))
                grads['theta'] = np.zeros((1, policy_grads['theta'].shape[1]))

            # Compute theta grad
            grads['theta'] = cost_grads['action'] @ policy_grads['theta'] + \
                             gamma * (grads['state'] @ model_grads['action'] @ policy_grads['theta'] +
                                      grads['theta'])
            # Compute state grad
            grads['state'] = cost_grads['state'] + cost_grads['action'] @ policy_grads['state'] +\
                             gamma * grads['state'] @ (model_grads['state'] +
                                                       model_grads['action'] @ policy_grads['state'])
        # Update avg_grads
        if avg_grads['theta'] is None and avg_grads['state'] is None:
            avg_grads['state'] = np.zeros((1, len(s)))
            avg_grads['theta'] = np.zeros((1, policy_grads['theta'].shape[1]))
        avg_grads['theta'] += grads['theta']
        avg_grads['state'] += grads['state']

    avg_grads['theta'] /= len(rollouts)
    avg_grads['state'] /= len(rollouts)
    print('grads:', avg_grads['theta'])
    return avg_grads

def setup_gradients(policy_model,
                    dynamics_model,
                    cost_tf,
                    sess,
                    rllab_env,
                    dynamics_configs,
                    ipython=False):
    n_states = rllab_env.observation_space.shape[0]
    n_actions = rllab_env.action_space.shape[0]

    s_in = tf.placeholder(tf.float32, [None, n_states])
    a_in = tf.placeholder(tf.float32, [None, n_actions])

    def cost_gradient(s, a):
        sess = tf.get_default_session()
        grad_opt = tf.get_collection('cost_grad_opt')[0]
        out = sess.run(grad_opt,
                       feed_dict={
                           s_in: s,
                           a_in: a
                       })
        return dict(zip(['state', 'action'], out))

    def flatten_lists(tensor_list):
        return np.concatenate([np.reshape(tensor, [-1]) for tensor in tensor_list], axis=0)

    def policy_gradient(s):
        sess = tf.get_default_session()
        grad_opt = tf.get_collection('policy_grad_opt')
        outs = sess.run(grad_opt,
                        feed_dict={
                            s_in: s
                        })
        grads = [[] for i in range(2)]
        for out in outs:
            grads[0].append(np.sum(out[0], axis=0))
            grads[1].append(flatten_lists(out[1:]))
        grads = [np.array(x) for x in grads]
        return dict(zip(['state', 'theta'], grads))

    def model_gradient(s, a):
        sess = tf.get_default_session()
        grad_opt = tf.get_collection('model_grad_opt')
        outs = sess.run(grad_opt,
                        feed_dict={
                            s_in: s,
                            a_in: a
                        })
        grads = [[] for i in range(2)]
        for out in outs:
            grads[0].append(np.sum(out[0], axis=0))
            grads[1].append(np.sum(out[1], axis=0))
        grads = [np.array(x) for x in grads]
        return dict(zip(['state', 'action'], grads))

    #TODO:Here is a hack to use the current state cost.
    cost_out = cost_tf(None, a_in, s_in)
    tf.add_to_collection('cost_grad_opt', tf.gradients(cost_out, [s_in, a_in]))

    policy_out = policy_model(s_in)
    _, theta_vars = get_policy_parameters(sess)
    for i in range(n_actions):
        tf.add_to_collection('policy_grad_opt', tf.gradients(policy_out[:, i], [s_in] + theta_vars))
    
    if not ipython:
        s_out = dynamics_model(tf.concat([s_in, a_in], axis=1),
                           **dynamics_configs)
    else:
        s_out = dynamics_model(tf.concat([s_in, a_in], axis=1),
                               'training_dynamics',
                               'model%d' % i,
                               **dynamics_configs)

    for i in range(n_states):
        tf.add_to_collection('model_grad_opt', tf.gradients(s_out[:, i], [s_in, a_in]))

    return cost_gradient, policy_gradient, model_gradient, theta_vars


def test_svg_gradient(policy_costs,
                      policy_training_init,
                      initial_state,
                      sess,
                      dynamics_in,
                      dynamics_out,
                      s_in,
                      policy_out,
                      svg_vars,
                      policy_grads_and_vars,
                      cost_np,
                      policy_gradient,
                      model_gradient,
                      cost_gradient
                      ):
    '''
    We can verify svg computation by inputing a simulated trajectory.
    The gradient output should be exactly the same as doing BPTT through
    the whole graph.
    '''
    # First make sure that the TF graph was built correctly.
    # We then get a sample roll-out.
    _policy_costs = sess.run(policy_costs,
             feed_dict={
                 policy_training_init: initial_state
             })

    sample_traj = []
    x = initial_state
    policy_cost = 0
    for t in range(100):
        u = np.clip(sess.run(policy_out,
                             feed_dict={
                                 s_in: x
                             }), -1, 1)
        x_next = sess.run(dynamics_out,
                          feed_dict={
                              dynamics_in: np.concatenate([x, u], axis=1)
                          })
        policy_cost += cost_np(x, u, x_next)
        sample_traj.append((x[0], u[0], x_next[0]))
        # Move forward 1 step.
        x = x_next
    assert np.allclose(_policy_costs[0], policy_cost),\
        'Graph: {}\nOne-step:{}'.format(_policy_costs[0], policy_cost)

    # Compute SVG
    grads = svg_gradient([sample_traj],
                         policy_gradient,
                         model_gradient,
                         cost_gradient)
    svg_grads = unflatten_tensors(np.squeeze(grads['theta']), svg_vars)

    # We check SVG with the automatic BPTT on simulated trajectory.
    var2grad = dict(zip(svg_vars, sess.run(svg_grads)))
    count = 0
    for grad, var in policy_grads_and_vars:
        if var in var2grad:
            grad_nu = sess.run(grad,
                               feed_dict={policy_training_init: np.zeros(10)[None]})
            np.allclose(var2grad[var], grad_nu)
            count+=1
    assert count == len(svg_vars)
