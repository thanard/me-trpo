import numpy as np
import random
import tensorflow as tf
import datetime
import logging
from multiprocessing import Process, Pipe

####################
#### Utils #########
####################
def get_experiment_name():
    x = datetime.datetime.now()
    return 'experiment-%d-%d-%d-%d-%d-%d'% (x.year, x.month, x.day, x.hour, x.minute, x.second)

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun
'''
A replacement for pool.map but more general. See
http://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
'''
def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in zip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

'''
Set global seeds
'''
def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


'''
This class can hold data and produce a randomized batch for next step SGD.
If data exceeds max size, we do first in first out.
'''
class data_collection:
    def __init__(self, max_size = int(5e4)):
        self.cur_idx = 0
        self.x = None
        self.y = None
        self.n_data = None
        self.max_size = max_size

    def clone(self, dc, first_i=None):
        assert first_i is None or first_i <= dc.n_data, "Not enough data for first_i."
        self.set_data(dc.x[:first_i], dc.y[:first_i])

    def cap_data_size(self):
        new_start_idx = self.x.shape[0] - self.max_size
        if new_start_idx > 0:
            self.x = self.x[new_start_idx:]
            self.y = self.y[new_start_idx:]
            self.n_data = self.max_size
            self.cur_idx -= new_start_idx

    # Set x and y of data
    def set_data(self, x, y, is_shuffled = False):
        assert x.shape[0] == y.shape[0]
        self.n_data = x.shape[0]
        self.x = x
        self.y = y
        self.cur_idx %= self.n_data
        self.cap_data_size()
        self.idx_mapping = list(range(self.n_data))
        if is_shuffled:
            self.reshuffle_indices()

    def add_data(self, x_new, y_new, is_shuffled = False):
        assert x_new.shape[0] == y_new.shape[0]
        if self.x is not None:
            # Point to new data
            self.cur_idx = self.x.shape[0]
            self.x = np.concatenate([self.x, x_new], axis =0)
            self.y = np.concatenate([self.y, y_new], axis =0)
        else:
            self.cur_idx = 0
            self.x = x_new
            self.y = y_new
        self.n_data = self.x.shape[0]
        self.cap_data_size()
        self.idx_mapping = list(range(self.n_data))
        if is_shuffled:
            self.reshuffle_indices()

    def get_num_data(self):
        if self.n_data is None:
            return 0
        return self.n_data

    def reshuffle_data(self):
        shuffled_indices = list(range(self.n_data))
        np.random.shuffle(shuffled_indices)
        self.x = self.x[shuffled_indices,:]
        self.y = self.y[shuffled_indices,:]

    # Not neccessary but can reshuffle indices once in a while
    def reshuffle_indices(self):
        np.random.shuffle(self.idx_mapping)

    # This remap the indices to data in case of shuffling
    def remap(self, list_idx):
        return list(map(lambda x: x, list_idx))

    # Compute and return next_batch x and y
    def get_next_batch(self, batch_size, is_shuffled=False):
        assert batch_size<=self.n_data, \
            "Batch size %d is larger than n_data %d"%(batch_size,self.n_data)
        start_idx = self.cur_idx
        end_idx = self.cur_idx + batch_size
        if end_idx > self.n_data:
            indices = list(range(start_idx, self.n_data)) + \
                list(range(0, batch_size - (self.n_data-start_idx)))
            self.cur_idx = batch_size - (self.n_data-start_idx)
        else:
            indices = list(range(start_idx, end_idx))
            self.cur_idx = end_idx
        assert(len(indices) == end_idx - start_idx)
        return self.x[self.remap(indices), :], self.y[self.remap(indices), :]

    # Uniformly random a batch of data (with replacement)
    def sample(self, batch_size):
        indices = np.floor(self.n_data*np.random.uniform(0.0, 1.0, size=batch_size)).astype(np.intp)
        return self.x[indices, :], self.y[indices, :]

def combine_data_collections(dc1, dc2):
    dc_out = data_collection(max_size = max(dc1.max_size, dc2.max_size))
    if dc2.max_size < dc1.max_size:
        x = np.concatenate([dc1.x,dc2.x], axis=0)
        y = np.concatenate([dc1.y,dc2.y], axis=0)
    else:
        x = np.concatenate([dc2.x,dc1.x], axis=0)
        y = np.concatenate([dc2.y,dc1.y], axis=0)
    dc_out.set_data(x,y)
    return dc_out

# Run this to see if data collection is working properly
def test_data_collection():
    dc = data_collection(3)
    x = np.array([[1,2],[3,4],[5,6],[7,8]])
    dc.set_data (x,x)
    dc.get_next_batch(2)
    dc.get_next_batch(2)
    dc.get_next_batch(2)
    dc.set_data(x,x, True)
    dc.get_next_batch(2)
    dc.get_next_batch(2)
    dc.get_next_batch(2)

    x_new = np.array([[0,0]])
    dc.add_data(x_new, x_new)
    dc.get_next_batch(2)
    dc.get_next_batch(2)
    dc.get_next_batch(2)
    x_new = np.array([[9,9],[8,8]])
    dc.add_data(x_new, x_new,True)
    dc.get_next_batch(2)
    dc.get_next_batch(2)
    dc.get_next_batch(2)

def test_combine_data_collection():
    dc1 = data_collection(10)
    x = np.reshape(np.arange(20),newshape=(10,2))
    dc1.set_data(x, x)
    y = np.reshape(-np.arange(10),newshape=(5,2))
    dc2 = data_collection(5)
    dc2.set_data(y,y)
    return [combine_data_collections(dc1, dc2),combine_data_collections(dc2, dc1)]

def data_summaries(data):
    tf.summary.histogram('histogram', data)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def flatten_tensors(tensor_list):
    return tf.concat([tf.reshape(tensor,[-1]) for tensor in tensor_list], axis=0)

def unflatten_tensors(flat_tensor, variables):
    cur_idx = 0
    tensors = []
    for var in variables:
        shape = var.get_shape().as_list()
        var_size = np.prod(shape)
        tensor = tf.reshape(flat_tensor[cur_idx:cur_idx+var_size], shape)
        tensors.append(tensor)
        cur_idx += var_size
    return tensors

def get_variables(scope, filter=''):
    vars = []
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
        if filter in v.name:
            vars.append(v)
    return vars

def update_variables(sess, tensors, variables):
    opts = []
    for i,v in enumerate(variables):
        opts.append(v.assign_add(tensors[i]))
    sess.run(opts)

def get_update_variable_opt(tensors, variables):
    opts = []
    for i,v in enumerate(variables):
        opts.append(v.assign_add(tensors[i]))
    return opts

def get_session(interactive=False, mem_frac=0.25, use_gpu=True):
    tf.reset_default_graph()
    if use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            gpu_options=gpu_options)
        if interactive:
            session = tf.InteractiveSession(config=tf_config)
        else:
            session = tf.Session(config=tf_config)
        print("AVAILABLE GPUS: ", get_available_gpus())
        return session
    # IF not using gpu
    config = tf.ConfigProto(
        device_count = {'GPU' : 0}
    )
    if interactive:
        return tf.InteractiveSession(config=config)
    return tf.Session(config=config)

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

# If the variable exists, reuse it. If not, create it.
def get_scope_variable(scope_name, var, shape=None, initializer= None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape=shape, dtype=tf.float32, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var, dtype=tf.float32)
    return v

def minimize_and_clip(optimizer, objective, var_list, clip_val=None, collect_summary=False):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            if clip_val is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
            if collect_summary:
                with tf.name_scope('%s/%s%s/gradients' % (objective.name[:-2], var.name[:-2], var.name[-1])):
                    variable_summaries(gradients[i][0])
                    tf.summary.scalar('norm', tf.norm(gradients[i]))
    return optimizer.apply_gradients(gradients)

def get_pickeable(namedtuple_object):
    _dict = {}
    for key, value in namedtuple_object._asdict().items():
        if not callable(value):
            _dict[key] = value
    return _dict

def stop_critereon(threshold, offset, percent_models_threshold = 0.5):
    def f(loss_old, loss_new, mode = 'scalar'):
        if mode == 'scalar':
            assert not hasattr(loss_new, '__iter__')
            return (loss_new-loss_old)/(np.abs(loss_old)+offset) > threshold
        else:
            assert mode == 'vector'
            assert isinstance(loss_new, np.ndarray)
            # Binary
            out = loss_new > loss_old
            return np.mean(out) > percent_models_threshold
    return f

def get_logger(logger_name, folderpath, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    to_generate = [('info.log',logging.INFO), ('debug.log',logging.DEBUG)]
    for logname, handler_level in to_generate:
        # Create a file handler
        handler = logging.FileHandler(folderpath+'/' + logname)
        handler.setLevel(handler_level)

        # Create a logging format
        if logname == 'debug.log':
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

import rllab.misc.logger as logger
def update_dictionary(old_dict, new_dict):
    for key, value in new_dict.items():
        if key in old_dict:
            if type(value) is dict:
                update_dictionary(old_dict[key], value)
            elif type(value) is list:
                logger.log('%s is changed.'%key)
                logger.log('WARNING {} is being replaced by {}'.format(old_dict[key], value))
                old_dict[key] = value
            else:
                logger.log('%s is changed.'%key)
                logger.log('{} is being replaced by {}'.format(old_dict[key], value))
                old_dict[key] = value
        else:
            old_dict[key] = value

def test_update_dictionary():
    a1={
        'my_home':{
            'size':'100sqft',
            'color':'red',
            'has_pillow':True
        },
        'my_car':{
            'objects':['wheels', 'driver']
        },
        'my_cat':[
            'ketty',
            {
                'jake':1,
                'john':3
            }
        ],
        'my_height':100
    }
    update_a1={
        'my_home':{
            'size':1
        },
        'my_car':{
            'objects':[],
            'people':['mom']
        },
        'my_height':90
    }
    update_dictionary(a1, update_a1)
    print(a1)

def get_ith_tensor(tensor, i, sliced_length):
    ''' row only '''
    assert tensor.shape[1] % sliced_length ==0
    return tensor[:, i*sliced_length: (i+1)*sliced_length]


def assign_eq_policy(policy, target):
    '''
    This helper function needs to be called after both policies have been initialized.
    :param policy: initial rllab Gaussian MLP
    :param target: target rllab Gaussian MLP
    :return:
    '''
    sess = tf.get_default_session()
    policy_layers = policy._mean_network.layers
    target_layers = target._mean_network.layers
    for i, policy_layer in enumerate(policy_layers[1:]):
        i += 1
        sess.run([policy_layer.W.assign(target_layers[i].W),
                  policy_layer.b.assign(target_layers[i].b)])