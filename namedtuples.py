from collections import namedtuple

Dynamics_opt_params = namedtuple('Dynamics_opt_params',
                                 ['log_every',
                                  'max_passes',
                                  'stop_critereon',
                                  'learning_rate',
                                  'batch_size',
                                  'sample_mode',
                                  'reinitialize',
                                  'num_passes_threshold'])

Policy_opt_params = namedtuple('Policy_opt_params',
                               ['validation_init_path',
                                'validation_reset_init_path',
                                'log_every',
                                'num_iters_threshold',
                                'max_iters',
                                'T',
                                'grad_norm_clipping',
                                'gamma',
                                'oracle_maxtimestep',
                                'stop_critereon',
                                'learning_rate',
                                'mode',
                                'trpo',
                                'vpg',
                                'batch_size',
                                'whole',
                                'sam_mode'])

Rollout_params = namedtuple('Rollout_params',
                            ['split_ratio',
                             'training_data_size',
                             'validation_data_size',
                             'exploration',
                             'use_same_dataset',
                             'splitting_mode',
                             'load_rollout_data',
                             'datapath', # Where to save rollout data if save.
                             'render_every', # Everyow many transitions we render.
                             'max_timestep',
                             'is_monitored',
                             'monitorpath'])
