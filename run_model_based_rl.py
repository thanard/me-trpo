from rllab.misc.instrument import run_experiment_lite
from training import train
import rllab.config as config
import os.path
import json
import argparse
import numpy as np


def get_aws_config(count, use_gpu=True):
    # TODO: running on us-east-2 has a problem when aws s3 sync or aws s3 cp.
    # 'us-east-2c', 'us-east-2a', 'us-east-2b'
    # 'ap-northeast-2c' -> doesn't have name
    if use_gpu:
        cheapest_zones = ['us-east-1d', 'us-east-1b', 'us-east-1a',  # 'us-east-1e',
                          'eu-west-1a', 'eu-west-1c']
        instance_type = "p2.xlarge"
    else:
        cheapest_zones = ['us-east-1b', 'us-east-1a', 'us-east-1d', 'us-east-1e',
                          'us-west-1c', 'us-west-1b',
                          'us-west-2c', 'us-west-2a', 'us-west-2b']
        instance_type = "c4.4xlarge"
    # zone = cheapest_zones[count % len(cheapest_zones)]
    zone = np.random.choice(cheapest_zones)
    region = zone[:-1]
    config.AWS_REGION_NAME = region
    aws_config = dict(
        instance_type=instance_type,
        image_id=config.ALL_REGION_AWS_IMAGE_IDS[region],
        key_name=config.ALL_REGION_AWS_KEY_NAMES[region],
        network_interfaces=[
            dict(
                SubnetId=config.ALL_SUBNET_INFO[zone]["SubnetID"],
                Groups=[config.ALL_SUBNET_INFO[zone]["Groups"]],
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ]
    )
    return aws_config


def replace_dict(main_dict, input_dict):
    for key, value in input_dict.items():
        assert key in main_dict
        if type(value) == dict:
            assert type(main_dict[key]) == dict
            replace_dict(main_dict[key], value)
        else:
            print("change key %s to value %s" % (str(key), str(value)))
            main_dict[key] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiment options')
    parser.add_argument('algo')
    parser.add_argument('-ec2', action="store_true", default=False)
    parser.add_argument('-env')
    parser.add_argument('-prefix')
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-replace', type=str, default="{}")
    parser.add_argument('-f',
                        action="store_true",
                        default=False,
                        help='force'
                        )
    options = parser.parse_args()
    working_dir = config.PROJECT_PATH
    if options.env in ['half-cheetah',
                       'snake',
                       'point2D',
                       'point-mass',
                       'hopper',
                       'ant',
                       'swimmer',
                       'humanoid']:
        param_path = os.path.join(working_dir,
                                  'sandbox/thanard/me-trpo/params/params-%s.json' % options.env)
    else:
        raise ('Value Error: not implemented.')

    with open(param_path, 'r') as f:
        params = json.load(f)

    # Write paths to upload to data_upload
    remove_list = [
        params['policy_opt_params']['validation_init_path'],
        params['policy_opt_params']['validation_reset_init_path'],
        params['rollout_params']['datapath']
    ]
    config.FAST_CODE_SYNC_IGNORES = filter(lambda v: v not in remove_list,
                                           config.FAST_CODE_SYNC_IGNORES)

    # Replace params keys with values from replace dict.
    input_dict = eval(options.replace)
    replace_dict(params, input_dict)

    if params['algo'] != options.algo:
        if options.f:
            response = 'Y'
        else:
            response = input("The algo option in params is %s. Are you sure you want to run %s [y/N]?"
                             % (params['algo'], options.algo))
        if response in ['Y', 'y']:
            params['algo'] = options.algo
        else:
            import sys

            sys.exit()


    # Exception
    def l_bfgs_exception(params):
        if params['algo'] == 'l-bfgs':
            params["policy_opt_params"]["log_every"] = 1
            params["policy_opt_params"]["max_iters"] = 1


    if params['env'] != options.env:
        response = input("The env option in params is %s. Are you sure you want to run %s [y/N]?"
                         % (params['env'], options.env))
        if response in ['Y', 'y']:
            params['env'] = options.env
        else:
            import sys

            sys.exit()

    if options.prefix is None:
        exp_prefix = params['env']
    else:
        exp_prefix = options.prefix


    def check_not_none(*args):
        for arg in args:
            assert arg is not None


    if options.ec2:
        mode = "ec2"
        # Neccessary otherwise fail.
        assert 'render_every' in params['rollout_params']
        params['rollout_params']['render_every'] = None
        count = 1
        for i in range(options.n):
            l_bfgs_exception(params)
            aws_config = get_aws_config(count)
            run_experiment_lite(
                train,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=dict(mode=mode, params=params, use_gpu=True, seed=i),
                dry=False,
                aws_config=aws_config,
                sync_s3_pkl=True,
                sync_s3_png=True,
                sync_s3_log=True,
                pre_commands=["pip install --upgrade pip",
                              "pip install mpi4py",
                              "pip install plotly",
                              "pip install pandas",
                              "pip install seaborn"],
                use_gpu=True
                # terminate_machine=False
            )
            print(count)
            count += 1
    else:
        mode = "local"
        l_bfgs_exception(params)
        import colored_traceback.always

        aws_config = get_aws_config(1)
        run_experiment_lite(
            train,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=dict(mode=mode, params=params, seed=options.seed),
            dry=False,
            snapshot_mode='last',
            aws_config=aws_config
        )
