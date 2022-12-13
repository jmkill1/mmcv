from __future__ import print_function
import argparse
import os
import json
import sys
import warnings
import collections
import numpy as np

from common.utils import *
# from common import system
# from common import special_op_list


# def _check_gpu_device(use_gpu):
#     gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
#     if use_gpu:
#         assert gpu_devices, "export CUDA_VISIBLE_DEVICES=\"x\" to test GPU performance."
#         assert len(gpu_devices.split(",")) == 1
#     else:
#         assert gpu_devices == "", "export CUDA_VISIBLE_DEVICES=\"\" to test CPU performance."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Specify the benchmark filename')
    parser.add_argument(
        '--config_id',
        type=int,
        default=None,
        help='Only import params of operator from json file in the specified position [0|1|...]'
    )
    parser.add_argument(
        '--unknown_dim',
        type=int,
        default=16,
        help='Specify the unknown dimension.')
    parser.add_argument(
        '--profiler',
        type=str,
        default="none",
        help='Choose which profiler to use [\"none\"|\"Default\"|\"OpDetail\"|\"AllOpDetail\"|\"pyprof\"]'
    )
    parser.add_argument(
        '--backward',
        type=str2bool,
        default=False,
        help='Whether appending grad ops [True|False]')
    parser.add_argument(
        '--convert_to_fp16',
        type=str2bool,
        default=False,
        help='Whether using gpu [True|False]')
    parser.add_argument(
        '--use_gpu',
        type=str2bool,
        default=False,
        help='Whether using gpu [True|False]')
    parser.add_argument(
        '--gpu_time',
        type=float,
        default=0,
        help='Total GPU kernel time parsed from nvprof')

    parser.add_argument(
        '--repeat', type=int, default=1, help='Iterations of Repeat running')
    parser.add_argument(
        '--allow_adaptive_repeat',
        type=str2bool,
        default=False,
        help='Whether use the value repeat in json config [True|False]')
    parser.add_argument(
        '--log_level', type=int, default=0, help='level of logging')

    args = parser.parse_args()

    # _check_gpu_device(args.use_gpu)
    print(args)
    return args


def test_main(op_obj=None, config=None):
    assert config is not None, "Operator json config must be set."
    args = parse_args()

    if args.config_id is not None and args.config_id >= 0:
        if args.convert_to_fp16:
            config.convert_to_fp16()
        config.init_config(args.config_id, args.unknown_dim)
        test_main_without_json(op_obj, config)


def _adaptive_repeat(config, args):
    if args.allow_adaptive_repeat and hasattr(
            config, "repeat"):
        if args.use_gpu:
            args.repeat = config.repeat


def _check_disabled(config, args):
    if config.disabled():
        status = collections.OrderedDict()
        status["name"] = config.op_name
        status["device"] = "GPU" if args.use_gpu else "CPU"
        status["backward"] = args.backward
        status["disabled"] = True
        status["parameters"] = config.to_string()
        print(json.dumps(status))
        return True
    return False


def test_main_without_json(op_obj=None, config=None):
    assert config is not None, "Operator config must be set."

    args = parse_args()


    _adaptive_repeat(config, args)
    config.backward = args.backward
    assert op_obj is not None, "Operator object is None."
    
    print(config)
    outputs, status = op_obj.run(config, args)

    if args.gpu_time is not None:
        status["gpu_time"] = args.gpu_time
        print_benchmark_result(
            status,
            log_level=args.log_level,
            config_params=config.to_string())
