from __future__ import print_function
import torch
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


def _check_gpu_device(use_gpu):
    gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if use_gpu:
        assert gpu_devices, "export CUDA_VISIBLE_DEVICES=\"x\" to test GPU performance."
        assert len(gpu_devices.split(",")) == 1
    else:
        assert gpu_devices == "", "export CUDA_VISIBLE_DEVICES=\"\" to test CPU performance."

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

    _check_gpu_device(args.use_gpu)
    print(args)
    return args


def test_main(config=None):
    assert config is not None, "Operator json config must be set."

    def _test_with_json_impl(config_id, unknown_dim,
                             convert_to_fp16):
        if convert_to_fp16:
            config.convert_to_fp16()
        config.load_input_from_json(config_id, unknown_dim)
        test_main_without_json(config)
        
    args = parse_args()
    
    if args.config_id is not None and args.config_id >= 0:
        _test_with_json_impl(args.config_id, args.unknown_dim,
                                args.convert_to_fp16)
    

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

def generate_random_data(shape, dtype):
    if dtype == "int64" or dtype == "int32":
        data = np.random.randint(100, size=shape, dtype=dtype)
        if range is not None:
            data = np.random.randint(
                range[0], range[1], size=shape, dtype=dtype)
    elif dtype == "bool":
        data = np.random.randint(2, size=shape, dtype=bool)
    elif dtype == "uint8" or dtype == "uint16":
        data = np.random.randint(0, 100, size=shape, dtype=dtype)
        range_low = max(0, range[0])
        if range is not None:
            data = np.random.randint(
                range_low, range[1], size=shape, dtype=dtype)
    else:
        data = np.random.random(shape).astype(dtype)
        if range is not None:
            data = range[0] + (range[1] - range[0]) * data
    return data
    
def run(data, config, args):
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    outputs, stats = run_impl(
        use_gpu=args.use_gpu,
        config=config,
        repeat=args.repeat,
        profiler=args.profiler)
    return outputs, stats

def run_impl(self, use_gpu, config, repeat=1, profiler="none"):
        def _run_main_iter():
            self.build_graph(config=config)
            if use_gpu:
                torch.cuda.synchronize(self._device)

            outputs = None
            if self._need_fetch:
                outputs = []
                for var in self.fetch_list:
                    if isinstance(var, torch.Tensor):
                        outputs.append(var.to("cpu").detach().numpy())
                    elif isinstance(var, list):
                        outputs.append(np.array(var))
                    else:
                        outputs.append(np.array([var]))
            return outputs

        # warmup run
        _run_main_iter()

        runtimes = []
        fetches = []
        self._status = IN_RUN
        with profile_context(self.name, use_gpu, profiler):
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter()
                runtimes.append(time.time() - begin)

        self._status = AFTER_RUN
        stats = self.get_running_stats(use_gpu, config, runtimes)
        return outputs, stats
    

def test_main_without_json(config=None):
    assert config is not None, "Operator config must be set."

    args = parse_args()
    if _check_disabled(config, args):
        return

    _adaptive_repeat(config, args)
    config.backward = args.backward
    feeder_adapter = None 
    op = config.op_name
    try:
        from mmcv.ops import op
    except Exception as e:
        sys.stderr.write(
            "Cannot import torch or mmcv.ops.(%s), maybe pytorch or mmcv is not installed.\n", op)

    
    data = generate_random_data(config)
    run(data)

    if args.task == "speed":
        torch_stats["gpu_time"] = args.gpu_time
        utils.print_benchmark_result(
            torch_stats,
            task=args.task,
            log_level=args.log_level,
            config_params=config.to_string())