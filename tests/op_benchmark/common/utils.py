import os
import json
import collections
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def use_gpu():
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""

def is_string(value):
    return isinstance(value, str)

def parse_string(value):
    import six
    # PY2     : PY3
    # unicode : str
    # str     : bytes
    if six.PY3:
        return value
    else:
        return value.encode("utf-8") if isinstance(value, str) else value

def parse_int(value):
    if isinstance(value, int):
        return value
    else:
        return int(parse_string(value))
    
def parse_list(value_str, sub_dtype="int"):
    value_str = parse_string(value_str)
    if sub_dtype in ["int", "int64"]:
        try:
            if value_str != "[]":
                value_str_list = value_str.replace("L", "").replace(
                    "[", "").replace("]", "").split(',')
                value_list = []
                for item in value_str_list:
                    value_list.append(int(item))
                return value_list
            else:
                return []
        except Exception as e:
            assert False, "Parse {} failed: {}".format(value_str, e)
    else:
        # TODO: check and support list of other data type.
        raise ValueError("Do not support parsing list of non-int data type.")


def parse_tuple(value_str, sub_dtype="int"):
    value_str = parse_string(value_str)
    if sub_dtype in ["int", "int64"]:
        try:
            if value_str != "()":
                value_str_list = value_str.replace("L", "").replace(
                    "(", "").replace(")", "").split(',')
                value_list = []
                for item in value_str_list:
                    value_list.append(int(item))
                return value_list
            else:
                return []
        except Exception as e:
            assert False, "Parse {} failed: {}".format(value_str, e)
    else:
        # TODO: check and support list of other data type.
        raise ValueError("Do not support parsing list of non-int data type.")
    
def print_benchmark_result(result,
                           task="speed",
                           log_level=0,
                           config_params=None):
    assert isinstance(result, dict), "Input result should be a dict."

    status = collections.OrderedDict()
    status["version"] = result["version"]
    status["name"] = result["name"]
    status["device"] = result["device"]


    runtimes = result.get("total", None)
    if runtimes is None:
        status["parameters"] = config_params
        print(json.dumps(status))
        return

    walltimes = result.get("wall_time", None)
    gpu_time = result.get("gpu_time", None)
    stable = result.get("stable", None)
    diff = result.get("diff", None)

    repeat = len(runtimes)
    for i in range(repeat):
        runtimes[i] *= 1000
        if walltimes is not None:
            walltimes[i] *= 1000

    sorted_runtimes = np.sort(runtimes)
    if repeat <= 2:
        num_excepts = 0
    elif repeat <= 10:
        num_excepts = 1
    elif repeat <= 20:
        num_excepts = 5
    else:
        num_excepts = 10
    begin = num_excepts
    end = repeat - num_excepts
    avg_runtime = np.average(sorted_runtimes[begin:end])
    if walltimes is not None:
        avg_walltime = np.average(np.sort(walltimes)[begin:end])
    else:
        avg_walltime = 0

    # print all times
    seg_range = [0, 0]
    if log_level == 0:
        seg_range = [0, repeat]
    elif log_level == 1 and repeat > 20:
        seg_range = [10, repeat - 10]
    for i in range(len(runtimes)):
        if i < seg_range[0] or i >= seg_range[1]:
            walltime = walltimes[i] if walltimes is not None else 0
            print("Iter %4d, Runtime: %.5f ms, Walltime: %.5f ms" %
                  (i, runtimes[i], walltime))

    if avg_runtime - avg_walltime > 0.001:
        total = avg_runtime - avg_walltime
    else:
        print(
            "Average runtime (%.5f ms) is less than average walltime (%.5f ms)."
            % (avg_runtime, avg_walltime))
        total = 0.001

    if stable is not None and diff is not None:
        status["precision"] = collections.OrderedDict()
        status["precision"]["stable"] = stable
        status["precision"]["diff"] = diff
    status["speed"] = collections.OrderedDict()
    status["speed"]["repeat"] = repeat
    status["speed"]["begin"] = begin
    status["speed"]["end"] = end
    status["speed"]["total"] = total
    status["speed"]["wall_time"] = avg_walltime
    status["speed"]["total_include_wall_time"] = avg_runtime
    if gpu_time is not None:
        avg_gpu_time = gpu_time / repeat
        status["speed"]["gpu_time"] = avg_gpu_time

        flop = result.get("flop", None)
        byte = result.get("byte", None)
        if flop is not None and abs(avg_gpu_time) > 1E-6:
            status["speed"]["gflops"] = float(flop) * 1E-6 / avg_gpu_time
        if byte is not None and abs(avg_gpu_time) > 1E-6:
            status["speed"]["gbs"] = float(byte) * 1E-6 / avg_gpu_time
    status["parameters"] = config_params
    print(json.dumps(status))