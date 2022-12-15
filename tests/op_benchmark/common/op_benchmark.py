import sys
import time
import importlib
import contextlib

from common.utils import *


try:
    import torch
except Exception as e:
    sys.stderr.write(
        "Cannot import pytorch, maybe pytorch is not installed.\n")

BEFORE_RUN = 0
IN_RUN = 1
AFTER_RUN = 2


@contextlib.contextmanager
def profile_context(name, use_gpu, profiler):
    if profiler == "nvprof":
        torch.cuda.cudart().cudaProfilerStart()
        yield
        torch.cuda.cudart().cudaProfilerStop()
    else:
        yield


class OpBenchmarkBase():
    def __init__(self, func):
        self.__inputs_dict = {}
        self.__generated_feed_values = []
        self.__status = BEFORE_RUN
        self.__test_func = func
        self.__test_kwargs = None
        self.__outputs_list = None

    def build_graph(self, config=None):
        if self.__test_kwargs is None:
            assert config.op_name is not None
            self.__test_kwargs = {}

            self.inputs_list = []
            for var in config.tensor_list:
                var_shape = getattr(config, var.name + '_shape')
                var_dtype = getattr(config, var.name + '_dtype')
                arg_name = var.name
                feed_var = self.variable(
                    name=var.name, shape=var_shape, dtype=var_dtype)
                self.__test_kwargs[arg_name] = feed_var
                self.inputs_list.append(feed_var)

            for param in config.attr_list:
                arg_name = param.name
                self.__test_kwargs[arg_name] = getattr(config, param.name)
                self.inputs_list.append(getattr(config, param.name))

        outputs = self.__test_func(*self.inputs_list)
        self.__outputs_list = outputs if isinstance(outputs, list) else [outputs]

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        if self.__status == BEFORE_RUN:
            assert shape is not None

            feed_value = generate_random_data(
                shape, dtype, value=value)
            requires_grad = True
            if stop_gradient or dtype not in ["float16", "float32", "float64"]:
                requires_grad = False

            var = torch.tensor(
                feed_value, requires_grad=requires_grad, device=self._device)
            if requires_grad:
                var.retain_grad()
            self.__inputs_dict[name] = var

            if value is None:
                self.__generated_feed_values.append(feed_value)
        else:
            var = self.__inputs_dict[name]
        return var

    def run_impl(self, use_gpu, config, repeat=1, profiler="none"):
        def _run_main_iter():
            self.build_graph(config=config)
            if use_gpu:
                torch.cuda.synchronize(self._device)

        # warmup run
        _run_main_iter()

        runtimes = []
        self.__status = IN_RUN
        with profile_context(self.name, use_gpu, profiler):
            for i in range(repeat):
                begin = time.time()
                _run_main_iter()
                outputs = self.__outputs_list
                runtimes.append(time.time() - begin)

        self.__status = AFTER_RUN
        stats = self.get_running_stats(use_gpu, config, runtimes)
        return outputs, stats

    def get_running_stats(self, use_gpu, config, runtimes, walltimes=None):
        try:
            module_name = "torch"
            module = importlib.import_module(module_name)
            version = module.__version__
        except Exception:
            version = "none"
            print("Failed to call torch.__version__")

        stats = {
            "version": version,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "total": runtimes
        }

        if walltimes is not None:
            stats["wall_time"] = walltimes

        return stats

    def run(self, config, args):
        self.name = config.op_name

        if args.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        outputs, stats = self.run_impl(
            use_gpu=args.use_gpu,
            config=config,
            repeat=args.repeat,
            profiler=args.profiler)
        return outputs, stats
