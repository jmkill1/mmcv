import sys
import time
import importlib
import contextlib

from common import feeder


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
    def __init__(self):
        self._feed_spec = None
        self._generated_feed_values = []
        self._feed_dict = {}
        self._status = BEFORE_RUN

    def build_graph(self, config=None):
        def _get_func(callable_api):
            callable_api_list = callable_api.split(".")
            func_name = callable_api_list[-1]
            callable_api_list.pop()
            module_name = ".".join(callable_api_list)
            try:
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                return func
            except Exception:
                print("Failed to import {}.{}".format(module_name, func_name))
            return None

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        if self._status == BEFORE_RUN:
            assert shape is not None

            if self._feed_spec is not None and value is None:
                i = len(self._feed_dict)
                range = self._feed_spec[i].get("range", None)
            else:
                range = None
            feed_value = feeder.generate_random_data(
                shape, dtype, range=range, value=value)

            requires_grad = True
            if stop_gradient or dtype not in ["float16", "float32", "float64"]:
                requires_grad = False

            var = torch.tensor(
                feed_value, requires_grad=requires_grad, device=self._device)
            if requires_grad:
                var.retain_grad()
            self._feed_dict[name] = var

            if value is None:
                self._generated_feed_values.append(feed_value)
        else:
            var = self._feed_dict[name]
        return var

    def generate_random_feeder(self, config):
        return feeder.FeederAdapter("pytorch", config.feed_spec,
                                    self._generated_feed_values)

    def append_gradients(self, targets, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        for var in inputs:
            var.grad = None

        if not isinstance(targets, list):
            if len(self._ones_like_targets) == 0:
                ones_like_targets = torch.ones_like(targets)
                self._ones_like_targets.append(ones_like_targets)
            else:
                ones_like_targets = self._ones_like_targets[0]
            targets.backward(gradient=ones_like_targets)
            targets.retain_grad()
            self._backward = True
        else:
            # torch.autograd.backward(tensors=inputs, grad_tensors=targets)
            assert False, "Gradients of list is not supported now!"
        for var in inputs:
            self.fetch_list.append(var.grad)

    def run_impl(self, use_gpu, config, repeat=1, profiler="none"):
        def _run_main_iter():
            self.build_graph(config=config)
            if use_gpu:
                torch.cuda.synchronize(self._device)

        # warmup run
        _run_main_iter()

        runtimes = []
        self._status = IN_RUN
        with profile_context(self.name, use_gpu, profiler):
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter()
                runtimes.append(time.time() - begin)

        self._status = AFTER_RUN
        stats = self.get_running_stats(use_gpu, config, runtimes)
        return outputs, stats

    def get_running_stats(self, use_gpu, config, runtimes, walltimes=None):
        try:
            module_name = "torch" if self._framework == "pytorch" else self._framework
            module = importlib.import_module(module_name)
            version = module.__version__
        except Exception:
            version = "none"
            print("Failed to call %s.__version__" % (self._framework))

        stats = {
            "framework": self._framework,
            "version": version,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }

        if walltimes is not None:
            stats["wall_time"] = walltimes

        flop, byte = self.compute_flop_and_byte(config)
        if flop is not None:
            stats["flop"] = flop
        if byte is not None:
            stats["byte"] = byte
        return stats

    def run(self, config, args):
        self.name = config.op_name

        self.reset()
        self._feed_spec = feeder.copy_feed_spec(config.feed_spec)

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
