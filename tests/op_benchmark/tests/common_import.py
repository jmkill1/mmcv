import nvidia_dlprof_pytorch_nvtx
import os, sys

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(package_path)
sys.path.append(package_path)
from common.operatorconfig import OperatorConfig
from common.main import test_main
from common.op_benchmark import OpBenchmarkBase

nvidia_dlprof_pytorch_nvtx.init()
