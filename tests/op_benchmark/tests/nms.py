from common_import import *
from mmcv.ops import nms
class NmsConfig(OperatorConfig):
    def __init__(self):
        super(NmsConfig, self).__init__("nms")

class NmsBenchmark(OpBenchmarkBase):
    def __init__(self):
        super(NmsBenchmark, self).__init__(nms)
        
if __name__ == "__main__":
    test_main(op_obj=NmsBenchmark(), config=NmsConfig())