from common_import import *
from mmcv.ops import roi_align
class RoiAlignConfig(OperatorConfig):
    def __init__(self):
        super(RoiAlignConfig, self).__init__("roi_align")

class RoiAlignBenchmark(OpBenchmarkBase):
    def __init__(self):
        super(RoiAlignBenchmark, self).__init__(roi_align)
        
if __name__ == "__main__":
    test_main(op_obj=RoiAlignBenchmark(), config=RoiAlignConfig())