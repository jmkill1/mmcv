from common_import import *

class BroadcastTensorsConfig(OperatorConfig):
    def __init__(self):
        super(BroadcastTensorsConfig, self).__init__("broadcast_tensors")
        
if __name__ == "__main__":
    op = BroadcastTensorsConfig()
    op.init_config()
    print("_____")