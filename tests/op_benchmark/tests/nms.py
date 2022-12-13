from common_import import *

class NmsConfig(OperatorConfig):
    def __init__(self):
        super(NmsConfig, self).__init__("nms")
        
class TorchNms(OpBenchmarkBase):
    def build_graph(self, config):
        boxes = self.variable(name='boxes', shape=config.boxes_shape, dtype=config.boxes_dtype)
        scores = self.variable(name='scores', shape=config.scores_shape, dtype=config.scores_dtype)

        
if __name__ == "__main__":
    test_main(op_obj=TorchNms(), config=NmsConfig())
   
    # config = LerpConfig()
    # config.init_config()
    # print(config.to_string())
    # print(config.w)
    
    print("_____")