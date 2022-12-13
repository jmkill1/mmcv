from common_import import *

class LerpConfig(OperatorConfig):
    def __init__(self):
        super(LerpConfig, self).__init__("lerp")
        
class TorchLerp(OpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)

        
if __name__ == "__main__":
    # test_main(op_obj=TorchLerp(), config=LerpConfig())
   
    # config = LerpConfig()
    # config.init_config()
    # print(config.to_string())
    # print(config.w)
    
    print("_____")