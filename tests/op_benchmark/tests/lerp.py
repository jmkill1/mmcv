from common_import import *

class LerpConfig(OperatorConfig):
    def __init__(self):
        super(LerpConfig, self).__init__("lerp")
        
if __name__ == "__main__":
    lerp_config = LerpConfig()
    lerp_config.init_config()
    print("_____")