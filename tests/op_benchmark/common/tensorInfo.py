from common.utils import *
from common.attrinfo import AttrInfo


class TensorInfo(AttrInfo):
    def __init__(self, name, type, dtype, shape, lod_level=0):
        self.name = self.trans2str(name)
        self.type = self.trans2str(type)
        self.dtype = self.trans2str(dtype)
        shape_str = self.trans2str(shape)
        if is_string(shape_str):
            self.shape = parse_list(shape_str)
        else:
            assert isinstance(shape_str, list)
            self.shape = shape_str
        self.lod_level = self.trans2str(lod_level)

    def _is_same(self, values):
        value_0 = values[0]
        for i in range(len(values)):
            if value_0 != values[i]:
                return False
        return True

    def to_string(self):
        if self.type == "Tensor":
            return self.name + " (Tensor) - dtype: " + str(
                self.dtype) + ", shape: " + str(self.shape)
        
