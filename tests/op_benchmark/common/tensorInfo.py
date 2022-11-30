from common.utils import *
from common.attrsinfo import AttrsInfo


class TensorInfo(AttrsInfo):
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
        if self.type == "Variable":
            return self.name + " (Variable) - dtype: " + str(
                self.dtype) + ", shape: " + str(self.shape)
        elif self.type == "list<Variable>":
            str_list = "%s (list<Variable>[%d]) - " % (self.name,
                                                       len(self.dtype))
            if self._is_same(self.dtype) and self._is_same(self.shape):
                params_len = 1
            else:
                params_len = len(self.dtype)
            for i in range(params_len):
                str_list = str_list + "dtype: " + str(self.dtype[
                    i]) + ", shape: " + str(self.shape[i]) + "; "
            return str_list
