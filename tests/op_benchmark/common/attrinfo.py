from common.utils import *
class AttrInfo(object):
    def __init__(self, name, type, value):
        self.name = self.trans2str(name)
        self.type = self.trans2str(type)
        self.value = self.trans2val(self.trans2str(value))

    def trans2str(self, item):
        if isinstance(item, list):
            item_str = []
            for ele in item:
                item_str.append(parse_string(ele))
            return item_str
        else:
            return parse_string(item)

    def to_string(self):
        return self.name + ' (' + self.type + '): ' + str(self.value)

    def trans2val(self, value_str):
        if self.type in ["float", "float32", "float64"]:
            return float(value_str)
        elif self.type in ["int", "int32", "int64", "long"]:
            return int(value_str)
        elif self.type == "bool":
            return eval(value_str)
        elif self.type in ["string", "str"]:
            return None if value_str == "None" else value_str
        elif self.type == "list":
            return parse_list(value_str, sub_dtype="int")
        elif self.type == "tuple":
            return parse_tuple(value_str, sub_dtype="int")
        else:
            raise ValueError("Unsupported type \"%s\" for %s, value is %s." %
                             (self.type, self.name, value_str))
