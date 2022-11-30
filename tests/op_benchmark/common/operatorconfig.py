import os
import json
from operator import attrgetter
from common.utils import *
from common.attrsinfo import AttrsInfo
from common.tensorInfo import TensorInfo

class OperatorConfig(object):
    def __init__(self, op_name):
        self.op_name = op_name
        self.inputs = None
        self.tensor_list = None
        self.attr_list = None
        self.backward = False
        self.repeat_time = None
        self.case_name = None

    def get_dtype(self):
        dtype = None
        for name, value in vars(self).items():
            # float16 is not supported for CPU.
            if name.endswith("_dtype"):
                if value == "float16":
                    dtype = "float16"
                elif dtype is not None:
                    dtype = value
        return dtype

    def disabled(self):
        use_gpu = use_gpu()
        if not use_gpu and self.get_dtype() == "float16":
            print(
                "Warning:\n"
                "  1. This config is disabled because float16 is not supported for %s on CPU.\n"
                % (self.op_name))
            return True
        return False

    def init_config(self, config_id=0, unknown_dim=16):
        json_file_path = os.path.join(os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "config"), self.op_name)+".json"
        print("---- Initialize OperatorConfig from %s" %
              (json_file_path))
        json_file_name = os.path.split(json_file_path)[-1]
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            operator = data[config_id]["operator"]
            self.case_name = data[config_id]["case_name"]
            assert operator == self.op_name, "(%s): The op type (%s) in json file is different from the name (%s). " \
                "The filename: %s, config_id: %d." % (
                    self.case_name, operator, self.op_name, json_file_name, config_id)

            self.inputs = data[config_id]["inputs"]

            if data[config_id].get("repeat_time", None) is not None:
                self.repeat = parse_int(data[config_id]["repeat_time"])

        self._parse_inputs()
        for param in self.attrs_list:
            setattr(self, param.name, param.value)
        for var in self.tensor_list:
            for i in range(len(var.shape)):
                if var.shape[i] == -1:
                    var.shape[i] = unknown_dim
            setattr(self, var.name + '_shape', var.shape)
            setattr(self, var.name + '_dtype', var.dtype)

        if not hasattr(self, "atol"):
            self.atol = 1e-3 if self.get_dtype() == "float16" else 1e-6
        return self

    def _parse_inputs(self):
        self.tensor_list = []
        self.attrs_list = []
        if self.inputs is None:
            self.tensor_list = None
            self.attrs_list = None
        else:
            for name, value in self.inputs.items():
                assert value.get("type", None) is not None
                if value["type"] == "Tensor":
                    tensor_info = TensorInfo(name, value["type"], value["dtype"],
                                        value["shape"])
                    self.tensor_list.append(tensor_info)
                else:
                    attrs_info = AttrsInfo(name, value["type"], value["value"])
                    self.attrs_list.append(attrs_info)

    def convert_to_fp16(self):
        """
        Convert all variables' dtype to float16.
        """

        def _is_floating_point_type(dtype):
            if dtype in ["float", "float32", "double", "float64"]:
                return True
            return False

        for var in self.tensor_list:
            if var.type == "Variable" and _is_floating_point_type(var.dtype):
                var.dtype = "float16"
                setattr(self, var.name + "_dtype", var.dtype)
            elif var.type == "list<Variable>":
                # convert each list member in list<Variable> from fp32 into fp16 type
                for i in range(len(var.dtype)):
                    if _is_floating_point_type(var.dtype[i]):
                        var.dtype[i] = "float16"
                setattr(self, var.name + "_dtype", var.dtype)
