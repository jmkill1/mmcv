import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def use_gpu():
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""

def is_string(value):
    return isinstance(value, str)

def parse_string(value):
    import six
    # PY2     : PY3
    # unicode : str
    # str     : bytes
    if six.PY3:
        return value
    else:
        return value.encode("utf-8") if isinstance(value, str) else value

def parse_int(value):
    if isinstance(value, int):
        return value
    else:
        return int(parse_string(value))
    
def parse_list(value_str, sub_dtype="int"):
    value_str = parse_string(value_str)
    if sub_dtype in ["int", "int64"]:
        try:
            if value_str != "[]":
                value_str_list = value_str.replace("L", "").replace(
                    "[", "").replace("]", "").split(',')
                value_list = []
                for item in value_str_list:
                    value_list.append(int(item))
                return value_list
            else:
                return []
        except Exception as e:
            assert False, "Parse {} failed: {}".format(value_str, e)
    else:
        # TODO: check and support list of other data type.
        raise ValueError("Do not support parsing list of non-int data type.")


def parse_tuple(value_str, sub_dtype="int"):
    value_str = parse_string(value_str)
    if sub_dtype in ["int", "int64"]:
        try:
            if value_str != "()":
                value_str_list = value_str.replace("L", "").replace(
                    "(", "").replace(")", "").split(',')
                value_list = []
                for item in value_str_list:
                    value_list.append(int(item))
                return value_list
            else:
                return []
        except Exception as e:
            assert False, "Parse {} failed: {}".format(value_str, e)
    else:
        # TODO: check and support list of other data type.
        raise ValueError("Do not support parsing list of non-int data type.")