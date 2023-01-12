import os
import json
import datetime
from utils import *
def scan_directory():
    op_name_list = []
    op_cuda_list = []
    out_path = json_file_path = os.path.join(os.path.join(os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "tests"), "output"))
    config_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "config")
    print(config_path)
    out_op_name_list = os.listdir(out_path)
    for i in range(len(out_op_name_list)):
        dir = out_op_name_list[i]
        if dir.startswith("out_"):
            op_name = dir[4:]
            op_name_list.append(op_name)
            config_name = os.path.join(config_path, op_name)+".json"
            with open(config_name, 'r') as f:
                data = json.load(f)
                op_cuda = data[0]["cuda_name"]
                op_cuda_list.append(op_cuda)
            f.close
    return op_name_list, op_cuda_list

if __name__ == "__main__":
    op_name_list, op_cuda_list = scan_directory()
    assert len(op_name_list) == len(op_cuda_list), "length can't be different between op_name_list and op_cuda list"
    perf_dict = collections.OrderedDict()
    perf_dict["exec_time"] = str(datetime.datetime.now())
    result = True
    flag = True
    for i in range(len(op_name_list)):
        op_name = op_name_list[i]
        op_cuda = op_cuda_list[i]
        base_agt = extract_data(op_name=op_name, op_cuda=op_cuda, path="baseline")
        test_agt = extract_data(op_name=op_name, op_cuda=op_cuda, path="tests")
        if test_agt > base_agt:
            result = False
            flag = False
        else:
            result = True
        perf_dict[op_name] = collections.OrderedDict()
        perf_dict[op_name]["avg_gpu_time_per"] = (base_agt-test_agt)/base_agt
        perf_dict[op_name]["pass"] = result
    with open ("../result/perf.json", "w") as f:
        json.dump(perf_dict, f)
    f.close
    assert flag is True, "op performance test failed"
