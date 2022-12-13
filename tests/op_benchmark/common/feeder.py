# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import numpy as np




def copy_feed_spec(feed_spec):
    if feed_spec is None:
        return None
    if not isinstance(feed_spec, list):
        feed_spec = [feed_spec]

    copy = []
    for feed_item in feed_spec:
        assert isinstance(feed_item, dict)
        item_copy = {}
        for key, value in feed_item.items():
            item_copy[key] = value
        copy.append(item_copy)
    return copy


def check_shape_and_dtype(shape, dtype, value):
    assert list(shape) == list(value.shape) or list(shape) + [
        1
    ] == list(value.shape) or list(shape) == list(
        value.shape) + [1], "Expected shape: %s. Recieved shape: %s." % (
            shape, value.shape)
    value = value.reshape(shape)

    # Allow different data type
    if dtype != value.dtype:
        value = value.astype(dtype)

    return value


def generate_random_data(shape, dtype, value=None):
    if value is not None:
        if isinstance(value, list):
            value = np.array(value)
        assert isinstance(
            value, np.ndarray
        ), "Expected value's type to be numpy.ndarray, but recieved {}.".format(
            type(value))
        data = check_shape_and_dtype(shape, dtype, value)
    else:
        if dtype == "int64" or dtype == "int32":
            data = np.random.randint(100, size=shape, dtype=dtype)
            if range is not None:
                data = np.random.randint(
                    range[0], range[1], size=shape, dtype=dtype)
        elif dtype == "bool":
            data = np.random.randint(2, size=shape, dtype=bool)
        elif dtype == "uint8" or dtype == "uint16":
            data = np.random.randint(0, 100, size=shape, dtype=dtype)
        else:
            data = np.random.random(shape).astype(dtype)
    return data


class FeederAdapter(object):
    def __init__(self, framework, feed_spec, feed_list):
        assert framework in ["paddle", "tensorflow", "pytorch"]
        if feed_spec is not None:
            assert len(feed_list) == len(
                feed_spec
            ), "Expected the number of feeding vars (%d) to be equal to the length of feed_spec (%d)." % (
                len(feed_list), len(feed_spec))

        self.__framework = framework
        self.__feed_spec = copy_feed_spec(feed_spec)
        self.__feed_list = feed_list

    

    def to_pytorch(self, feed_vars=None):
        target_framework = "pytorch"
        if self.__framework == target_framework:
            return self.__feed_list
        else:
            return self._to_other(target_framework, feed_vars)

    def _to_other(self, target_framework, feed_vars=None):
        assert self.__framework == "paddle"
        assert isinstance(feed_vars, list)
        assert len(feed_vars) == len(self.__feed_list)

        feed_list = []
        for i in range(len(feed_list)):
            value = self.__feed_list[i]

            if self.__feed_spec is not None and self.__feed_spec[i].get(
                    "permute", None) is not None:
                permute_paddle2other = self.__feed_spec[i]["permute"]
                value = np.transpose(value, permute_paddle2other)

            # On dynamic mode, the check is skipped.
            if feed_vars is not None:
                # Check shape and dtype
                var = feed_list[i]
                var_shape = var.shape
                if target_framework == "tensorflow":
                    var_dtype = _convert_tensorflow_dtype(
                        var.dtype, to_string=True)
                value = check_shape_and_dtype(var_shape, var_dtype, value)

            feed_list.append(value)
        return feed_list

    @property
    def framework(self):
        return self.__framework
