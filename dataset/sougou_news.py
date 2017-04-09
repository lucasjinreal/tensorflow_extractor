# Copyright 2017 Jin Fagang. All Rights Reserved.
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
# =======================================================================
import numpy as np


class SougouNews(object):

    def __init__(self):
        pass

    @staticmethod
    def batch_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
        if length_from > length_to:
            raise ValueError('length_from > length_to')

        def random_length():
            if length_from == length_to:
                return length_from
            return np.random.randint(length_from, length_to + 1)

        while True:
            yield [
                np.random.randint(low=vocab_lower,
                                  high=vocab_upper,
                                  size=random_length()).tolist()
                for _ in range(batch_size)
            ]


