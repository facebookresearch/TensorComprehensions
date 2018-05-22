# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

import unittest
from tensor_comprehensions.tc import CudaMappingOptions


class TestOptions(unittest.TestCase):
    def test_options(self):
        print('\nCreating mapping_options')
        options =CudaMappingOptions("naive")
        options.useSharedMemory(True)
        options.unrollCopyShared(False)
        options.mapToBlocks([256, 8])
        options.mapToThreads([4, 16, 4])
        options.tile([2, 8, 64, 128])
        options.unroll(128)
        options.fixParametersBeforeScheduling(False)
        options.scheduleFusionStrategy("Max")
        options.outerScheduleFusionStrategy("Preserve3Coincident")
        print('Mapping options created successfully')


if __name__ == '__main__':
    unittest.main()
