# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from exetera.core.utils import valid_range_fac_inc
from exeteracovid.algorithms.age_from_year_of_birth import CalculateAgeFromYearOfBirth, calculate_age_from_year_of_birth_v1


class TestAgeFromYearOfBirth(unittest.TestCase):

    def test_age_from_year_of_birth(self):

        yobs = np.asarray([1, 10, 100, 1000, 1900, 1910, 1920, 1930, 1940, # too old
                           1950, 1960, 1980, 2000, 2004,
                           2010, 2019, 2020, 2030 # too young
                          ], dtype=np.int32)
        yob_flags = np.ones_like(yobs, dtype=np.bool)

        ages = np.zeros(len(yobs), dtype=np.int32)
        age_flags = np.ones_like(ages, dtype=np.bool)
        age_range_flags = np.ones_like(ages, dtype=np.bool)

        calculate_age_from_year_of_birth_v1(
            yobs, yob_flags, 16, 90,
            ages, age_flags, age_range_flags,
            2020)

        expected_ages = np.asarray([2019, 2010, 1920, 1020, 120, 110, 100, 90, 80, 70, 60, 40, 20, 16, 10, 1, 0, -10],
                                   dtype=np.int32)
        self.assertListEqual(expected_ages.tolist(), ages.tolist())
        expected_age_flags = np.ones(len(expected_ages), dtype=np.bool)
        self.assertListEqual(expected_age_flags.tolist(), age_flags.tolist())
        expected_age_range_flags = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.bool)
        self.assertListEqual(expected_age_range_flags.tolist(), age_range_flags.tolist())