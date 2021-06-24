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
from exetera.core.utils import check_input_lengths


class CheckTestingConsistency:
    """
    Checking consistency over patient's 'had_covid_test' field from assessments dataframe, with 'result' field from tests
        dataframe, mark inconsistent patients.
    """
    def __init__(self, f_inconsistent_not_tested, f_inconsistent_tested):
        """
        To initialize the inconsistency check.

        :param f_inconsistent_not_tested: A array of patients that 'had_covid_test' set as False, but has 'result' set.
        :param f_inconsistent_tested: A array of patients that 'had_covid_test' set as True, but has no 'result' set.
        """
        self.f_inconsistent_not_tested = f_inconsistent_not_tested
        self.f_inconsistent_tested = f_inconsistent_tested

    def __call__(self, had_test, test_result, flags):
        """
        To perform the inconsistency check.

        :param had_test: The 'had_covid_test' column from assessments dataframe.
        :param test_result: The 'result' result column from tests dataframe.
        :param flags: The flag identify inconsistency.
        """
        # if len(had_test) != len(test_result):
        #     error_str = "'had_test' (length {} must be the same length as 'test_result' (length {})"
        #     raise ValueError(error_str.format(len(had_test), len(test_result)))
        # if len(had_test) != len(flags):
        #     error_str = "'had_test' (length {} must be the same length as 'flags' (length {})"
        #     raise ValueError(error_str.format(len(had_test), len(flags)))
        check_input_lengths(('had_test', 'test_result', 'flags'), (had_test, test_result, flags))

        for i_r in range(len(had_test)):
            ht = had_test[i_r]
            tr = test_result[i_r]
            if ht != 2 and tr != 0:
                flags[i_r] |= self.f_inconsistent_not_tested
            if ht == 2 and tr == 0:
                flags[i_r] |= self.f_inconsistent_tested
