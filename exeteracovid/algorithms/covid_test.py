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

import numpy as np
import exetera.core.dataframe as edf

class ValidateCovidTestResultsFacVersion1PreHCTFix:
    """
    Deprecated, please use covid_test.ValidateCovidTestResultsFacVersion2().
    """
    def __init__(self, hcts, tcps, filter_status, hct_results, results, filter_flag, show_debug=False):
        self.valid_transitions = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 2), 3: (0, 3)}
        self.upgrades = {0: (0, 1, 2, 3), 1: (2, 3), 2: tuple(), 3: tuple()}
        self.hcts = hcts
        self.tcps = tcps
        self.hct_results = hct_results
        self.results = results
        self.filter_status = filter_status
        self.filter_flag = filter_flag
        self.show_debug = show_debug

    def __call__(self, patient_id, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = 0
        for j in range(start, end + 1):
            # allowable transitions
            value = self.tcps[j]
            if value not in self.valid_transitions[max_value]:
                invalid = True
                break
            if value in self.upgrades[max_value]:
                max_value = value
            self.results[j] = max_value

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.tcps[j]
                filter_status[j] |= self.filter_flag

        self.hct_results[start:end+1] = self.hcts[start:end+1]
        if invalid:
            for j in range(start, end + 1):
                filter_status[j] |= self.filter_flag

        if self.show_debug == True:
            if invalid or not np.array_equal(self.hcts[start:end+1], self.hct_results[start:end+1])\
               or not np.array_equal(self.tcps[start:end+1], self.results[start:end+1]):
                reason = 'inv' if invalid else 'diff'
                print(reason, start, 'hct:', self.hcts[start:end+1], self.hct_results[start:end+1])
                print(reason, start, 'tcp:', self.tcps[start:end+1], self.results[start:end+1])

        # TODO: remove before v0.1.8
        # for j in range(start, end + 1):
        #     self.hct_results[j] = self.hcts[j]

        # if invalid:
        #     for j in range(start, end + 1):
        #         if self.hct_results[j] == 1 and self.results[j] != 0:
        #             print('hct:', start, self.hcts[start:end+1], self.hct_results[start:end+1])
        #             print('tcp:', start, self.tcps[start:end+1], self.results[start:end+1])
        #             break


class ValidateCovidTestResultsFacVersion1:
    """
    Deprecated, please use covid_test.ValidateCovidTestResultsFacVersion2().
    """
    def __init__(self, hcts, tcps, filter_status, hct_results, results, filter_flag, show_debug=False):
        self.valid_transitions = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 2), 3: (0, 3)}
        self.upgrades = {0: (0, 1, 2, 3), 1: (2, 3), 2: tuple(), 3: tuple()}
        self.hcts = hcts
        self.tcps = tcps
        self.hct_results = hct_results
        self.results = results
        self.filter_status = filter_status
        self.filter_flag = filter_flag
        self.show_debug = show_debug

    def __call__(self, patient_id, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = 0
        for j in range(start, end + 1):
            # allowable transitions
            value = self.tcps[j]
            if value not in self.valid_transitions[max_value]:
                invalid = True
                break
            if value in self.upgrades[max_value]:
                max_value = value
            self.results[j] = max_value

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.tcps[j]
                filter_status[j] |= self.filter_flag

        if not invalid:
            first_hct_false = -1
            first_hct_true = -1
            for j in range(start, end + 1):
                if self.hcts[j] == 1:
                    if first_hct_false == -1:
                        first_hct_false = j
                elif self.hcts[j] == 2:
                    if first_hct_true == -1:
                        first_hct_true = j

            max_value = 0
            for j in range(start, end + 1):
                if j == first_hct_false:
                    max_value = max(max_value, 1)
                if j == first_hct_true:
                    max_value = 2

                if self.hct_results is None or self.hct_results[j] is None:
                    print(j, self.hct_results[j], max_value)
                self.hct_results[j] = max_value
        else:
            for j in range(start, end + 1):
                self.hct_results[j] = self.hcts[j]
                filter_status[j] |= self.filter_flag

        if self.show_debug == True:
            if invalid or not np.array_equal(self.hcts[start:end+1], self.hct_results[start:end+1])\
               or not np.array_equal(self.tcps[start:end+1], self.results[start:end+1]):
                reason = 'inv' if invalid else 'diff'
                print(reason, start, 'hct:', self.hcts[start:end+1], self.hct_results[start:end+1])
                print(reason, start, 'tcp:', self.tcps[start:end+1], self.results[start:end+1])

        # TODO: remove before v0.1.8
        # for j in range(start, end + 1):
        #     self.hct_results[j] = self.hcts[j]

        # if invalid:
        #     for j in range(start, end + 1):
        #         if self.hct_results[j] == 1 and self.results[j] != 0:
        #             print('hct:', start, self.hcts[start:end+1], self.hct_results[start:end+1])
        #             print('tcp:', start, self.tcps[start:end+1], self.results[start:end+1])
        #             break


class ValidateCovidTestResultsFacVersion2:
    def __init__(self, hcts, tcps, filter_status, hct_results, results, filter_flag, show_debug=False):
        """
        Check if the test result reported by users logical, for example if multiple test results reported, then should
            follow a sequence from empty to yes or no, not the other way around.

        :param htcs: not used.
        :param tcps: The test results column.
        :param filter_status: A field to indicates invalid test results.
        :param hct_results: Not used.
        :param results: An intermedia array to store if the user ever reported positive tests.
        :param filter_flag: A flag to filter invalid tests.
        :param show_debug: A flag to switch debugging info, such as the status of test results.
        """
        self.valid_transitions = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 2), 3: (0, 3)}
        self.valid_transitions_before_yes =\
            {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 1, 2, 3), 3: (0, 3)}
        self.upgrades = {0: (0, 1, 2, 3), 1: (2, 3), 2: tuple(), 3: tuple()}
        self.upgrades_before_yes = {0: (1, 2, 3), 1: (3,), 2: (1, 3), 3: tuple()}

        self.tcps = tcps
        self.results = results
        self.filter_status = filter_status
        self.filter_flag = filter_flag
        self.show_debug = show_debug

    def __call__(self, patient_id, filter_status, start, end):
        """
        Perform the result check.

        :param patient_id: Not used.
        :param filter_status: A field to indicates invalid test results.
        :param start: Start position of the rows to perform the check.
        :param end: End position of the rows to perform the check.
        """
        # validate the subrange
        invalid = False
        max_value = 0
        first_waiting = -1
        first_no = -1
        first_yes = -1

        for j in range(start, end + 1):
            value = self.tcps[j]
            if value == 'waiting' and first_waiting == -1:
                first_waiting = j
            if value == 'no' and first_no == -1:
                first_no = j
            if value == 'yes' and first_yes == -1:
                first_yes = j

        for j in range(start, end + 1):
            valid_transitions = self.valid_transitions_before_yes if j <= first_yes else self.valid_transitions
            upgrades = self.upgrades_before_yes if j <= first_yes else self.upgrades
            # allowable transitions
            value = self.tcps[j]
            if value not in valid_transitions[max_value]:
                invalid = True
                break
            if j < first_yes and value == 'no':
                value == 'waiting'
            if value in upgrades[max_value]:
                max_value = value
            self.results[j] = max_value

        #rescue na -> waiting -> no -> waiting
        if invalid and first_yes == -1 and self.tcps[end] == 'waiting':
            invalid = False
            max_value = ''
            for j in range(start, end+1):
                value = self.tcps[j]
                if max_value == '' and value != '':
                    max_value = 'waiting'
                self.results[j] = max_value

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.tcps[j]
                filter_status[j] |= self.filter_flag
            if self.show_debug:
                print(self.tcps[j], end=': ')
                for j in range(start, end + 1):
                    if j > start:
                        print(' ->', end=' ')
                    value = self.tcps[j]
                    print('na' if value == '' else value, end='')
                print('')

def match_assessment_v1(test_df, assessment_df, dest_df, gap, positive_only=False):
    """
    Mapping the test with a previous assessment record.

    :param test_df: The tests dataframe.
    :param assessment_df: The assessment dataframe.
    :param dest_df: The destination dataframe to write the result to.
    :param gap: Limit the number of days assessment prior the tests.
    :param positive_only: Filter tests with positive result only.
    :return: The result dataframe.
    """
    # merge tests with assessment
    edf.merge(test_df, assessment_df, dest=dest_df, left_on='patient_id', right_on='patient_id', how='inner')
    # created_at_l test date, created_at_r assessment date
    flt = dest_df['created_at_r'] <= dest_df['created_at_l']  # assessment happens before test
    flt &= dest_df['created_at_r'] + gap * 3600 * 24 >= dest_df['created_at_l']  # asmt < test < asmt + gap
    if positive_only:
        flt &= dest_df['result'] == 4
    dest_df.apply_filter(flt)
    return dest_df


#TODO use numba
def unique_tests_v1(src_test,
                   fields=('patient_id', 'result', 'pcr_standard', 'date_effective_test')):
    """"
    Group tests per patient. Use unique dates & mechanism per patient.

    :param src_test: The tests dataframe.
    :return: A numpy array of boolean identifying unique rows.
    """
    pat_id = src_test['patient_id']
    print('creating spans')
    spans = pat_id.get_spans()
    span_start = spans[0:-1]
    span_end = spans[1:]
    clean_filt = np.zeros(len(pat_id), dtype='bool')
    dict_fields = {}
    print(clean_filt.sum())
    for f in fields:
        print(f)
        dict_fields[f] = src_test[f].data[:]
    dict_fields['date_eff'] = np.where(dict_fields['date_taken_specific'] > 0,
                                       dict_fields['date_taken_specific'],
                                       dict_fields['date_taken_between_start'])

    print('dict done')
    date_eff = dict_fields['date_eff']
    mechanism = dict_fields['mechanism']
    pat_tmp = dict_fields['patient_id']
    count_pat = 0
    print('starting loop')
    for i in range(len(span_start)):
        i_s = span_start[i]
        i_e = span_end[i]

        if i_e - i_s == 1:
            clean_filt[i_s] = 1
            count_pat += 1
            # print(np.sum(clean_filt), 'from unique')
        else:  # found unique date & mechanism
            num_pat = len(np.unique(pat_tmp[i_s:i_e]))
            if num_pat > 1:
                print('number of patid', num_pat)
            possible_dates = np.unique(date_eff[i_s:i_e])
            possible_mechanisms = np.unique(mechanism[i_s:i_e])
            current = 1
            for d in possible_dates:
                for m in possible_mechanisms:
                    found = False
                    # print(d,m)
                    for j in reversed(range(i_s, i_e)):
                        # print(i_s, d, m, date_eff[j], mechanism[j])
                        if found is True:  # TODO dead code
                            clean_filt[j] = 0
                            continue
                        if date_eff[j] == d and mechanism[j] == m:
                            found = True
                            # print(d,m)
                            clean_filt[j] = 1
                            break
                    # print(np.sum(clean_filt))
            count_pat += 1
    print(count_pat)
    return clean_filt

def multiple_tests_start_with_negative_v1(s, src_asmt):
    """

    :param s: The session instance.
    :param src_asmt: The assessment dataframe.
    :return: The filter indicates patient that has multiple tests with first negative and following positive tests.
    """
    # Remap had_covid_test to 0/1 2 to binary 0,1
    tcp_flat = np.where(src_asmt['tested_covid_positive'].data[:] < 1, 0, 1)
    spans = src_asmt['patient_id'].get_spans()
    # Get the first index at which the hct field is maximum
    firstnz_tcp_ind = s.apply_spans_index_of_max(spans, tcp_flat)
    # Get the index of first element of patient_id when sorted
    first_hct_ind = spans[:-1]
    filt_tl = first_hct_ind != firstnz_tcp_ind
    sel_max_ind = s.apply_filter(filter_to_apply=filt_tl, reader=firstnz_tcp_ind)

    return sel_max_ind
