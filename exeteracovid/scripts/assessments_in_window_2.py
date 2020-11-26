from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import h5py

from exetera.core.session import Session
from exetera.core.utils import Timer, build_histogram
from exetera.core.persistence import foreign_key_is_in_primary_key
from exetera.core.operations import is_ordered
from exetera.core.validation import _check_equal_length
from exeteracovid.algorithms.test_type_from_mechanism import test_type_from_mechanism_v1



class PatientTests:
    def __init__(self):
        self.tests = list()

    def append(self, test):
        self.tests.append(test)


class Test:
    def __init__(self, tid, ts, pos):
        self.test_id = tid
        self.timestamp = ts
        self.positive = pos


def get_test_dates(ts_exact, ts_start, ts_end):
    return np.where(ts_exact != 0, ts_exact, (ts_end + ts_start) / 2)


def get_proximal_test_filter(s, test_pids, test_ts, test_res, window):
    if len(set([len(test_pids), len(test_ts), len(test_res)])) > 1:
        msg = "test_pids, test_ts, and test_res should be equal_length but are {}, {}, and {}"
        raise ValueError(msg.format(len(test_pids), len(test_ts), len(test_res)))

    remaining = np.ones(len(test_ts), dtype=np.bool)

    t_spans = s.get_spans(test_pids)
    print("t_spans:", len(t_spans))

    for i in range(len(t_spans) - 1):
        sps, spe = t_spans[i:i + 2]
        ptnt_test_ts = test_ts[sps:spe]
        test_order = np.argsort(ptnt_test_ts)[::-1]

        cur_ts = test_ts[sps + test_order[0]]
        for j in test_order[1:]:
            prev_ts = test_ts[sps + j]
            if cur_ts - prev_ts < 86400 * window:
                remaining[sps + j] = False
            else:
                cur_ts = prev_ts

    return remaining

def get_antibody_and_related_pcr_test(s, test_pids, test_atb, test_pcr, test_ts, test_res, window):
    _check_equal_length('test_pids', test_pids, 'test_atb', test_atb)
    _check_equal_length('test_pids', test_pids, 'test_pcr', test_pcr)
    _check_equal_length('test_pids', test_pids, 'test_ts', test_ts)
    _check_equal_length('test_pids', test_pids, 'test_res', test_res)

    test_pcr_res = np.zeros_like(test_res)
    t_spans=s.get_spans(test_pids)   # all tests for one pid grouped together
    for i in range(len(t_spans) - 1):
        sps, spe = t_spans[i], t_spans[i + 1]
        # test set of timestamps for a specific pid
        ptnt_test_ts = test_ts[sps:spe]
        test_order = np.argsort(ptnt_test_ts)[::-1]   # ordering of the timestamps; -1 is from newest to oldest

        for j in range(len(test_order)):
            j_idx = sps + test_order[j]   # global index in the collection
            if test_atb[j_idx]:
                j_ts = test_ts[j_idx]
                pcr_res = 0
                for k in range(j+1, len(test_order)):
                    # going through all the tests and finding AB+/- and finding all the pcr tests X days back
                    k_idx = sps + test_order[k]
                    if test_pcr[k_idx]:
                        k_ts = test_ts[k_idx]
                        if j_ts - k_ts > 86400*window:
                            break
                        pcr_res = max(pcr_res, test_res[k_idx])   # storing the positive result for pcr
                test_pcr_res[j_idx] = pcr_res
    return test_pcr_res


def find_earliest_match(s, test_ts, asmt_ts, symptom, symptom_threshold, window, test_symptom_delta):
    _check_equal_length('test_ts', test_ts, 'tests_symptom_delta', test_symptom_delta)
    _check_equal_length('asmt_ts', asmt_ts, 'symptom', symptom)

    def _inner(test_start, test_end, asmt_start, asmt_end):
        for t in range(test_start, test_end):
            cur_test_ts = test_ts[t]

            symptom_to_test_delta = 0
            for a in range(asmt_start, asmt_end):
                cur_asmt_ts = asmt_ts[a]
                delta = cur_test_ts - cur_asmt_ts
                if delta < 0:
                    break
                if symptom[a] >= symptom_threshold:
                    if delta < 86400 * window:
                        symptom_to_test_delta = delta
                        break
            test_symptom_delta[t] = symptom_to_test_delta

    return _inner


def iterate_over_matching_subsets(s, left_keys, right_keys, pred, needs_both=True):
    # run through all tests for one pid.
    # for each test, we find the earliest assessment with one symptom reported
    if not is_ordered(left_keys):
        raise ValueError("'left_keys' must be in sorted order")
    if not is_ordered(right_keys):
        raise ValueError("'right_keys' must be in sorted order")

    # iterate over spans
    left_spans = s.get_spans(left_keys)
    right_spans = s.get_spans(right_keys)

    left_count = len(left_spans) - 1
    right_count = len(right_spans) - 1

    l_i = 0
    r_i = 0
    while l_i < left_count and r_i < right_count:
        l_id = left_spans[l_i]
        r_id = right_spans[r_i]
        l_k = left_keys[l_id]
        r_k = right_keys[r_id]
        if l_k < r_k:
            if not needs_both:
                pred(left_spans[l_i], left_spans[l_i+1], None, None)
            l_i += 1
        elif l_k > r_k:
            if not needs_both:
                pred(None, None, right_spans[r_i], right_spans[r_i+1])
            r_i += 1
        else:
            pred(left_spans[l_i], left_spans[l_i+1], right_spans[r_i], right_spans[r_i+1])
            l_i += 1
            r_i += 1


def filter_and_generate_consolidated_mappings(s, src, dest, start_ts, end_ts,
                                              window, patient_fields, assessment_fields, test_fields):
    s_tests = src['tests']
    s_asmts = src['assessments']
    s_ptnts = src['patients']
    print(s_tests.keys())

    # initial_patient_filtering
    # -------------------------

    with Timer("getting patient fields"):
        print([k for k in s_ptnts.keys() if '_to_' in k])
        p_ids = s.get(s_ptnts['id']).data[:]
        p_ages = s.get(s_ptnts['16_to_90_years']).data[:]
        p_gndrs = s.get(s_ptnts['gender']).data[:]
        p_bmis = s.get(s_ptnts['15_to_55_bmi']).data[:]
        p_asmt_counts = s.get(s_ptnts['assessment_count']).data[:]
        patient_filter = p_ages & p_bmis & (p_asmt_counts > 0) & (p_gndrs > 0) & (p_gndrs < 3)
    print("patient_filter:", patient_filter.sum(), len(patient_filter))

    valid_patients = set(p_ids[patient_filter])

    # initial test filtering
    # ----------------------

    with Timer('getting test fields'):
        t_pids = s.get(s_tests['patient_id']).data[:]
        t_cats = s.get(s_tests['created_at']).data[:]
        t_rslts = s.get(s_tests['result']).data[:]

    # keep only tests for patients who have pass the patient filter, above
    t_patient_filter = np.zeros(len(t_pids), dtype=np.bool)
    for i in range(len(t_patient_filter)):
        if t_pids[i] in valid_patients:
            t_patient_filter[i] = True

    with Timer('filtering tests by date'):
        t_date_filter = (t_cats >= start_ts) & (t_cats < end_ts)
    print("t_date_filter:", t_date_filter.sum(), len(t_date_filter))

    with Timer('filtering tests by result'):
        t_result_filter = t_rslts > 2
    print("t_result_filter:", t_result_filter.sum(), len(t_result_filter))

    with Timer('getting test mechanism filter for pcr and antibody', new_line=True):
        pcr1 = np.zeros(len(t_pids), dtype=np.bool)
        pcr2 = np.zeros(len(t_pids), dtype=np.bool)
        pcr3 = np.zeros(len(t_pids), dtype=np.bool)
        atb1 = np.zeros(len(t_pids), dtype=np.bool)
        atb2 = np.zeros(len(t_pids), dtype=np.bool)
        atb3 = np.zeros(len(t_pids), dtype=np.bool)

        test_type_from_mechanism_v1(s, s.get(s_tests['mechanism']), s.get(s_tests['mechanism_freetext']),
                                    pcr1, pcr2, pcr3, atb1, atb2, atb3)
        # everyone who has pcr tests
        t_pcr_filter = pcr1 | pcr2

        # everyone who has atb tests
        t_atb_filter = atb1 | atb2

        # everyone who has positive or negative antibody result
        t_atb_result_filter = t_result_filter & t_atb_filter
        print("t_atb_result_filter:", t_atb_result_filter.sum(), len(t_atb_result_filter))

    t_test_filter = t_patient_filter & t_date_filter & t_atb_result_filter
    print("t_test_filter:", t_test_filter.sum(), len(t_test_filter))

    t_tats = get_test_dates(s.get(s_tests['date_taken_specific']).data[:],
                            s.get(s_tests['date_taken_between_start']).data[:],
                            s.get(s_tests['date_taken_between_end']).data[:])

    t_pcr_res = get_antibody_and_related_pcr_test(s, t_pids, t_atb_filter, t_pcr_filter, t_tats, t_rslts, window)

    t_pids = s.apply_filter(t_test_filter, t_pids)
    t_tats = s.apply_filter(t_test_filter, t_tats)
    t_rslts = s.apply_filter(t_test_filter, t_rslts)
    t_pcr_res = s.apply_filter(t_test_filter, t_pcr_res)

    t_pos = t_rslts == 4


    # initial assessment filtering
    # ----------------------------

    with Timer("getting asmt fields"):
        a_pids = s.get(s_asmts['patient_id']).data[:]
        a_cats = s.get(s_asmts['created_at']).data[:]

    # keep only assessments for patients who have pass the patient filter, above
    # a_patient_filter = np.zeros(len(a_pids), dtype=np.bool)
    # for i in range(len(a_patient_filter)):
    #     if a_pids[i] in valid_patients:
    #         a_patient_filter[i] = True

    # we retain assessments who still have tests / patients in the list of tests
    a_patient_filter = foreign_key_is_in_primary_key(t_pids, a_pids)
    print('assessment patient', a_patient_filter.sum(), len(a_patient_filter))

    with Timer("filtering asmts by date"):
        a_date_filter = (a_cats >= (start_ts - 86400 * window)) & (a_cats < end_ts)
    print("a_date_filter:", a_date_filter.sum(), len(a_date_filter))

    assessment_filter = a_patient_filter & a_date_filter
    print("a_assessment_filter:", assessment_filter.sum(), len(assessment_filter))

    a_ids = s.apply_filter(assessment_filter, s.get(s_asmts['id']).data[:])
    a_pids = s.apply_filter(assessment_filter, a_pids)
    a_cats = s.apply_filter(assessment_filter, a_cats)

    # iterate over the assessments relevant to each test to find the earliest time point (within the window) that
    # each symptom starts showing
    # ---------------------------

    d_tests = dest.create_group('tests')
    # find the delta between the start of relevant symptoms and the test
    for k in symptoms:
        src_symptom_f = s.get(s_asmts[k])
        print(k, src_symptom_f.keys)
        with Timer("getting symptom deltas for {}".format(k)):
            symptom_threshold = custom_symptom_thresholds.get(k, 2)
            t_symptom_deltas = np.zeros(len(t_pids), dtype=np.float32)
            op = find_earliest_match(s, t_tats, a_cats,
                                     s.apply_filter(assessment_filter, src_symptom_f.data[:]),
                                     symptom_threshold, window, t_symptom_deltas)
            iterate_over_matching_subsets(s, t_pids, a_pids, op)
            print("t_symptom_deltas:", np.count_nonzero(t_symptom_deltas), len(t_symptom_deltas))
        dst_symptom_f = s.create_numeric(d_tests, "{}_delta".format(k), 'float32')
        dst_symptom_f.data.write(t_symptom_deltas)

    dst_pid_f = s.get(s_tests['patient_id']).create_like(d_tests, 'patient_id')
    dst_pid_f.data.write(t_pids)
    dst_rslt_f = s.get(s_tests['result']).create_like(d_tests, 'result')
    dst_rslt_f.data.write(t_rslts)
    dst_pcr_rslt_f = s.get(s_tests['result']).create_like(d_tests, 'pcr_result')
    dst_pcr_rslt_f.data.write(t_pcr_res)


def merge_consolidated_fields(s, src, dest, src_ptnt_keys):
    # d_consolidated = dest['consolidated']
    s_ptnts = src['patients']
    # d_asmts = dest['assessments']
    d_tests = dest['tests']

    print("destination test keys:", d_tests.keys())
    # do the assessment consolidation here!
    # generate a filter that you apply to d_consolidated/pid aid and tid
    # write out the modified daily assessments if you are taking the max of all daily symptoms, for example, to a new
    # table called 'daily_assessments'
    # then do the merge (using 'daily_assessments' instead of assessments)

    p_ids = s.get(s_ptnts['id']).data[:]
    t_pids = s.get(d_tests['patient_id']).data[:]

    ptnt_fields = tuple(k for k in src_ptnt_keys if k != 'id')
    s.merge_left(t_pids, p_ids,
                 right_fields=tuple(s.get(s_ptnts[k]) for k in ptnt_fields),
                 right_writers=tuple(s.get(s_ptnts[k]).create_like(d_tests, k).writeable() for k in ptnt_fields))


with h5py.File('/home/ben/covid/ds_20201101_full.hdf5', 'r') as src:
    with h5py.File('/home/ben/covid/ds_assessments_in_window_2.hdf5', 'w') as dst:
        s = Session()

        symptoms = ['persistent_cough', 'fever', 'fatigue', 'delirium', 'shortness_of_breath', 'diarrhoea',
                    'abdominal_pain', 'chest_pain', 'hoarse_voice', 'skipped_meals', 'loss_of_smell', 'headache',
                    'sore_throat']
        custom_symptom_thresholds = {'fatigue': 3, 'shortness_of_breath': 3}
        asmt_fields = ['patient_id'] + symptoms + ['updated_at_day']

        ptnt_fields = ['country_code', 'age', 'gender', 'bmi_clean',
                       'has_diabetes', 'has_heart_disease', 'has_lung_disease', 'has_kidney_disease', 'has_cancer']

        test_fields = ['result', 'mechanism']

        window = 28   # window for looking at the pcr and symptoms BEFORE AB+/-

        filter_and_generate_consolidated_mappings(s, src, dst,
                                                  datetime(2020, 4, 1).timestamp(),
                                                  datetime(2020, 9, 25).timestamp(),
                                                  window, ptnt_fields, asmt_fields, test_fields)

        merge_consolidated_fields(s, src, dst, ptnt_fields)

        # print(dst['consolidated'].keys())
        # print(len(s.get(dst['consolidated']['pid']).data))
        # print(len(s.get(dst['consolidated']['country_code']).data))
        # pdf = pd.DataFrame({k: s.get(dst['patients'][k]).data[:] for k in dst['patients'].keys()})
        # cdf = pd.DataFrame({k: s.get(dst['consolidated'][k]).data[:] for k in dst['consolidated'].keys()})
        tdf = pd.DataFrame({k: s.get(dst['tests'][k]).data[:] for k in dst['tests'].keys()})
        print(tdf.keys())
        tdf.to_csv('/home/ben/covid/assessments_in_window_2.csv')
