from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import h5py

from exetera.core.session import Session
from exetera.core.utils import Timer, build_histogram
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
    # u_t_pids = s.apply_spans_first(t_spans, test_pids)
    print("t_spans:", len(t_spans))

    # tcounts = defaultdict(int)
    for i in range(len(t_spans)-1):
        sps, spe = t_spans[i:i+2]
        ptnt_test_ts = test_ts[sps:spe]
        test_order = np.argsort(ptnt_test_ts)[::-1]

        # # for insight only
        # tcounts[u_t_pids[i]] += len(ptnt_test_ts)

        cur_ts = test_ts[sps + test_order[0]]
        for j in test_order[1:]:
            prev_ts = test_ts[sps + j]
            if cur_ts - prev_ts < 86400 * window:
                remaining[sps + j] = False
            else:
                cur_ts = prev_ts

    #print(np.unique(list(tcounts.values()), return_counts=True))

    return remaining


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
    
    with Timer('filtering by pcr test only'):
        pcr1 = np.zeros(len(t_pids), dtype=np.bool)
        pcr2 = np.zeros(len(t_pids), dtype=np.bool)
        pcr3 = np.zeros(len(t_pids), dtype=np.bool)
        atb1 = np.zeros(len(t_pids), dtype=np.bool)
        atb2 = np.zeros(len(t_pids), dtype=np.bool)
        atb3 = np.zeros(len(t_pids), dtype=np.bool)

        test_type_from_mechanism_v1(s, s.get(s_tests['mechanism']), s.get(s_tests['mechanism_freetext']),
                                    pcr1, pcr2, pcr3, atb1, atb2, atb3)
        t_pcr_filter = pcr1 | pcr2

    t_test_filter = t_patient_filter & t_date_filter & t_result_filter & t_pcr_filter
    print("t_test_filter:", t_test_filter.sum(), len(t_test_filter))

    t_tats = get_test_dates(s.get(s_tests['date_taken_specific']).data[:],
                            s.get(s_tests['date_taken_between_start']).data[:],
                            s.get(s_tests['date_taken_between_end']).data[:])

    t_pids = s.apply_filter(t_test_filter, t_pids)
    t_tats = s.apply_filter(t_test_filter, t_tats)
    t_rslts = s.apply_filter(t_test_filter, t_rslts)

    # remove tests that are too close to other tests
    t_proximal_filter = get_proximal_test_filter(s, t_pids, t_tats, t_rslts, window)
    print("t_proximal_filter: ", t_proximal_filter.sum(), len(t_proximal_filter))

    t_pids = s.apply_filter(t_proximal_filter, t_pids)
    t_tats = s.apply_filter(t_proximal_filter, t_tats)
    t_rslts = s.apply_filter(t_proximal_filter, t_rslts)
    t_ids = s.apply_filter(t_proximal_filter, s.apply_filter(t_test_filter, s.get(s_tests['id'])))
    t_pos = t_rslts == 4

    # initial assessment filtering
    # ----------------------------

    with Timer("getting asmt fields"):
        a_pids = s.get(s_asmts['patient_id']).data[:]
        a_cats = s.get(s_asmts['created_at']).data[:]

    # keep only assessments for patients who have pass the patient filter, above
    a_patient_filter = np.zeros(len(a_pids), dtype=np.bool)
    for i in range(len(a_patient_filter)):
        if a_pids[i] in valid_patients:
            a_patient_filter[i] = True

    with Timer("filtering asmts by date"):
        a_date_filter = (a_cats >= (start_ts - 86400 * window)) & (a_cats < end_ts)
    print("a_date_filter:", a_date_filter.sum(), len(a_date_filter))

    assessment_filter = a_patient_filter & a_date_filter
    print("a_assessment_filter:", assessment_filter.sum(), len(assessment_filter))

    a_ids = s.apply_filter(assessment_filter, s.get(s_asmts['id']).data[:])
    a_pids = s.apply_filter(assessment_filter, a_pids)
    a_cats = s.apply_filter(assessment_filter, a_cats)

    # group tests by patient_id for easy lookup when iterating over assessments
    # -------------------------------------------------------------------------

    test_by_patient = defaultdict(PatientTests)
    for i in range(len(t_pids)):
        test_by_patient[t_pids[i]].append(Test(t_ids[i], t_tats[i], t_pos[i]))

    print('unique test patients:', len(test_by_patient))

    # select the assessments that contribute to each tests
    # ----------------------------------------------------

    patient_assessment_counts = defaultdict(int)
    test_assessment_counts = defaultdict(int)
    total_count = 0
    relevant_asmts = np.zeros(len(a_pids), dtype=np.bool)
    relevant_asmt_counts = np.zeros(len(a_pids), dtype=np.int8)
    a_pid_spans = s.get_spans(a_pids)
    print("a_pid_spans:", len(a_pid_spans))
    u_a_pids = s.apply_spans_first(a_pid_spans, a_pids)
    print("u_a_pids:", len(u_a_pids))
    for i in range(len(u_a_pids)):
        pid = u_a_pids[i]
        if pid in test_by_patient:
            sps, spe = a_pid_spans[i:i+2]
            ptnt_cats = a_cats[sps:spe]
            for t in test_by_patient[pid].tests:
                ptnt_relevant_asmts = (ptnt_cats >= t.timestamp - 86400 * window) & (ptnt_cats <= t.timestamp)
                relevant_asmts[sps:spe] |= ptnt_relevant_asmts
                relevant_asmt_counts[sps:spe] += ptnt_relevant_asmts
                total_count += ptnt_relevant_asmts.sum()
                patient_assessment_counts[pid] += ptnt_relevant_asmts.sum()
                test_assessment_counts[t.test_id] += ptnt_relevant_asmts.sum()

    print('relevant_asmts:', relevant_asmts.sum(), len(relevant_asmts))
    print('relevant assessment counts:', np.unique(relevant_asmt_counts, return_counts=True))
    print('patient_assessment_counts:', sum(list(patient_assessment_counts.values())), sorted(build_histogram(patient_assessment_counts.values())))
    print('test_assessment_counts:', sum(list(test_assessment_counts.values())), sorted(build_histogram(test_assessment_counts.values())))
    print(total_count)

    consolidated_pids = np.zeros(total_count, dtype=a_pids.dtype)
    consolidated_aids = np.zeros(total_count, dtype=a_ids.dtype)
    consolidated_tids = np.zeros(total_count, dtype=t_ids.dtype)

    acc = 0
    for i in range(len(u_a_pids)):
        pid = u_a_pids[i]
        if pid in test_by_patient:
            sps, spe = a_pid_spans[i:i+2]
            ptnt_cats = a_cats[sps:spe]
            for t in test_by_patient[pid].tests:
                ptnt_relevant_asmts = (ptnt_cats >= t.timestamp - 86400 * window) & (ptnt_cats <= t.timestamp)

                for a in range(len(ptnt_relevant_asmts)):
                    if ptnt_relevant_asmts[a] == True:
                        consolidated_pids[acc] = pid
                        consolidated_tids[acc] = t.test_id
                        consolidated_aids[acc] = a_ids[sps + a]
                        acc += 1

    # Note: Patients get filtered out by this stage for either of the following reasons:
    # * they never had any assessments in the first place
    # * they don't have any assessments within the window of interest for any of their tests
    print("unique consolidated pids:", len(np.unique(consolidated_pids)))
    print("unique consolidated tids:", len(np.unique(consolidated_tids)))
    print("unique consolidated aids:", len(np.unique(consolidated_aids)))

    # map fields from assessments
    d_asmts = dest.create_group('assessments')
    for k in ['id'] + assessment_fields:
        sf = s.get(s_asmts[k])
        df = sf.create_like(d_asmts, k)
        df.data.write(s.apply_filter(assessment_filter, sf.data[:]))

    # map fields from patients
    d_ptnts = dest.create_group('patients')
    for k in ['id'] + patient_fields:
        sf = s.get(s_ptnts[k])
        df = sf.create_like(d_ptnts, k)
        df.data.write(s.apply_filter(patient_filter, sf.data[:]))

    # map fields from tests
    d_tests = dest.create_group('tests')
    for k in ['id'] + test_fields:
        sf = s.get(s_tests[k])
        df = sf.create_like(d_tests, k)
        df.data.write(s.apply_filter(t_proximal_filter, s.apply_filter(t_test_filter, sf.data[:])))

    d_consolidated = dest.create_group('consolidated')
    s.get(s_ptnts['id']).create_like(d_consolidated, 'pid').data.write(consolidated_pids)
    s.get(s_tests['id']).create_like(d_consolidated, 'tid').data.write(consolidated_tids)
    s.get(s_asmts['id']).create_like(d_consolidated, 'aid').data.write(consolidated_aids)


def merge_consolidated_fields(s, dest):

    d_consolidated = dest['consolidated']
    d_ptnts = dest['patients']
    d_asmts = dest['assessments']
    d_tests = dest['tests']

    # do the assessment consolidation here!
    # generate a filter that you apply to d_consolidated/pid aid and tid
    # write out the modified daily assessments if you are taking the max of all daily symptoms, for example, to a new
    # table called 'daily_assessments'
    # then do the merge (using 'daily_assessments' instead of assessments)

    p_ids = s.get(d_ptnts['id']).data[:]
    c_pids = s.get(d_consolidated['pid']).data[:]
    a_ids = s.get(d_asmts['id']).data[:]
    c_aids = s.get(d_consolidated['aid']).data[:]
    t_ids = s.get(d_tests['id']).data[:]
    c_tids = s.get(d_consolidated['tid']).data[:]

    ptnt_fields = tuple(k for k in d_ptnts.keys() if k != 'id')
    s.merge_left(c_pids, p_ids,
                 right_fields=tuple(s.get(d_ptnts[k]) for k in ptnt_fields),
                 right_writers=tuple(s.get(d_ptnts[k]).create_like(d_consolidated, k).writeable() for k in ptnt_fields))
    asmt_fields = tuple(k for k in d_asmts.keys() if k != 'id')
    s.merge_left(c_aids, a_ids,
                 right_fields=tuple(s.get(d_asmts[k]) for k in asmt_fields),
                 right_writers=tuple(s.get(d_asmts[k]).create_like(d_consolidated, k).writeable() for k in asmt_fields))
    test_fields = tuple(k for k in d_tests.keys() if k != 'id')
    s.merge_left(c_tids, t_ids,
                 right_fields=tuple(s.get(d_tests[k]) for k in test_fields),
                 right_writers=tuple(s.get(d_tests[k]).create_like(d_consolidated, k).writeable() for k in test_fields))


with h5py.File('/home/ben/covid/ds_20201014_full.hdf5', 'r') as src:
    with h5py.File('/home/ben/covid/ds_asmt_window.hdf5', 'w') as dst:
        s = Session()

        asmt_fields = ['persistent_cough', 'fever', 'fatigue', 'delirium', 'shortness_of_breath', 'diarrhoea',
                       'abdominal_pain', 'chest_pain', 'hoarse_voice', 'skipped_meals', 'loss_of_smell', 'headache',
                       'sore_throat']

        ptnt_fields = ['country_code', 'age', 'gender', 'bmi_clean', 'healthcare_professional', 'contact_health_worker',
                       'has_diabetes', 'has_heart_disease', 'has_lung_disease', 'has_kidney_disease']

        test_fields = ['result']


        if "consolidated" not in dst.keys():
            filter_and_generate_consolidated_mappings(s, src, dst,
                                                      datetime(2020, 8, 1).timestamp(),
                                                      datetime(2020, 10, 30).timestamp(),
                                                      3, ptnt_fields, asmt_fields, test_fields)

        merge_consolidated_fields(s, dst)

        print(dst['consolidated'].keys())
        print(len(s.get(dst['consolidated']['pid']).data))
        print(len(s.get(dst['consolidated']['country_code']).data))
        pdf = pd.DataFrame({k: s.get(dst['patients'][k]).data[:] for k in dst['patients'].keys()})
        cdf = pd.DataFrame({k: s.get(dst['consolidated'][k]).data[:] for k in dst['consolidated'].keys()})

