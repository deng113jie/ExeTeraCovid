import datetime
import time

import numpy as np
from numba import njit

from exetera.core.session import Session
from exetera.core.utils import Timer
from exetera.core.dataframe import merge

# src_filename = '/nvme0_mounts/nvme0lv01/exetera/recent/ds_20210929_full.hdf5'
# dst_filename = './non_covid_respiratory_outcomes.hdf5'
src_filename = '/home/ben/covid/ds_20211014_full.hdf5'
dst_filename = '/home/ben/covid/symptom_lengths_10_14.hdf5'


@njit
def resize_if_required(array, index, delta):
    if index >= len(array):
        temp_ = np.zeros(len(array) + delta, array.dtype)
        # print('temp len', len(temp_))
        temp_[:len(array)] = array
        array = temp_
        # print('resizing to', len(array))

    return array


@njit
def truncate_if_required(array, length):
    if len(array) > length:
        temp_ = np.zeros(length, array.dtype)
        temp_[:] = array[:length]
        array = temp_

    return array


# @njit
def count_illnesses(aspans, adates, ahealthy, asymptoms, alocation,  # atreatment,
                    t_pids, tspans, tdates, tresult,
                    pos_illness_counts, neg_illness_counts,
                    threshold_days, illness_length_threshold_days,
                    t_illness_pids, t_illness_start, t_illness_end,
                    t_illness_symptoms, t_illness_hospitalised, t_illness_covid,
                    verbose=False):
    has_symptom = asymptoms
    #     t_illness_pids = np.zeros(0, dtype='S32')
    #     t_illness_starts = np.zeros(0, dtype='float64')
    #     t_illness_ends = np.zeros(0, dtype='float64')
    #     t_illness_symptoms = np.zeros(0, dtype='uint32')
    #     t_illness_hospitalised = np.zeros(0, dtype='int8')
    #     t_illness_covid = np.zeros(0, dtype='int8')

    resize_delta = 100000
    threshold = 86400 * threshold_days
    illness_length_threshold = 86400 * illness_length_threshold_days
    total_illness_count = 0

    for i in range(len(aspans) - 1):
        # if i % 100000 == 0:
        #     print(i)
        astart, aend = aspans[i], aspans[i + 1]
        tstart, tend = tspans[i], tspans[i + 1]

        healthy = True
        # illness_count = 0
        start_date = 0
        # cur_end_date = 0
        start_index = -1
        # end_index = -1
        last_unhealthy_index = -1
        first_healthy_after_unhealthy_index = -1
        hospitalised = False
        symptoms = np.uint32(0)
        # iterate to one past the end so that we can handle tying up an open period of
        # unhealthiness

        pos_illness_count = 0
        neg_illness_count = 0

        if verbose:
            first_date = adates[astart]
            for a in range(astart, aend):
                print(a, (adates[a] - first_date) / 86400, has_symptom[a])

        for a in range(astart, aend + 1):
            transition_to_healthy = False
            transition_to_unhealthy = False
            if a == aend:
                # handle wrapping up an open period of unhealthiness
                if not healthy:
                    transition_to_healthy = True
            else:
                if healthy:
                    if has_symptom[a] == 1:
                        # healthy status and unhealthy record so transition to unhealthy status
                        transition_to_unhealthy = True
                else:
                    if has_symptom[a] == 0:
                        # unhealthy status and healthy record:
                        # . in this case, track the duration of healthy assessments
                        #   . if it passes the threshold, end the period of ill health at the
                        #     first healthy assessment
                        #   . if it is too short, ignore the intervening unhealthy records
                        if first_healthy_after_unhealthy_index == -1:
                            first_healthy_after_unhealthy_index = a
                        if adates[a] - adates[last_unhealthy_index] > threshold:
                            transition_to_healthy = True
                    elif has_symptom[a] == 1:
                        # unhealthy status and unhealthy record:

                        # check if the gap is sufficiently large that it counts as a new period of unhealthiness

                        if adates[a] - adates[a - 1] > threshold:
                            transition_to_healthy = True
                            transition_to_unhealthy = True
                        else:
                            # reset any intervening healthy index to void any intervening healthy records
                            first_healthy_after_unhealthy_index = -1
                            last_unhealthy_index = a
                            hospitalised = hospitalised or (alocation[a] == 2 or alocation[a] == 3)
                            symptoms = symptoms | asymptoms[a]

            if transition_to_healthy:
                # first check whether we are transitioning to healthy because we have a long
                # enough healthy period; in this case we go back to the unhealthy -> healthy
                # transition assessment
                if first_healthy_after_unhealthy_index != -1:
                    end_index = first_healthy_after_unhealthy_index
                else:
                    end_index = a

                if end_index == aend:
                    # true end date is unknown; count as last unhealthy entry + threshold days
                    end_date = adates[last_unhealthy_index - 1] + threshold
                else:
                    end_date = min(adates[end_index], adates[last_unhealthy_index] + threshold)

                if verbose:
                    print(start_index, last_unhealthy_index, start_date / 86400, end_date / 86400,
                          (end_date - start_date) / 86400)

                # record the illness
                if end_date - start_date >= illness_length_threshold:
                    # check whether there is a positive pcr test within 7 days of the illness start
                    test_result = 0
                    for t in range(tstart, tend):
                        td = tdates[t]
                        # TODO: pull this parameter out to be varied
                        if td != 0.0 and td >= start_date - (86400 * 5) and td <= end_date - (86400 * 5):
                            if tresult[t] == 3:
                                test_result = 1
                            if tresult[t] == 4:
                                test_result = 2
                                break

                    t_illness_pids = resize_if_required(t_illness_pids, total_illness_count,
                                                        resize_delta)
                    t_illness_pids[total_illness_count] = t_pids[tstart]

                    t_illness_start = resize_if_required(t_illness_start, total_illness_count,
                                                         resize_delta)
                    t_illness_start[total_illness_count] = start_date

                    t_illness_end = resize_if_required(t_illness_end, total_illness_count,
                                                       resize_delta)
                    t_illness_end[total_illness_count] = end_date

                    t_illness_symptoms = resize_if_required(t_illness_symptoms, total_illness_count,
                                                            resize_delta)
                    t_illness_symptoms[total_illness_count] = symptoms

                    t_illness_hospitalised = resize_if_required(t_illness_hospitalised, total_illness_count,
                                                                resize_delta)
                    t_illness_hospitalised[total_illness_count] = hospitalised

                    t_illness_covid = resize_if_required(t_illness_covid, total_illness_count,
                                                         resize_delta)
                    t_illness_covid[total_illness_count] = test_result
                    total_illness_count += 1

                    if test_result == 2:
                        pos_illness_count += 1
                    else:
                        neg_illness_count += 1

                # reset to healthy
                healthy = True
                start_date = 0
                start_index = -1
                last_unhealthy_index = -1
                # cur_end_date = 0
                # end_index = -1
                first_healthy_after_unhealthy_index = -1
                hospitalised = False
                symptoms = np.int32(0)

            if transition_to_unhealthy:
                healthy = False
                start_date = adates[a]
                start_index = a
                last_unhealthy_index = a
                first_healthy_after_unhealthy = -1
                hospitalised = alocation[a] == 2 or alocation[a] == 3
                symptoms = symptoms | asymptoms[a]

        pos_illness_counts[i] = pos_illness_count
        neg_illness_counts[i] = neg_illness_count

    return (truncate_if_required(t_illness_pids, total_illness_count),
            truncate_if_required(t_illness_start, total_illness_count),
            truncate_if_required(t_illness_end, total_illness_count),
            truncate_if_required(t_illness_symptoms, total_illness_count),
            truncate_if_required(t_illness_hospitalised, total_illness_count),
            truncate_if_required(t_illness_covid, total_illness_count))


def go(reprocess):
    with Session() as s:
        max_gap_days = 7
        illness_len_days = 7

        src_ds = s.open_dataset(src_filename, 'r', 'src')
        if reprocess:
            open_mode = 'w'
        else:
            open_mode = 'r+'
        dst_ds = s.open_dataset(dst_filename, open_mode, 'dst')

        s_ptnts = src_ds['patients']
        s_asmts = src_ds['assessments']
        s_tests = src_ds['tests']

        print("patient_keys")
        print(s_ptnts.keys())
        print("assessment_keys")
        print(s_asmts.keys())

        print(len(s_ptnts['id']))
        p_filter = np.ones(len(s_ptnts['id']), dtype=bool)

        with Timer("filter out patients with insufficient assessments", new_line=True):
            p_filter = s_ptnts['assessment_counts'].data[:] >= 2
            print(np.unique(p_filter, return_counts=True))

        with Timer("filter out patients with insufficient tests", new_line=True):
            p_filter = p_filter & (s_ptnts['test_counts'].data[:] >= 1)
            # print(np.unique(s_ptnts['test_counts'].data[:], return_counts=True))
            print(np.unique(p_filter, return_counts=True))

        print("filtered out {} of {} patients".format(len(p_filter) - np.count_nonzero(p_filter), len(p_filter)))

        # with Timer("get assessment_spans", new_line=True):
        #   a_spans = s_asmts['patient_id'].get_spans()

        if reprocess or 'filtered_patients' not in dst_ds.keys():
            dst_fptnts = dst_ds.create_dataframe("filtered_patients")
            dst_fptnts['id'] = s_ptnts['id'].apply_filter(p_filter)
            print(len(dst_fptnts['id']))
        else:
            dst_fptnts = dst_ds['filtered_patients']

        if reprocess or 'filtered_assessments' not in dst_ds.keys():
            dst_fasmts = dst_ds.create_dataframe("filtered_assessments")

            asmt_to_merge = [
                'altered_smell', 'brain_fog', 'chest_pain', 'chills_or_shivers', 'country_code', 'created_at',
                'created_at_day', 'delirium', 'dizzy_light_headed', 'ear_ringing', 'earache', 'eye_soreness',
                'fatigue', 'fever', 'headache', 'headache_frequency', 'health_status', 'hoarse_voice', 'id',
                'inconsistent_healthy', 'inconsistent_not_healthy', 'level_of_isolation', 'location',
                'loss_of_smell', 'nausea', 'patient_id', 'persistent_cough', 'runny_nose', 'shortness_of_breath',
                'skin_burning', 'skipped_meals', 'sneezing', 'sore_throat', 'swollen_glands', 'temperature',
                'temperature_35_to_42_inclusive', 'temperature_c_clean', 'temperature_modified',
                'temperature_valid', 'treatment', 'typical_hayfever', 'unusual_joint_pains',
                'unusual_muscle_pains', 'updated_at', 'version']
            #            asmt_to_merge = ["id", "patient_id", "created_at"]

            with Timer("filter assessments by remaining patients", new_line=True):
                merge(dst_fptnts, s_asmts, dst_fasmts, "id", "patient_id", left_fields=["id"],
                      right_fields=asmt_to_merge,
                      hint_left_keys_ordered=True, hint_right_keys_ordered=True)
        else:
            dst_fasmts = dst_ds['filtered_assessments']

        #         pretty_print(dst_fasmts, [(0, 100), (200,300)], ['id', 'patient_id', 'created_at'])
        print("remaining assessments:", len(dst_fasmts['id']))

        if reprocess or 'filtered_tests' not in dst_ds.keys():
            dst_ftests = dst_ds.create_dataframe("filtered_tests")

            print(s_tests.keys())
            # tests_to_merge =\
            #     [k for k in s_tests.keys() if k not in
            #      ('days_in_fridge_valid', 'effective_test_date_valid',
            #       'location_other', 'mechanism_freetext')]
            tests_to_merge = list(s_tests.keys())
            for k in tests_to_merge:
                print(k, s_tests[k])

            with Timer("filter tests by remaining patients", new_line=True):
                merge(dst_fptnts, s_tests, dst_ftests, "id", "patient_id", left_fields=["id"],
                      right_fields=tests_to_merge,
                      hint_left_keys_ordered=True, hint_right_keys_ordered=True)

        else:
            dst_ftests = dst_ds['filtered_tests']

        if reprocess or 'filtered_spans' not in dst_ds.keys():
            dst_fspans = dst_ds.create_dataframe('filtered_spans')

            # get spans for filtered assessments
            with Timer("get spans for filtered patients' assessments", new_line=True):
                a_spans = dst_fasmts['patient_id'].get_spans()
                dst_fspans.create_numeric('a_spans', 'int64').data.write(a_spans)
            # get spans for filtered tests
            with Timer("get spans for filtered patients' tests", new_line=True):
                t_spans = dst_ftests['patient_id'].get_spans()
                dst_fspans.create_numeric('t_spans', 'int64').data.write(t_spans)
        else:
            dst_fspans = dst_ds['filtered_spans']

        print('a_spans:', len(dst_fspans['a_spans']))
        print('t_spans:', len(dst_fspans['t_spans']))

        #         for k in ('chest_pain', 'chills_or_shivers', 'fever', 'hoarse_voice', 'persistent_cough', 'runny_nose', 'shortness_of_breath'):
        #             print(k, np.unique(dst_fasmts[k].data[:], return_counts=True))

        if 'illness_5' in dst_ds.keys():
            del dst_ds['illness_5']
        if 'illness_7' in dst_ds.keys():
            del dst_ds['illness_7']

        # symps_to_check = ('altered_smell', 'brain_fog', 'fatigue', 'shortness_of_breath', 'headache')
        symps_to_check = ('headache',)
        for symp in symps_to_check:
            print("calculating illness durations for '{}'".format(symp))

            illness_table_name = 'illness_{}'.format(symp)

            if illness_table_name in dst_ds.keys():
                del dst_ds[illness_table_name]

            if reprocess or illness_table_name not in dst_ds.keys():
                aspans_ = dst_fspans['a_spans'].data[:]
                adates_ = dst_fasmts['created_at'].data[:]
                ahealthy_ = dst_fasmts['health_status'].data[:]
                # asymptoms_ = np.zeros(len(dst_fasmts['health_status']), dtype='uint32')
                channel = 1
                #             symps_to_check = ('altered_smell', 'brain_fog', 'chest_pain', 'chills_or_shivers', 'fatigue', 'fever', 'persistent_cough',
                #                               'runny_nose', 'shortness_of_breath', 'sneezing', 'sore_throat', 'swollen_glands', 'unusual_joint_pains',
                #                               'unusual_muscle_pains')
                #                 for symp in symps_to_check:
                #                     print(symp, dst_fasmts[symp].keys)
                #                     if len(dst_fasmts[symp].keys) == 5:
                #                         symp_flag = np.where(dst_fasmts[symp].data[:] >= 3, channel, 0)
                #                     else:
                #                         symp_flag = np.where(dst_fasmts[symp].data[:] == 2, channel, 0)
                #                     asymptoms_ = asymptoms_ | symp_flag
                #                     # print(np.unique(ahealth_, return_counts=True))
                #                     channel *= 2

                if len(dst_fasmts[symp].keys) == 5:
                    symps_ = np.where(dst_fasmts[symp].data[:] >= 3, 1, 0)
                else:
                    symps_ = np.where(dst_fasmts[symp].data[:] == 2, 1, 0)
                print("symptom: {}".format(np.unique(symps_, return_counts=True)))

                alocations_ = dst_fasmts['location'].data[:]
                #                 print(np.unique(ahealthy_, return_counts=True), dst_fasmts['health_status'].keys)
                #                 print(np.unique(alocations_, return_counts=True), dst_fasmts['location'].keys)
                tpids_ = dst_ftests['patient_id'].data[:]
                tspans_ = dst_fspans['t_spans'].data[:]
                tdates_ = dst_ftests['date_taken_specific'].data[:]
                tresults_ = dst_ftests['result'].data[:]
                print(np.count_nonzero(tdates_ == 0.0), len(tdates_))

                clean_healthy = np.where(ahealthy_ == 2, 1, 0)

                df = dst_ds.create_dataframe(illness_table_name)

                print("aspans: {}, tspans: {}".format(len(aspans_), len(tspans_)))
                pos_illness_counts = np.zeros(len(aspans_) - 1, dtype=np.int32)
                neg_illness_counts = np.zeros(len(aspans_) - 1, dtype=np.int32)
                t_illness_pids = np.zeros(0, dtype='S32')
                t_illness_starts = np.zeros(0, dtype='float64')
                t_illness_ends = np.zeros(0, dtype='float64')
                t_illness_symptoms = np.zeros(0, dtype='uint32')
                t_illness_hospitalised = np.zeros(0, dtype='int8')
                t_illness_covid = np.zeros(0, dtype='int8')
                t_illness_pids, t_illness_starts, t_illness_ends, t_illness_symptoms, t_illness_hospitalised, t_illness_covid = \
                    count_illnesses(aspans_, adates_, clean_healthy, symps_, alocations_,  # None,
                                    tpids_, tspans_, tdates_, tresults_,
                                    pos_illness_counts, neg_illness_counts, max_gap_days, illness_len_days,
                                    t_illness_pids, t_illness_starts, t_illness_ends,
                                    t_illness_symptoms, t_illness_hospitalised, t_illness_covid)
                print(pos_illness_counts.sum(), neg_illness_counts.sum())
                print("illness_count:", len(t_illness_pids))
                # print(np.unique(illness_counts, return_counts=True))
                print("illness covid counts:", np.unique(t_illness_covid, return_counts=True))
                print("illness symptom counts:", np.unique(t_illness_symptoms, return_counts=True))

                df.create_fixed_string('patient_id', '32').data.write(t_illness_pids)
                df.create_numeric('illness_start', 'float64').data.write(t_illness_starts)
                df.create_numeric('illness_end', 'float64').data.write(t_illness_ends)
                df.create_numeric('illness_symptoms', 'uint32').data.write(t_illness_symptoms)
                df.create_numeric('hospitalised', 'int8').data.write(t_illness_hospitalised)
                df.create_numeric('covid_status', 'int8').data.write(t_illness_covid)


#             else:
#                 illness_5 = dst_ds['illness_5']
#                 illness_7 = dst_ds['illness_7']


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', action='store_true', help="force reprocessing")

#     args = parser.parse_args()

#     go(reprocess=args.f)
go(reprocess=False)
print("done!")
