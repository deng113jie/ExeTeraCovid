import numpy as np
import csv
from datetime import datetime, timezone

from exetera.core import session
from exetera.core.persistence import DataStore
from exetera.core import utils
from exetera.core import persistence as prst
from exeteracovid.algorithms.test_type_from_mechanism import test_type_from_mechanism_v1_standard_input
from exeteracovid.algorithms.covid_test import unique_tests_v1, multiple_tests_start_with_negative_v1
from exeteracovid.algorithms.covid_test_date import covid_test_date_v1
from exeteracovid.algorithms.test_type_from_mechanism import pcr_standard_summarize_v1

list_symptoms = ['fatigue', 'abdominal_pain', 'chest_pain', 'sore_throat', 'shortness_of_breath',
                     'skipped_meals', 'loss_of_smell', 'unusual_muscle_pains', 'headache', 'hoarse_voice', 'delirium',
                     'diarrhoea',
                     'fever', 'persistent_cough', 'dizzy_light_headed', 'eye_soreness', 'red_welts_on_face_or_lips',
                     'blisters_on_feet']


def date_filtering(spans_test, spans_asmt, dict_max, out_test_fin, out_asmt_init, first_healthy):
    """
    Compare the test date verse the assessment date.
    """
    filt_hf = np.zeros(spans_asmt[-1])
    filt_date = np.zeros(spans_asmt[-1], dtype='bool')
    filt_testwithasmt = np.zeros(spans_test[-1])
    count_ta = 0
    spans_start_test = spans_test[:-1]
    spans_end_test = spans_test[1:]
    spans_start_asmt = spans_asmt[:-1]
    spans_end_asmt = spans_asmt[1:]

    date_eff = out_test_fin['date_effective_test'].data[:]
    date_update = out_asmt_init['created_at'].data[:]

    dict_symp = {}
    for f in list_symptoms:
        dict_symp[f] = out_asmt_init[f].data[:]

    for i in range(len(spans_start_test)):
        i_sa = spans_start_asmt[i]
        i_ea = spans_end_asmt[i]
        i_st = spans_start_test[i]
        i_et = spans_end_test[i]
        filt_hf[i_sa:i_ea] = first_healthy[i]
        dates_test = date_eff[i_st:i_et]
        dates_asmt = date_update[i_sa:i_ea]
        for (e, d) in enumerate(dates_test):
            dates_pos = np.where(np.logical_and(dates_asmt <= d, dates_asmt >= d - 86400 * 14), 1, 0)
            filt_date[i_sa:i_ea] += dates_pos.astype('bool')

            if np.sum(dates_pos) > 0:  # and first_healthy[i]==1:
                filt_testwithasmt[e + i_st] = 1
                dict_max['healthy_first'].append(first_healthy[i])
                for f in list_symptoms:
                    values = dict_symp[f][i_sa:i_ea]
                    values_pos = values[dates_pos == 1]
                    dict_max[f].append(np.max(values_pos))
                count_ta += 1
    print(count_ta)
    return filt_date, filt_testwithasmt, dict_max


def save_df_to_csv(df, csv_name, fields, chunk=200000):  # chunk=100k ~ 20M/s
    with open(csv_name, 'w', newline='') as csvfile:
        columns = list(fields)
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        field1 = columns[0]
        for current_row in range(0, len(df[field1].data), chunk):
            torow = current_row + chunk if current_row + chunk < len(df[field1].data) else len(df[field1].data)
            batch = list()
            for k in fields:
                batch.append(df[k].data[current_row:torow])
            writer.writerows(list(zip(*batch)))


def max_symp_up14days_beforetest(s, source, output):

    ds = DataStore()
    ts = str(datetime.now(timezone.utc))
    src_pat = source['patients']
    print(src_pat.keys())

    src_test = source['tests']
    list_testid = src_test['patient_id']
    list_testcreate = src_test['created_at']
    src_test = source['tests']
    out_test = output.create_dataframe('test_count')
    print(src_test.keys())

    pat_id = src_test['patient_id']
    result = src_test['result']
    mechanism = src_test['mechanism']
    created_at = src_test['created_at']
    print(len(pat_id))
    # ====
    # tests
    # ====

    # filter unique tests
    print('creating filter')
    clean_filt = unique_tests_v1(src_test, fields=('patient_id', 'result', 'mechanism', 'date_taken_specific',
                                                'date_taken_between_start', 'created_at'))
    out_test = output.create_dataframe('tests')
    src_test.apply_filter(clean_filt, ddf=out_test)

    # convert test date
    covid_test_date_v1(s, out_test, out_test, 'date_effective_test')

    # Filtering only definite results
    results_raw = out_test['result'].data[:]
    results_filt = np.where(np.logical_or(results_raw == 4, results_raw == 3), True, False)
    out_test.apply_filter(results_filt)

    # Creating clean mechanism
    test_type_from_mechanism_v1_standard_input(ds, out_test)
    pcr_standard_summarize_v1(s, out_test)

    # tests_fin
    out_test_fin = output.create_dataframe('tests_fin')
    writers_dict = {}
    # other fields
    for k in ('patient_id', 'date_effective_test', 'result', 'pcr_standard'):
        values = out_test[k].data[:]
        if k == 'result':
            values -= 3
        writers_dict[k] = out_test[k].create_like(out_test_fin, k, ts).data
        print(len(values), k)
        writers_dict[k].write_part(values)
    # converted_test
    values = np.zeros(len(out_test_fin['patient_id'].data), dtype='bool')
    writers_dict['converted_test'] = out_test_fin.create_numeric('converted_test', 'bool', timestamp=ts).data
    writers_dict['converted_test'].write_part(values)

    # ====
    # 1 assessments patients w/ multiple test and first ok
    # ====
    print('taking care of the old test')
    src_asmt = source['assessments']
    print(src_asmt.keys())

    # Remap had_covid_test to 0/1 2 to binary 0,1
    # test_positive = np.where(src_asmt['tested_covid_positive'].data[:] < 1, 0, 1)
    # spans = src_asmt['patient_id'].get_spans()
    # # Get the first index at which the hct field is maximum  group by
    # firstnz_tcp_ind = ds.apply_spans_index_of_max(spans, test_positive)  # positive if multiple tests
    # # Get the index of first element of patient_id when sorted
    # first_hct_ind = spans[:-1]
    # filt_tl = first_hct_ind != firstnz_tcp_ind  # COND1 has multiple tests w/ first negative and later positive
    # # Get the indices for which hct changed value (indicating that test happened after the first input)
    # sel_max_ind = ds.apply_filter(filter_to_apply=filt_tl, reader=firstnz_tcp_ind)
    #
    # # Get the index at which test is maximum and for which that hct is possible
    # max_tcp_ind = ds.apply_spans_index_of_max(spans, src_asmt['tested_covid_positive'].data[:])
    #
    # sel_max_tcp = ds.apply_indices(filt_tl, max_tcp_ind)
    #
    # sel_maxtcp_ind = ds.apply_filter(filter_to_apply=filt_tl, reader=max_tcp_ind)
    sel_max_ind, sel_max_tcp = multiple_tests_start_with_negative_v1(s, src_asmt)

    # Define usable assessments with correct test based on previous filter on indices
    usable_asmt_tests = output.create_dataframe('usable_asmt_tests')
    # ====
    # 2 assessments patients w/ multiple test and first ok
    # ====
    for k in ('id', 'patient_id', 'created_at', 'had_covid_test'):
        fld = src_asmt[k].create_like(usable_asmt_tests, k, ts)
        src_asmt[k].apply_index(sel_max_ind, target=fld)

    fld = src_asmt['created_at'].create_like(usable_asmt_tests, 'eff_result_time', ts)
    #src_asmt['created_at'].apply_index(sel_maxtcp_ind, target=fld)
    src_asmt['created_at'].apply_index(sel_max_tcp, target=fld)

    fld = src_asmt['tested_covid_positive'].create_like(usable_asmt_tests, 'eff_result', ts)
    #src_asmt['tested_covid_positive'].apply_index(sel_maxtcp_ind, target=fld)
    src_asmt['tested_covid_positive'].apply_index(sel_max_tcp, target=fld)

    fld = src_asmt['tested_covid_positive'].create_like(usable_asmt_tests, 'tested_covid_positive', ts)
    src_asmt['tested_covid_positive'].apply_index(sel_max_tcp, target=fld)

    # usable_asmt_tests now is patients with at least one negative tests and at one positive test

    # Making sure that the test is definite (either positive or negative)
    # ====
    # 2 assessments patients w/ multiple test and first ok ; and only positive
    # ====
    print('filter definite test answers')
    filt_deftest = usable_asmt_tests['tested_covid_positive'].data[:] > 1  # all positive tests given COND1
    print(np.sum(filt_deftest), len(usable_asmt_tests['tested_covid_positive'].data))
    # print(len(ds.get_reader(usable_asmt_tests['patient_id'])))
    for k in (
            'id', 'patient_id', 'created_at', 'had_covid_test', 'tested_covid_positive', 'eff_result_time',
            'eff_result'):
        fld = usable_asmt_tests[k]
        usable_asmt_tests[k].apply_filter(filt_deftest, target=fld)
    # usable_asmt_tests now is the positive tests of patients who have (at least one negative tests and at one positive test)

    # Getting gap between created at (max of hct date) and max of test result (eff_result_time)
    # ====
    # 3 delta_days_test
    # ====
    reader_hct = usable_asmt_tests['created_at'].data[:]
    reader_tcp = usable_asmt_tests['eff_result_time'].data[:]
    with utils.Timer('doing delta time'):
        delta_time = reader_tcp - reader_hct
        delta_days = delta_time / 86400
    print(delta_days[:10], delta_time[:10])
    # writer = ds.get_numeric_writer(usable_asmt_tests, 'delta_days_test', 'float32', ts, writemode='overwrite')
    writer = usable_asmt_tests.create_numeric('delta_days_test', 'float32', ts).data
    writer.write(delta_days)
    print(delta_days.shape, ' size of delta days')

    # ====
    # 4 Final day of test
    # ====
    date_final_test = np.where(delta_days < 7, reader_hct, reader_tcp - 2 * 86400)
    # writer = ds.get_timestamp_writer(usable_asmt_tests, 'date_final_test', ts, writemode='overwrite')
    writer = usable_asmt_tests.create_timestamp('date_final_test', ts).data
    writer.write(date_final_test)
    # print(ds.get_reader(usable_asmt_tests['date_final_test'])[:10], date_final_test[:10])
    # ====
    # 5 pcr_standard
    # ====
    pcr_standard = np.ones(len(usable_asmt_tests['patient_id'].data))
    # writer = ds.get_numeric_writer(usable_asmt_tests, 'pcr_standard', dtype='bool', timestamp=ts,
    #                               writemode='overwrite')
    writer = usable_asmt_tests.create_numeric('pcr_standard', 'bool', timestamp=ts).data
    writer.write(np.array(pcr_standard, dtype=bool))  # pcr_standard all ones here?

    # ====
    # out_test_fin copy from usable_asmt_tests
    # ====
    list_init = ('patient_id', 'date_final_test', 'tested_covid_positive', 'pcr_standard')
    list_final = ('patient_id', 'date_effective_test', 'result', 'pcr_standard')
    # Join copy usable_asmt_tests to out_test_fin
    for (i, f) in zip(list_init, list_final):
        values = usable_asmt_tests[i].data[:]
        if f == 'result':
            values -= 2
        # writers_dict[f] = reader.get_writer(out_test_fin, f, ts)
        print(len(values), f)
        if len(values) > 0:
            writers_dict[f].write(values)  # writers_dict write to df out_test_fin ('tests_fin')
    if len(usable_asmt_tests['patient_id'].data) > 0:
        writers_dict['converted_test'].write(np.ones(len(usable_asmt_tests['patient_id'].data), dtype='bool'))
    converted_fin = out_test_fin['converted_test'].data
    result_fin = out_test_fin['result'].data[:]
    pat_id_fin = out_test_fin['patient_id'].data[:]

    # ====
    # out_test_init 1 patients in both out_test_fin and assessments
    # ====
    out_asmt_init = output.create_dataframe('init_asmt')
    filt_asmt = prst.foreign_key_is_in_primary_key(out_test_fin['patient_id'].data[:],
                                                   src_asmt['patient_id'].data[:])  # patients that have assessment
    print(np.sum(filt_asmt), 'kept with filt_asmt')
    for k in list_symptoms + ['created_at', 'patient_id']:
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(out_asmt_init, k, ts)
        # ds.apply_filter(filt_asmt, reader, writer)
        fld = src_asmt[k].create_like(out_asmt_init, k, timestamp=ts)
        src_asmt[k].apply_filter(filt_asmt, target=fld)  # patient asmt that is in tests

    # ====
    # out_test_init 2 patients in both out_asmt_init and out_test_fin
    # ====
    filt_asmt2 = prst.foreign_key_is_in_primary_key(out_asmt_init['patient_id'].data[:],
                                                    out_test_fin['patient_id'].data[:])

    for k in out_test_fin.keys():
        out_test_fin[k].apply_filter(filt_asmt2, in_place=True)

    list_patid = out_test_fin['patient_id'].data[:]
    list_patcreate = out_test_fin['date_effective_test'].data[:]
    ind_pat = np.arange(len(list_patid), dtype=np.int64)
    with utils.Timer('sorting_patients'):
        #ind_pat = ds.dataset_sort(readers=(list_patid, list_patcreate), index=ind_pat)
        ind_pat = s.dataset_sort_index(sort_indices=(list_patid, list_patcreate), index=ind_pat)

    # ====
    # out_test_fin2 1 copy from out_test_fin and re-order
    # ====
    out_test_fin2 = output.create_dataframe('test_fin2')
    with utils.Timer('applying sort'):
        for k in out_test_fin.keys():
            # reader = ds.get_reader(out_test_fin[k])
            # writer = reader.get_writer(out_test_fin2, k, ts)
            # ds.apply_sort(ind_pat, reader, writer)
            fld = out_test_fin[k].create_like(out_test_fin2, k, ts)
            out_test_fin[k].apply_index(ind_pat, target=fld)

    print(len(np.unique(out_asmt_init['patient_id'].data[:])),
          len(np.unique(out_test_fin2['patient_id'].data[:])))

    spans_test = out_test_fin2['patient_id'].get_spans()
    spans_asmt = out_asmt_init['patient_id'].get_spans()

    print('Create filtering of dates')

    dict_max = {}
    for f in list_symptoms:
        dict_max[f] = []
    dict_max['healthy_first'] = []

    # ====
    # out_asmt_init x patients that start healthy
    # ====
    sum_symp = np.zeros(len(out_asmt_init['patient_id'].data))
    for k in list_symptoms:
        values = out_asmt_init[k].data[:]
        print(k, np.unique(values))
        values = np.where(values > 1, np.ones_like(values), np.zeros_like(values))
        sum_symp += values

    first_sum = sum_symp[spans_asmt[:-1]]  # first record of each patient
    first_healthy = first_sum == 0
    print(np.sum(first_healthy), ' number of start healthy')

    with utils.Timer('pruning to assessments that are 14 days before test'):
        filt_date, filt_testwithasmt, dict_max = date_filtering(spans_test, spans_asmt, dict_max, out_test_fin, out_asmt_init, first_healthy)

    # ====
    # out_asmt_filt 1 copy from out_asmt_init & assessments 14 days before test
    # ====
    out_asmt_filt = output.create_dataframe('asmt_filtered')
    for f in list_symptoms + ['created_at', ]:
        fld = out_asmt_init[f].create_like(out_asmt_filt, f, ts)
        out_asmt_init[f].apply_filter(filt_date, target=fld)

    print('pruning done')
    out_test_filt = output.create_dataframe('test_filtered')

    # ====
    # out_test_filt 1 copy from symptom data
    # ====
    idx = out_test_filt.create_numeric('r', 'int')  # the index field when write to csv
    idx.data.write(list(range(len(dict_max['healthy_first']))))
    for k in list_symptoms:
        # writer = ds.get_numeric_writer(out_test_filt, k, timestamp=ts, dtype='int64')
        if k in out_test_filt.keys():
            writer = out_test_filt[k].data
        else:
            writer = out_test_filt.create_numeric(k, 'int64', timestamp=ts).data
        writer.write(np.asarray(dict_max[k]))
    out_test_filt.create_numeric('healthy_first', 'bool', timestamp=ts).data.write(dict_max['healthy_first'])

    # ====
    # out_test_filt 2 copy from out_test_fin2 & assessments 14 days before test
    # ====
    for k in out_test_fin2.keys():
        # reader = ds.get_reader(out_test_fin2[k])
        # writer = reader.get_writer(out_test_filt, k, ts)
        # ds.apply_filter(filt_testwithasmt.astype('bool'), reader, writer)
        fld = out_test_fin2[k].create_like(out_test_filt, k, ts)
        out_test_fin2[k].apply_filter(filt_testwithasmt.astype('bool'), target=fld)
    out_fields = ['r'] + list_symptoms + ['healthy_first', 'converted_test', 'date_effective_test', 'patient_id',
                                  'pcr_standard', 'result']
    save_df_to_csv(out_test_filt, '/home/jd21/data/MaxBefTestForLongAll.csv', out_fields)

    # filter information on patients
    # ====
    # out_pat 1 patients that in out_test_filt
    # ====
    src_pat = source['patients']
    out_pat = output.create_dataframe('patients_characteristics')
    pat_id = src_pat['id'].data
    filt_patfin = prst.foreign_key_is_in_primary_key(out_test_filt['patient_id'].data[:], pat_id[:])
    list_fields = ['id', 'year_of_birth', 'gender', 'healthcare_professional']
    idx = out_pat.create_numeric('r', 'int')  # the index field when write to csv
    idx.data.write(list(range(np.sum(filt_patfin == True))))  # mind == can not be replaced by is

    for k in list_fields:
        fld = src_pat[k].create_like(out_pat, k, ts)
        src_pat[k].apply_filter(filt_patfin, target=fld)
    save_df_to_csv(out_pat, '/home/jd21/data/PatBefTestForLongAll.csv', ['r'] + list_fields)


if __name__ == "__main__":
    with session.Session() as s:
        source = s.open_dataset('/home/jd21/data/post.h5', 'r', 'source')
        output = s.open_dataset('/home/jd21/data/out2.hdf5', 'w', 'output')
        max_symp_up14days_beforetest(s, source, output)


