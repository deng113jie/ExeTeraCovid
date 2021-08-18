import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from exetera.core.session import Session
from exetera.core import utils, dataframe, dataset
from exetera.core import persistence as prst
from exeteracovid.algorithms.test_type_from_mechanism import test_type_from_mechanism_v1_standard_input
from exeteracovid.algorithms.covid_test_date import covid_test_date_v1
from exeteracovid.algorithms.covid_test import multiple_tests_start_with_negative_v1
from exeteracovid.algorithms.filter_syptom import sum_up_symptons_v1, filter_asymp_and_firstnz_v1
from exeteracovid.algorithms.test_type_from_mechanism import pcr_standard_summarize_v1


def save_df_to_csv(df, csv_name, chunk=200000):  # chunk=100k ~ 20M/s
    with open(csv_name, 'w', newline='') as csvfile:
        columns = list(df.keys())
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        field1 = columns[0]
        for current_row in range(0, len(df[field1].data), chunk):
            torow = current_row + chunk if current_row + chunk < len(df[field1].data) else len(df[field1].data)
            batch = list()
            for k in df.keys():
                batch.append(df[k].data[current_row:torow])
            writer.writerows(list(zip(*batch)))


def merging_results(s, source, output):
    list_symptoms = ['abdominal_pain', 'altered_smell', 'blisters_on_feet', 'brain_fog',
                     'chest_pain', 'chills_or_shivers', 'delirium', 'diarrhoea',
                     'diarrhoea_frequency', 'dizzy_light_headed', 'ear_ringing', 'earache',
                     'eye_soreness', 'fatigue', 'feeling_down', 'fever', 'hair_loss',
                     'headache', 'headache_frequency', 'hoarse_voice',
                     'irregular_heartbeat', 'loss_of_smell', 'nausea', 'persistent_cough', 'rash',
                     'red_welts_on_face_or_lips', 'runny_nose',
                     'shortness_of_breath', 'skin_burning', 'skipped_meals', 'sneezing',
                     'sore_throat', 'swollen_glands', 'typical_hayfever', 'unusual_muscle_pains']

    path = '/home/jd21/data'
    #ds = DataStore()
    ts = str(datetime.now(timezone.utc))

    # # Same but for test
    src_test = source['tests']
    list_testid = src_test['patient_id']
    list_testcreate = src_test['created_at']
    out_test = output.create_dataframe('tests')
    # ====
    # out_test step 1 copy from src_test
    # ====
    with utils.Timer('applying sort'):
        for k in src_test.keys():
            dataframe.copy(src_test[k], out_test, k)

    # convert test date
    covid_test_date_v1(s, out_test, out_test, 'date_effective_test')

    # Filtering only definite results

    results_raw = out_test['result'].data[:]
    results_filt = np.where(np.logical_or(results_raw == 4, results_raw == 3), True, False)
    for k in out_test.keys():
        out_test[k].apply_filter(results_filt, in_place=True)

    # Filter check
    # sanity_filter = (date_fin == 0)
    # print(np.sum(sanity_filter))

    # Creating clean mechanism
    reader_mec = out_test['mechanism'].data
    s_reader_mec = s.get(out_test['mechanism'])

    print(len(reader_mec), len(out_test['patient_id'].data))

    reader_ftmec = out_test['mechanism_freetext'].data
    s_reader_ftmec = s.get(out_test['mechanism_freetext'])

    test_type_from_mechanism_v1_standard_input(s, out_test)

    pcr_standard_summarize_v1(s, out_test)

    out_test_fin = output.create_dataframe('tests_fin')
    # ====
    # out_test_fin step 1 copy from out_test
    # ====
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

    # Taking care of the old test
    src_asmt = source['assessments']
    print(src_asmt.keys())

    # # Remap had_covid_test to 0/1 2 to binary 0,1
    # tcp_flat = np.where(src_asmt['tested_covid_positive'].data[:] < 1, 0, 1)
    # spans = src_asmt['patient_id'].get_spans()
    # # Get the first index at which the hct field is maximum
    # firstnz_tcp_ind = ds.apply_spans_index_of_max(spans, tcp_flat)
    # # Get the index of first element of patient_id when sorted
    # first_hct_ind = spans[:-1]
    # filt_tl = first_hct_ind != firstnz_tcp_ind
    # # Get the indices for which hct changed value (indicating that test happened after the first input)
    # sel_max_ind = ds.apply_filter(filter_to_apply=filt_tl, reader=firstnz_tcp_ind)
    # # Get the index at which test is maximum and for which that hct is possible
    # # max_tcp_ind = ds.apply_spans_index_of_max(spans, src_asmt['tested_covid_positive'].data[:])
    # # filt_max_test = ds.apply_indices(filt_tl, max_tcp )
    # sel_max_tcp = ds.apply_indices(filt_tl, firstnz_tcp_ind)
    # sel_maxtcp_ind = ds.apply_filter(filter_to_apply=filt_tl, reader=firstnz_tcp_ind)
    # # Define usable assessments with correct test based on previous filter on indices

    sel_max_ind, sel_max_tcp = multiple_tests_start_with_negative_v1(s, src_asmt)

    usable_asmt_tests = output.create_group('usable_asmt_tests')
    # ====
    # usable_asmt_tests step 1: copy from src_asmt, filter patients w/ multiple test and first ok
    # ====
    for k in ('id', 'patient_id', 'created_at', 'had_covid_test'):
        fld = src_asmt[k].create_like(usable_asmt_tests, k)
        src_asmt[k].apply_index(sel_max_ind, target=fld)
        print(usable_asmt_tests[k].data[0])

    src_asmt['created_at'].create_like(usable_asmt_tests, 'eff_result_time')
    src_asmt['created_at'].apply_index(sel_max_tcp, target=usable_asmt_tests['eff_result_time'])

    src_asmt['tested_covid_positive'].create_like(usable_asmt_tests, 'eff_result')
    src_asmt['tested_covid_positive'].apply_index(sel_max_tcp, target=usable_asmt_tests['eff_result'])

    src_asmt['tested_covid_positive'].create_like(usable_asmt_tests, 'tested_covid_positive')
    src_asmt['tested_covid_positive'].apply_index(sel_max_tcp, target=usable_asmt_tests['tested_covid_positive'])

    # ====
    # usable_asmt_tests step 2: filter only positive
    # ====
    # Making sure that the test is definite (either positive or negative)
    filt_deftest = usable_asmt_tests['tested_covid_positive'].data[:] > 1
    # print(len(ds.get_reader(usable_asmt_tests['patient_id'])))
    for k in (
            'id', 'patient_id', 'created_at', 'had_covid_test', 'tested_covid_positive', 'eff_result_time',
            'eff_result'):
        usable_asmt_tests[k].apply_filter(filt_deftest, in_place=True)

    # ====
    # usable_asmt_tests step 3: add delta_days_test, date_final_test, and pcr_standard fields
    # ====
    # Getting difference between created at (max of hct date) and max of test result (eff_result_time)
    reader_hct = usable_asmt_tests['created_at'].data[:]
    reader_tcp = usable_asmt_tests['eff_result_time'].data[:]
    with utils.Timer('doing delta time'):
        delta_time = reader_tcp - reader_hct
        delta_days = delta_time / 86400
    print(delta_days[:10], delta_time[:10])
    writer = usable_asmt_tests.create_numeric('delta_days_test', 'float32')
    writer.data.write(delta_days)

    # Final day of test
    date_final_test = np.where(delta_days < 7, reader_hct, reader_tcp - 2 * 86400)
    writer = usable_asmt_tests.create_timestamp('date_final_test')
    writer.data.write(date_final_test)
    # print(ds.get_reader(usable_asmt_tests['date_final_test'])[:10], date_final_test[:10])

    pcr_standard = np.ones(len(usable_asmt_tests['patient_id'].data))
    writer = usable_asmt_tests.create_numeric('pcr_standard', 'int')
    writer.data.write(pcr_standard)

    # ====
    # out_test_fin step 2 copy from usable_asmt_tests
    # ====
    list_init = ('patient_id', 'date_final_test', 'tested_covid_positive', 'pcr_standard')
    list_final = ('patient_id', 'date_effective_test', 'result', 'pcr_standard')
    # Join
    for (i, f) in zip(list_init, list_final):
        values = usable_asmt_tests[i].data[:]
        if f == 'result':
            values -= 2
        # writers_dict[f] = reader.get_writer(out_test_fin, f, ts)
        print(len(values), f)
        writers_dict[f].write(values)
    writers_dict['converted_test'].write(np.ones(len(usable_asmt_tests['patient_id'].data), dtype='bool'))

    # ====
    # out_pos step 1: copy from out_test_fin, filter valid result, and write to csv
    # ====
    result_fin = out_test_fin['result'].data[:]
    filt_pos = result_fin == 1
    out_pos = output.create_dataframe('out_pos')
    for k in out_test_fin.keys():
        out_test_fin[k].create_like(out_pos, k)
        out_test_fin[k].apply_filter(filt_pos, target=out_pos[k])
        print(k, len(out_test_fin[k].data), len(filt_pos))

    pat_pos_len = len(out_pos['patient_id'].get_spans())-1
    dataset.copy(out_pos, output, 'out_pos_copy')
    save_df_to_csv(out_pos, 'TestedPositiveTestDetails.csv')

    # ====
    # out_pos step 2 filter patient that has assessment
    # ====
    with utils.Timer('Mapping index asmt to pos only'):
        test2pat = prst.foreign_key_is_in_primary_key(out_pos['patient_id'].data[:],
                                                      foreign_key=src_asmt['patient_id'].data[:])

    for f in ['created_at', 'patient_id', 'treatment', 'other_symptoms', 'country_code', 'location',
              'updated_at'] + list_symptoms:
        #print(f)
        if(f in list(out_pos.keys())):
            out_pos[f].data.clear()
            src_asmt[f].apply_filter(test2pat, target=out_pos[f])
        else:
            src_asmt[f].create_like(out_pos, f)
            src_asmt[f].apply_filter(test2pat, target=out_pos[f])


    # print(len(np.unique(ds.get_reader(out_pos['patient_id'])[:])), len(np.unique(pat_pos[:])))
    print(len(out_pos['patient_id'].get_spans())-1, pat_pos_len)
    unique_other, counts = np.unique(out_pos['other_symptoms'].data[:], return_counts=True)
    dict_other = {'other': unique_other, 'counts': counts}

    df_other = pd.DataFrame.from_dict(dict_other)
    df_other.to_csv('OtherSymptoms.csv')

    #  this is duplicated with 265-273
    # for k in list_symptoms:
    #     print(k)
    #     if k in list(out_pos.keys()):
    #         src_asmt[k].apply_filter(test2pat, target=out_pos[k])
    #     else:
    #         src_asmt[k].create_like(out_pos, k)
    #         src_asmt[k].apply_filter(test2pat, target=out_pos[k])
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(out_pos, k,ts,write_mode='overwrite')
        # ds.apply_filter(test2pat, reader,writer)

    # ====
    # summarize the symptoms
    # ====

    # sum_symp = np.zeros(len(out_pos['patient_id'].data))
    # for k in list_symptoms:
    #     values = out_pos[k].data[:]
    #     if k == 'fatigue' or k == 'shortness_of_breath':
    #         values = np.where(values > 2, np.ones_like(values), np.zeros_like(values))
    #     else:
    #         values = np.where(values > 1, np.ones_like(values), np.zeros_like(values))
    #     sum_symp += values
    sum_symp = sum_up_symptons_v1(out_pos)
    out_pos.create_numeric('sum_symp', 'int').data.write(sum_symp)
    # writer = ds.get_numeric_writer(out_pos, 'sum_symp', dtype='int', timestamp=ts, writemode='overwrite')
    # writer.write(sum_symp)

    # ====
    # filter the symptoms
    # ====
    # symp_flat = np.where(out_pos['sum_symp'].data[:] < 1, 0, 1)
    # spans = out_pos['patient_id'].get_spans()
    # print('Number definitie positive is', len(spans) - 1)
    #
    # # Get the first index at which the hct field is maximum
    # firstnz_symp_ind = ds.apply_spans_index_of_max(spans, symp_flat)
    # max_symp_check = symp_flat[firstnz_symp_ind]
    # # Get the index of first element of patient_id when sorted
    #
    # filt_asymptomatic = max_symp_check == 0
    # print('Number asymptomatic is ', len(spans) - 1 - np.sum(max_symp_check), np.sum(filt_asymptomatic))
    #
    # first_symp_ind = spans[:-1]
    # not_healthy_first = first_symp_ind != firstnz_symp_ind
    # print('Number not healthy first is ', len(spans) - 1 - np.sum(not_healthy_first))
    #
    # spans_valid = ds.apply_filter(not_healthy_first, first_symp_ind)
    # pat_sel = ds.apply_indices(spans_valid, out_pos['patient_id'].data[:])
    # filt_sel = prst.foreign_key_is_in_primary_key(pat_sel, out_pos['patient_id'].data[:])
    #
    # spans_asymp = ds.apply_filter(filt_asymptomatic, first_symp_ind)
    spans_asymp, filt_sel = filter_asymp_and_firstnz_v1(s, out_pos)
    # ====
    # out_pos step 3 filter asymptomatic
    # ====
    pat_asymp = out_pos['patient_id'].apply_index(spans_asymp)
    #pat_asymp = ds.apply_indices(spans_asymp, ds.get_reader(out_pos['patient_id']))
    filt_pata = prst.foreign_key_is_in_primary_key(pat_asymp.data[:], out_pos['patient_id'].data[:])

    # ====
    # out_pos_hs step 1 copy from out_pos and apply filter not healthy first
    # ====
    out_pos_hs = output.create_dataframe('out_pos_hs')
    for k in list_symptoms + ['created_at', 'patient_id', 'sum_symp', 'country_code', 'location', 'treatment',
                              'updated_at']:
        #print(k)
        out_pos[k].create_like(out_pos_hs, k)
        out_pos[k].apply_filter(filt_sel, target=out_pos_hs[k])
        # reader = ds.get_reader(out_pos[k])
        # writer = reader.get_writer(out_pos_hs, k, ts)
        # ds.apply_filter(filt_sel, reader, writer)

    # dict_final = {}
    # for k in out_pos_hs.keys():
    #     dict_final[k] = out_pos_hs[k].data[:]
    #
    # df_final = pd.DataFrame.from_dict(dict_final)
    # df_final.to_csv(path + '/PositiveSympStartHealthyAllSymptoms.csv')
    save_df_to_csv(out_pos_hs, 'PositiveSympStartHealthyAllSymptoms.csv')



    print('out_pos_asymp')
    # ====
    # out_pos_as 1 out_pos filter asymptomatic
    # ====
    out_pos_as = output.create_dataframe('out_pos_asymp')
    for k in list_symptoms + ['created_at', 'patient_id', 'sum_symp', 'country_code', 'location',
                              'treatment']:
        out_pos[k].create_like(out_pos_as, k)
        out_pos[k].apply_filter(filt_pata, target=out_pos_as[k])
        # reader = ds.get_reader(out_pos[k])
        # writer = reader.get_writer(out_pos_as, k, ts)
        # ds.apply_filter(filt_pata, reader, writer)

    # dict_finala = {}
    # for k in out_pos_as.keys():
    #     dict_finala[k] = out_pos_as[k].data[:]
    #
    # df_finala = pd.DataFrame.from_dict(dict_finala)
    # df_finala.to_csv(path + '/PositiveAsympAllSymptoms.csv')
    save_df_to_csv(out_pos_as, 'PositiveAsympAllSymptoms.csv')

    # Based on the final selected patient_id, select the appropriate rows of the patient_table
    src_pat = source['patients']
    filt_pat = prst.foreign_key_is_in_primary_key(out_pos_hs['patient_id'].data[:], src_pat['id'].data[:])
    list_interest = ['has_cancer', 'has_diabetes', 'has_lung_disease', 'has_heart_disease', 'has_kidney_disease',
                     'has_asthma',
                     'race_is_other', 'race_is_prefer_not_to_say', 'race_is_uk_asian', 'race_is_uk_black',
                     'race_is_uk_chinese', 'race_is_uk_middle_eastern', 'race_is_uk_mixed_other',
                     'race_is_uk_mixed_white_black', 'race_is_uk_white', 'race_is_us_asian', 'race_is_us_black',
                     'race_is_us_hawaiian_pacific', 'race_is_us_indian_native', 'race_is_us_white', 'race_other',
                     'year_of_birth', 'is_smoker', 'smoker_status', 'bmi_clean', 'is_in_uk_twins',
                     'healthcare_professional', 'gender', 'id', 'blood_group', 'lsoa11cd', 'already_had_covid']
    out_pat = output.create_dataframe('patient_pos')
    print('patient_pos')
    for k in list_interest:
        src_pat[k].create_like(out_pat, k)
        src_pat[k].apply_filter(filt_pat, target=out_pat[k])
        # reader = ds.get_reader(src_pat[k])
        # writer = reader.get_writer(out_pat, k, ts)
        # ds.apply_filter(filt_pat, reader, writer)

    # dict_pat = {}
    # for k in list_interest:
    #     values = out_pat[k].data[:]
    #     dict_pat[k] = values
    #
    # df_pat = pd.DataFrame.from_dict(dict_pat)
    # df_pat.to_csv(path + '/PositiveSympStartHealthy_PatDetails.csv')
    save_df_to_csv(out_pat, 'PositiveSympStartHealthy_PatDetails.csv')


    #spans_asymp = ds.apply_filter(filt_asymptomatic, first_symp_ind)
    #pat_asymp = ds.apply_indices(spans_asymp, ds.get_reader(out_pos['patient_id']))
    pat_asymp = out_pos['patient_id'].apply_index(spans_asymp)
    filt_asymp = prst.foreign_key_is_in_primary_key(pat_asymp.data[:], src_pat['id'].data[:])
    out_pat_asymp = output.create_dataframe('patient_asymp')
    for k in list_interest:
        src_pat[k].create_like(out_pat_asymp, k)
        src_pat[k].apply_filter(filt_asymp, target=out_pat_asymp[k])
        # reader = ds.get_reader(src_pat[k])
        # writer = reader.get_writer(out_pat_asymp, k, ts)
        # ds.apply_filter(filt_asymp, reader, writer)

    # dict_pata = {}
    # for k in list_interest:
    #     values = out_pat_asymp[k].data[:]
    #     dict_pata[k] = values
    #
    # df_pata = pd.DataFrame.from_dict(dict_pata)
    # df_pata.to_csv(path + '/PositiveAsymp_PatDetails.csv')
    save_df_to_csv(out_pat_asymp, 'PositiveAsymp_PatDetails.csv')


if __name__ == "__main__":
    with Session() as s:
        source = s.open_dataset('/home/jd21/data/post.h5', 'r', 'src')
        output = s.open_dataset('/home/jd21/data/May17_processed_mrslt.hdf5', 'w', 'out')
        merging_results(s, source, output)

