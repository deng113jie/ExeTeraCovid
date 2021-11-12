import os
import time
from datetime import datetime

import numpy as np

from exetera.core.session import Session
import exetera.core.dataset as ds
import exetera.core.dataframe as df
import exetera.core.operations as ops

def output_kerstin_csv(src_filename, dst_filename, date_to, date_from):
    """
    illness_vacc merge patients to get healthcare_professional,age,age_decade,gender,nr_comorbidities,has_comorbidities,obesity,bmi,
    Then merge vaccine_doses to get brand, date_taken_specific,
    Then merge tests to get result_coded
    problem: imd_bin
    :param src_ds:
    :param dst_ds:
    :return:
    """
    with Session() as s:
        src_ds = s.open_dataset(src_filename, 'r', 'src')
        dst_ds = s.open_dataset(dst_filename, 'w', 'dst')
        s_tests = src_ds['tests']



        #3 with tests as
        print(datetime.now(),'Working on tests...')
        ds.copy(s_tests, dst_ds, 'tests')
        filter = dst_ds['tests']['date_taken_specific'].data[:] < datetime.strptime(date_to, '%Y%m%d').timestamp()
        filter &= dst_ds['tests']['updated_at'].data[:] < datetime.strptime(date_to, '%Y%m%d').timestamp()
        filter &= np.isin(dst_ds['tests']['result'].data[:], [3, 4])
        filter &= np.isin(dst_ds['tests']['mechanism'].data[:], [1, 2, 3, 4])
        dst_ds['tests'].apply_filter(filter)
        dst_ds['tests'].create_numeric('result_coded', 'int32')
        dst_ds['tests']['result_coded'].data.write(np.where(dst_ds['tests']['result'].data[:]==4, 1, 0))

        #22 add_previous_infection
        print(datetime.now(),'Working on add_previous_infection...')
        add_previous_infection = dst_ds.create_dataframe('add_previous_infection')
        span = dst_ds['tests']['patient_id'].get_spans()
        distinct_pid = span[:-1]

        first_pos_date = np.zeros(len(span)-1, float)
        previous_pos_test_date = np.zeros(len(span) - 1, float)

        for i in range(len(span)-1):
            pos_dates = dst_ds['tests']['date_taken_specific'].data[span[i]:span[i+1]][dst_ds['tests']['result_coded'].data[span[i]:span[i+1]] == 1]
            first_pos_date[i] = np.min(pos_dates) if len(pos_dates>0) else np.nan
            # max date 'between unbounded preceding and 1 preceding', which is second greatest date
            dates = np.sort(dst_ds['tests']['date_taken_specific'].data[span[i]:span[i + 1]][dst_ds['tests']['result_coded'].data[span[i]:span[i + 1]] == 1])
            previous_pos_test_date[i] = dates[-2] if len(dates)>1 else np.nan

        #filter instead of copy/write to make it quicker
        dst_ds['tests']['patient_id'].create_like(add_previous_infection, 'patient_id')
        dst_ds['tests']['patient_id'].apply_index(distinct_pid, add_previous_infection['patient_id'])

        dst_ds['tests']['date_taken_specific'].create_like(add_previous_infection, 'date_taken_specific')
        dst_ds['tests']['date_taken_specific'].apply_index(distinct_pid, add_previous_infection['date_taken_specific'])

        dst_ds['tests']['result_coded'].create_like(add_previous_infection, 'result_coded')
        dst_ds['tests']['result_coded'].apply_index(distinct_pid, add_previous_infection['result_coded'])

        dst_ds['tests']['date_taken_specific'].create_like(add_previous_infection, 'first_pos_date')
        add_previous_infection['first_pos_date'].data.write(first_pos_date)

        dst_ds['tests']['date_taken_specific'].create_like(add_previous_infection, 'previous_pos_test_date')
        add_previous_infection['previous_pos_test_date'].data.write(previous_pos_test_date)

        #41 tag_first_episode
        print(datetime.now(),'Working on tag_first_episode...')

        add_previous_infection.create_numeric('days_since_first_pos', 'int32')
        add_previous_infection['days_since_first_pos'].data.write(add_previous_infection['date_taken_specific']-add_previous_infection['first_pos_date'])
        add_previous_infection.create_numeric('days_since_last_pos', 'int32')
        add_previous_infection['days_since_last_pos'].data.write(add_previous_infection['date_taken_specific'] - add_previous_infection['previous_pos_test_date'])

        #53 rank_pos_tests
        print(datetime.now(),'Working on rank_pos_tests...')
        ds.copy(add_previous_infection, dst_ds, 'rank_pos_tests')
        rank_pos_tests = dst_ds['rank_pos_tests']
        filter = rank_pos_tests['result_coded'].data[:] == 1
        filter2 = rank_pos_tests['days_since_last_pos'] >= 90
        filter2 |=  rank_pos_tests['days_since_first_pos'] == 0
        filter &= filter2.data[:]
        rank_pos_tests.apply_filter(filter)
        rank_pos_tests_distinct = dst_ds.create_dataframe('rank_pos_tests_distinct')
        rank_pos_tests.drop_duplicates(by = ['patient_id','date_taken_specific', 'result_coded', 'days_since_first_pos',
                                             'days_since_last_pos'], ddf=rank_pos_tests_distinct)

        episode_number = np.zeros(len(rank_pos_tests_distinct['patient_id']))
        span = rank_pos_tests_distinct['patient_id'].get_spans()
        for i in range(len(span)-1):
            rank = np.argsort(rank_pos_tests_distinct['days_since_first_pos'].data[span[i]:span[i+1]])
            episode_number[span[i]:span[i+1]] = rank  # note here is slightly differnt from DENSE_RANK on equal values
        rank_pos_tests_distinct.create_numeric('episode_number', 'int32')
        rank_pos_tests_distinct['episode_number'].data.write(episode_number)

        # 70 number_all_episodes
        print(datetime.now(), 'Working on number_all_episodes...')
        number_all_episodes = dst_ds.create_dataframe('number_all_episodes')
        df.merge(add_previous_infection, rank_pos_tests_distinct, dest=number_all_episodes, how='left', left_on='patient_id',
                 right_on='patient_id', left_fields=['patient_id', 'date_taken_specific', 'result_coded', 'days_since_first_pos', 'first_pos_date'],
                 right_fields=['patient_id', 'episode_number', 'date_taken_specific'])
        filter = number_all_episodes['date_taken_specific_l'] >= number_all_episodes['date_taken_specific_r']
        number_all_episodes.apply_filter(filter)
        number_all_episodes_max = dst_ds.create_dataframe('number_all_episodes_max')
        number_all_episodes.groupby(by=['patient_id_l', 'date_taken_specific_l', 'result_coded', 'days_since_first_pos',
                                        'first_pos_date']).max(['episode_number', 'date_taken_specific_r'], ddf=number_all_episodes_max)

        number_all_episodes_max['patient_id'] = number_all_episodes_max['patient_id_l']
        number_all_episodes_max['date_taken_specific'] = number_all_episodes_max['date_taken_specific_l']
        number_all_episodes_max['episode_start_date'] = number_all_episodes_max['date_taken_specific_r_max']
        number_all_episodes_max['episode_number'] = number_all_episodes_max['episode_number_max']
        # 89 mark_sickness_intervals
        print(datetime.now(), 'Working on mark_sickness_intervals...')
        keep_test = np.where((number_all_episodes_max['date_taken_specific'].data[:] != number_all_episodes_max['episode_start_date'].data[:])
                             & (abs(number_all_episodes_max['date_taken_specific'].data[:]-number_all_episodes_max['episode_start_date'].data[:])<90*24*3600),
                             0, 1)
        number_all_episodes_max.create_numeric('keep_test', 'int32')
        number_all_episodes_max['keep_test'].data.write(keep_test)

        episode_start = np.where(number_all_episodes_max['date_taken_specific'].data[:] != number_all_episodes_max['episode_start_date'].data[:], 1, 0)
        number_all_episodes_max.create_numeric('episode_start', 'int32')
        number_all_episodes_max['episode_start'].data.write(episode_start)

        reinfection_date = np.where((number_all_episodes_max['date_taken_specific'].data[:] == number_all_episodes_max['episode_start_date'].data[:])
                                    & (number_all_episodes_max['episode_number'].data[:] == 2), number_all_episodes_max['date_taken_specific'].data[:], np.nan)

        span = number_all_episodes_max['patient_id'].get_spans()
        for i in range(len(span)-1):
            reinfection_date[span[i]:span[i+1]] = min(reinfection_date[span[i]:span[i+1]])

        number_all_episodes_max['date_taken_specific'].create_like(number_all_episodes_max, 'reinfection_date')
        number_all_episodes_max['reinfection_date'].data.write(reinfection_date)

        number_all_episodes_m_distinct = dst_ds.create_dataframe('number_all_episodes_m_distinct')

        number_all_episodes_max.drop_duplicates(by = ['patient_id', 'date_taken_specific', 'result_coded',
                                                       'episode_number', 'days_since_first_pos', 'first_pos_date',
                                                       'keep_test', 'episode_start', 'reinfection_date'],
                                                ddf=number_all_episodes_m_distinct)

        #103 filter_sickness_intervals
        print(datetime.now(), 'Working on filter_sickness_intervals...')
        filter = number_all_episodes_m_distinct['keep_test'].data[:] == 1
        filter2 = number_all_episodes_m_distinct['episode_number'].data[:] == np.nan
        filter2 |= number_all_episodes_m_distinct['episode_number'].data[:] == 1
        filter2 |= (number_all_episodes_m_distinct['episode_number'].data[:] == 2) & (number_all_episodes_m_distinct['episode_start'].data[:] == 1)
        filter &= filter2
        number_all_episodes_m_distinct.apply_filter(filter)

        #number_all_episodes_m_distinct['episode_number'].data[np.isnan(number_all_episodes_m_distinct['episode_number'].data[:])] = 0
        number_all_episodes_m_distinct['episode_number'].data[:] = np.nan_to_num(number_all_episodes_m_distinct['episode_number'].data[:])
        previously_infected = np.where(number_all_episodes_m_distinct['episode_number'].data[:] >= 1, 1, 0)
        number_all_episodes_m_distinct.create_numeric('previously_infected', 'int32')
        number_all_episodes_m_distinct['previously_infected'].data.write(previously_infected)

        #139 filter_test_timeframe
        print(datetime.now(), "Working on filter_test_timeframe...")
        filter = number_all_episodes_m_distinct['date_taken_specific'].data[:]  >= datetime.strptime(date_to, '%Y%m%d').timestamp()
        filter &= number_all_episodes_m_distinct['date_taken_specific'].data[:]  >= datetime(2020, 12, 1).timestamp()
        number_all_episodes_m_distinct.apply_filter(filter)

        #patients_with_tests
        print(datetime.now(), "Working on patients_with_tests...")
        s_pat = src_ds['patients']
        filter = s_pat['country_code'].data[:] == b'GB'
        filter &= (15 < s_pat['age'].data[:]) & (s_pat['age'].data[:] < 100)
        filter &= (15 < s_pat['bmi'].data[:]) & (s_pat['bmi'].data[:] < 55)
        filter &= np.isin(s_pat['gender'].data[:], [0,1])
        d_pat = dst_ds.create_dataframe('patients')
        s_pat.apply_filter(filter, ddf=d_pat)
        d_pat.create_fixed_string('age_decade',4)
        age_decade = np.where(d_pat['age'].data[:]<40, b'<40',np.where(d_pat['age'].data[:]>=40, b'>=40', b'miss'))
        d_pat['age_decade'].data.write(age_decade)

        d_pat.create_numeric('nr_comorbidities','int32')
        nr_comorbidities = np.zeros(len(d_pat['has_diabetes'].data))
        for k in ['has_diabetes', 'has_heart_disease', 'has_lung_disease', 'does_chemotherapy', 'has_kidney_disease', 'has_cancer', 'takes_immunosuppressants']:
            nr_comorbidities += np.where(d_pat[k].data[:] == 2, 1, 0)
        d_pat['nr_comorbidities'].data.write(nr_comorbidities)

        d_pat.create_numeric('obesity', 'int8')
        d_pat['obesity'].data.write(np.where(d_pat['bmi'].data[:]<30,0,1))

        #d_pat.create_numeric('healthcare_professional_s','int16')
        healthcare_professional_s = np.zeros(len(d_pat['healthcare_professional'].data), bool)
        for k in ['have_worked_in_hospital_care_facility', 'have_worked_in_hospital_clinic', 'have_worked_in_hospital_home_health',
                  'have_worked_in_hospital_inpatient', 'have_worked_in_hospital_other', 'have_worked_in_hospital_outpatient', 'have_worked_in_hospital_school_clinic']:
            healthcare_professional_s |= np.where(d_pat['have_worked_in_hospital_care_facility'] == 2, True, False)
        healthcare_professional_s = np.where(np.isin(d_pat['healthcare_professional'].data[:], [2,3,4,5]) | (healthcare_professional_s),
                                           1, np.where(d_pat['healthcare_professional'].data[:]==0, 0, 999))
        d_pat['healthcare_professional'].data[:] = healthcare_professional_s
        # TODO patient_id is null?
        # TODO imd
        patients_with_tests = dst_ds.create_dataframe('patients_with_tests')
        df.merge(number_all_episodes_m_distinct, d_pat, dest=patients_with_tests,  left_on='patient_id', right_on='id',
                 right_fields=['id', 'age', 'age_decade', 'nr_comorbidities', 'obesity', 'bmi', 'gender', 'healthcare_professional'])

        # 206 both_doses_wide
        print(datetime.now(), "Working on both_doses_wide...")
        s_vaccds = src_ds['vaccine_doses']
        filter = s_vaccds['date_taken_specific'].data[:] > 0
        filter &= s_vaccds['country_code'].data[:] == b'GB'
        filter &= s_vaccds['updated_at'].data[:] < datetime.strptime(date_to, '%Y%m%d').timestamp()
        filter &= s_vaccds['date_taken_specific'].data[:] < datetime.strptime(date_to, '%Y%m%d').timestamp()
        d_vaccds = dst_ds.create_dataframe('vaccine_doses')
        s_vaccds.apply_filter(filter, ddf=d_vaccds)
        d_vaccds_uniq = dst_ds.create_dataframe('vaccine_doses_uniq')
        d_vaccds.drop_duplicates(by=['patient_id', 'vaccine_id'], ddf=d_vaccds_uniq)

        sorted_by_fields_data = np.asarray([d_vaccds[k].data[:] for k in ['patient_id', 'vaccine_id']])
        spans = ops._get_spans_for_multi_fields(sorted_by_fields_data)
        first_dose_date = np.zeros(len(spans) - 1, float)
        second_dose_date = np.zeros(len(spans) - 1, float)
        brand = np.zeros(len(spans) - 1, 'int8')
        nr_vaccines_logged = np.zeros(len(spans) - 1, 'int8')

        p_span = d_vaccds['patient_id'].get_spans()
        j = 0
        d_vaccds['patient_id'].data[p_span[j]]
        nr_vacc = len(np.unique(d_vaccds['vaccine_id'].data[p_span[j]:p_span[j + 1]]))

        for i in range(len(spans) - 1):
            first_dose_date[i] = max(np.where(d_vaccds['sequence'].data[spans[i]:spans[i + 1]] == 1,
                                              d_vaccds['date_taken_specific'].data[spans[i]:spans[i + 1]], 0))
            second_dose_date[i] = max(np.where(d_vaccds['sequence'].data[spans[i]:spans[i + 1]] == 2,
                                               d_vaccds['date_taken_specific'].data[spans[i]:spans[i + 1]], 0))
            brand[i] = max(
                np.where(d_vaccds['date_taken_specific'].data[spans[i]:spans[i + 1]] < datetime(2021, 1, 4).timestamp(),
                         2, d_vaccds['brand'].data[spans[i]:spans[i + 1]]))
            if d_vaccds['patient_id'].data[spans[i]] != d_vaccds['patient_id'].data[p_span[j]]:  # next pid
                nr_vacc = len(np.unique(d_vaccds['vaccine_id'].data[p_span[j]:p_span[j + 1]]))
                j += 1
                nr_vaccines_logged[i] = nr_vacc
            else:
                nr_vaccines_logged[i] = nr_vacc

        d_vaccds_uniq.create_timestamp('first_dose_date').data.write(first_dose_date)
        d_vaccds_uniq.create_timestamp('second_dose_date').data.write(second_dose_date)
        d_vaccds_uniq.create_numeric('brand', 'int8').data.write(brand)
        d_vaccds_uniq.create_numeric('nr_vaccines_logged', 'int8').data.write(nr_vaccines_logged)

        # 221 first dose
        print(datetime.now(), 'Working on first_dose...')
        d_vaccds_uniq.create_fixed_string('latest_vaccine_status', 9)
        # todo first == 0 won't check second?
        latest_vaccine_status = np.where((d_vaccds_uniq['nr_vaccines_logged'].data[:]!= 1)
                                         | ((d_vaccds_uniq['second_dose_date'].data[:]-d_vaccds_uniq['first_dose_date'].data[:])<3600*24*15)
                                         | ((d_vaccds_uniq['first_dose_date'].data[:] < datetime(2020, 12, 8).timestamp()) |  (d_vaccds_uniq['second_dose_date'].data[:] < datetime(2020, 12, 8).timestamp()))
                                         | (d_vaccds_uniq['first_dose_date'].data[:] == 0), b'invalid',
                                         np.where((d_vaccds_uniq['first_dose_date'].data[:] != 0) & (d_vaccds_uniq['second_dose_date'].data[:] != 0), b'fully',
                                                  np.where((d_vaccds_uniq['first_dose_date'].data[:] != 0) & (d_vaccds_uniq['second_dose_date'].data[:] == 0), b'partially', b'invalid')))
        d_vaccds_uniq['latest_vaccine_status'].data.write(latest_vaccine_status)

        #241 vaccined
        print(datetime.now(), 'Working on vaccinated...')
        vaccinated = dst_ds.create_dataframe('vaccinated')
        df.merge(patients_with_tests, d_vaccds_uniq, dest=vaccinated, left_on='patient_id', right_on='patient_id')
        tested_date_week = [time.mktime((dt - timedelta(dt.weekday())).date().timetuple()) for dt in vaccinated['date_taken_specific'].data[:]]
        vaccinated.create_timestamp('tested_date_week').data.write(tested_date_week)

        vaccine_status = np.where(vaccinated['latest_vaccine_status'].data[:] == b'invalid', b'inval',
                                  np.where(vaccinated['date_taken_specific'].data[:] < vaccinated['first_dose_date'].data[:], b'unvac',
                                           np.where((vaccinated['date_taken_specific'].data[:] >= vaccinated['first_dose_date'].data[:])
                                                    & ((vaccinated['date_taken_specific'].data[:] < vaccinated['second_dose_date'].data[:]) | (vaccinated['latest_vaccine_status'].data[:] == b'partially')), b'parti',
                                                    np.where((vaccinated['date_taken_specific'].data[:] >= vaccinated['second_dose_date'].data[:])&(vaccinated['latest_vaccine_status'].data[:] == b'fully'), b'fully', b'inval'))))
        vaccinated.create_fixed_string('vaccine_status', 5).data.write(vaccine_status)

        vaccine_status_coded = np.where(vaccine_status==b'inval', -99,
                                        np.where(vaccine_status==b'unvac',-100,
                                                 np.where(vaccine_status==b'parti', 1,
                                                          np.where(vaccine_status==b'fully', 2, -99))))
        vaccinated.create_numeric('vaccine_status_coded','int16').data.write(vaccine_status_coded)

        vaccinated_distinct = dst_ds.create_dataframe('vaccinated_distinct')
        vaccinated.drop_duplicates(by=['patient_id', 'date_taken_specific', 'result_coded', 'episode_number', 'previously_infected',
                                       'first_pos_date', 'reinfection_date', 'days_since_first_pos', 'episode_start', 'age', 'age_decade',
                                       'nr_comorbidities', 'obesity', 'bmi', 'gender',  'healthcare_professional', 'tested',
                                       'tested_date_week', 'vaccine_status', 'vaccine_status_coded', 'first_dose_date',
                                       'second_dose_date', 'brand'], ddf=vaccinated_distinct)

        #268 compute_days_since_vaccine
        print(datetime.now(),'Working on compute_days_since_vaccine...')
        days_since_vaccinated= np.where(vaccinated_distinct['vaccine_status']==b'inval', -99,
                                        np.where(vaccinated_distinct['vaccine_status']==b'unvac', -100,
                                                  np.where(vaccinated_distinct['vaccine_status']==b'parti', (vaccinated_distinct['date_taken_specific'].data[:]-vaccinated_distinct['first_dose_date'].data[:])/86400,
                                                           np.where(vaccinated_distinct['vaccine_status']==b'fully', (vaccinated_distinct['date_taken_specific'].data[:]-vaccinated_distinct['second_dose_date'].data[:])/86400,-99))))
        vaccinated_distinct.create_numeric('days_since_vaccinated','int16').data.write(days_since_vaccinated)

        has_comorbidities = np.where(vaccinated_distinct['nr_comorbidities']==0,0,1)
        vaccinated_distinct.create_numeric('has_comorbidities','int8').data.write(has_comorbidities)

        #304 bin_days_since_vaccine
        print(datetime.now(),'Working on bin_days_since_vaccine...')
        days_since_vaccinated_bin = np.where(vaccinated_distinct['vaccine_status'].data[:]==b'inval', - 99,
                                             np.where(vaccinated_distinct['vaccine_status'].data[:]==b'unvac'),-100,
                                             np.where((np.isin(vaccinated_distinct['vaccine_status'].data[:],[b'parti',b'fully'])) & (vaccinated_distinct['days_since_vaccinated'].data[:] < 14 ), -99,
                                                      np.where(np.isin(vaccinated_distinct['vaccine_status'].data[:],[b'parti',b'fully']),int(vaccinated_distinct['days_since_vaccinated'].data[:]/30),-99)))
        vaccinated_distinct.create_numeric('days_since_vaccinated_bin', 'int16').data.write(days_since_vaccinated_bin)

        #output csv
        print(datetime.now(), 'Output to csv...')
        vaccinated_distinct.to_csv('test.csv')

        #335 weekly_status
        print(datetime.now(), 'Working on weekly_status...')
        weekly_status = dst_ds.create_dataframe('weekly_status')
        vaccinated_distinct.groupby(by = ['patient_id', 'healthcare_professional', 'age', 'age_decade', 'gender', 'nr_comorbidities',
                                          'has_comorbidities', 'obesity', 'bmi', 'brand', 'imd_bin', 'tested_date_week']).max(target=['first_dose_date',
                                          'second_dose_date','vaccine_status_coded','days_since_vaccinated','days_since_vaccinated_bin',
                                          'result_coded','episode_number','previously_infected','episode_start','reinfection_date',
                                          'first_pos_date','days_since_first_pos'],ddf=weekly_status)
        weekly_status.to_csv('weekly_status.csv')





































if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True, help="the path of the source dataset, which contains the patients, tests and vaccine_does tables.")
    parser.add_argument('-d', required=True, help="the path / name of the destination dataset.")
    parser.add_argument('-t', required=True, help="the date_to parameter, in format yyyymmdd, e.g. 20211004")
    parser.add_argument('-f', required=True, help="the date_from parameter, in format yyyymmdd, e.g. 20211004")

    args = parser.parse_args()

    output_kerstin_csv(args.s, args.d, args.t, args.f)
