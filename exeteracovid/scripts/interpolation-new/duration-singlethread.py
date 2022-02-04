import math
from datetime import datetime, timedelta
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

from numba import njit
import pandas as pd
import numpy as np


from exetera.core.session import Session
import exetera.core.dataframe as df
import exetera.core.dataset as ds
import exetera.core.operations as ops
import exetera.core.fields as flds



list_symptoms = ['altered_smell', 'fatigue', 'abdominal_pain', 'chest_pain', 'sore_throat', 'shortness_of_breath',
                 'nausea',
                 'skipped_meals', 'loss_of_smell', 'unusual_muscle_pains', 'headache', 'hoarse_voice', 'delirium',
                 'diarrhoea',
                 'fever', 'persistent_cough', 'dizzy_light_headed', 'eye_soreness', 'red_welts_on_face_or_lips',
                 'blisters_on_feet', 'chills_or_shivers', 'runny_nose',
                 'sneezing', 'brain_fog', 'swollen_glands', 'rash', 'skin_burning', 'ear_ringing', 'earache',
                 'feeling_down', 'hair_loss', 'irregular_heartbeat']

list_symptoms_bin = list_symptoms + ['sob2', 'fatigue2']

criteria_output = ['patient_id','interval_days','abdominal_pain','altered_smell','blisters_on_feet','brain_fog','chest_pain','chills_or_shivers','delirium','diarrhoea','diarrhoea_frequency','dizzy_light_headed','ear_ringing','earache','eye_soreness','fatigue','feeling_down','fever','hair_loss','headache','headache_frequency','hoarse_voice','irregular_heartbeat','loss_of_smell','nausea','persistent_cough','rash','red_welts_on_face_or_lips','runny_nose','shortness_of_breath','skin_burning','skipped_meals','sneezing','sore_throat','swollen_glands','typical_hayfever','unusual_muscle_pains','created_at','sum_symp','country_code','location','treatment','updated_at','first_entry_sum','last_entry_sum','date_update','sob2','fatigue2','created_interp','date_effective_test','result','pcr_standard','converted_test','health_status','max_symp','health','health_interp','health_back','nan_healthy','nan_uh','count_healthy','count_nans','count_nothealthy','nan_or_health','count_nans_or_health','first_stuh_hg7','latest_first_7','last_stuh_hg7','first_stuh_nhg7','last_stuh_nhg7','date_first_stuh_hg7','date_latest_first_7','delay','dropped','count_uh_nans','count_utoh_nans','count_htouh_nans','max_uh_nans','meeting_post_criteria7','postcrit_ok7','postcrit_aok7','check_altered_smell','sumcheck_altered_smell','check_fatigue','sumcheck_fatigue','check_abdominal_pain','sumcheck_abdominal_pain','check_chest_pain','sumcheck_chest_pain','check_sore_throat','sumcheck_sore_throat','check_shortness_of_breath','sumcheck_shortness_of_breath','check_nausea','sumcheck_nausea','check_skipped_meals','sumcheck_skipped_meals','check_loss_of_smell','sumcheck_loss_of_smell','check_unusual_muscle_pains','sumcheck_unusual_muscle_pains','check_headache','sumcheck_headache','check_hoarse_voice','sumcheck_hoarse_voice','check_delirium','sumcheck_delirium','check_diarrhoea','sumcheck_diarrhoea','check_fever','sumcheck_fever','check_persistent_cough','sumcheck_persistent_cough','check_dizzy_light_headed','sumcheck_dizzy_light_headed','check_eye_soreness','sumcheck_eye_soreness','check_red_welts_on_face_or_lips','sumcheck_red_welts_on_face_or_lips','check_blisters_on_feet','sumcheck_blisters_on_feet','check_chills_or_shivers','sumcheck_chills_or_shivers','check_runny_nose','sumcheck_runny_nose','check_sneezing','sumcheck_sneezing','check_brain_fog','sumcheck_brain_fog','check_swollen_glands','sumcheck_swollen_glands','check_rash','sumcheck_rash','check_skin_burning','sumcheck_skin_burning','check_ear_ringing','sumcheck_ear_ringing','check_earache','sumcheck_earache','check_feeling_down','sumcheck_feeling_down','check_hair_loss','sumcheck_hair_loss','check_irregular_heartbeat','sumcheck_irregular_heartbeat','sick7','sick7_altered_smell','day7_altered_smell','start7_altered_smell','sumsick7_altered_smell','sick7_fatigue','day7_fatigue','start7_fatigue','sumsick7_fatigue','sick7_abdominal_pain','day7_abdominal_pain','start7_abdominal_pain','sumsick7_abdominal_pain','sick7_chest_pain','day7_chest_pain','start7_chest_pain','sumsick7_chest_pain','sick7_sore_throat','day7_sore_throat','start7_sore_throat','sumsick7_sore_throat','sick7_shortness_of_breath','day7_shortness_of_breath','start7_shortness_of_breath','sumsick7_shortness_of_breath','sick7_nausea','day7_nausea','start7_nausea','sumsick7_nausea','sick7_skipped_meals','day7_skipped_meals','start7_skipped_meals','sumsick7_skipped_meals','sick7_loss_of_smell','day7_loss_of_smell','start7_loss_of_smell','sumsick7_loss_of_smell','sick7_unusual_muscle_pains','day7_unusual_muscle_pains','start7_unusual_muscle_pains','sumsick7_unusual_muscle_pains','sick7_headache','day7_headache','start7_headache','sumsick7_headache','sick7_hoarse_voice','day7_hoarse_voice','start7_hoarse_voice','sumsick7_hoarse_voice','sick7_delirium','day7_delirium','start7_delirium','sumsick7_delirium','sick7_diarrhoea','day7_diarrhoea','start7_diarrhoea','sumsick7_diarrhoea','sick7_fever','day7_fever','start7_fever','sumsick7_fever','sick7_persistent_cough','day7_persistent_cough','start7_persistent_cough','sumsick7_persistent_cough','sick7_dizzy_light_headed','day7_dizzy_light_headed','start7_dizzy_light_headed','sumsick7_dizzy_light_headed','sick7_eye_soreness','day7_eye_soreness','start7_eye_soreness','sumsick7_eye_soreness','sick7_red_welts_on_face_or_lips','day7_red_welts_on_face_or_lips','start7_red_welts_on_face_or_lips','sumsick7_red_welts_on_face_or_lips','sick7_blisters_on_feet','day7_blisters_on_feet','start7_blisters_on_feet','sumsick7_blisters_on_feet','sick7_chills_or_shivers','day7_chills_or_shivers','start7_chills_or_shivers','sumsick7_chills_or_shivers','sick7_runny_nose','day7_runny_nose','start7_runny_nose','sumsick7_runny_nose','sick7_sneezing','day7_sneezing','start7_sneezing','sumsick7_sneezing','sick7_brain_fog','day7_brain_fog','start7_brain_fog','sumsick7_brain_fog','sick7_swollen_glands','day7_swollen_glands','start7_swollen_glands','sumsick7_swollen_glands','sick7_rash','day7_rash','start7_rash','sumsick7_rash','sick7_skin_burning','day7_skin_burning','start7_skin_burning','sumsick7_skin_burning','sick7_ear_ringing','day7_ear_ringing','start7_ear_ringing','sumsick7_ear_ringing','sick7_earache','day7_earache','start7_earache','sumsick7_earache','sick7_feeling_down','day7_feeling_down','start7_feeling_down','sumsick7_feeling_down','sick7_hair_loss','day7_hair_loss','start7_hair_loss','sumsick7_hair_loss','sick7_irregular_heartbeat','day7_irregular_heartbeat','start7_irregular_heartbeat','sumsick7_irregular_heartbeat','sick7_fatigue_mild','day7_fatigue_mild','start7_fatigue_mild','sumsick7_fatigue_mild','end7_fatigue_mild','sick7_fatigue_severe','day7_fatigue_severe','start7_fatigue_severe','sumsick7_fatigue_severe','end7_fatigue_severe','sick7_shortness_of_breath_mild','day7_shortness_of_breath_mild','start7_shortness_of_breath_mild','sumsick7_shortness_of_breath_mild','end7_shortness_of_breath_mild','sick7_shortness_of_breath_severe','day7_shortness_of_breath_severe','start7_shortness_of_breath_severe','sumsick7_shortness_of_breath_severe','end7_shortness_of_breath_severe']

TEMPH5 = None


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
                # if isinstance(df[k], flds.TimestampField):
                #     batch.append([get_ts_str(d) for d in df[k].data[current_row:torow]])
                # else:
                batch.append(df[k].data[current_row:torow])
            writer.writerows(list(zip(*batch)))


@njit
def reduce_symptom(onesymp, sum, spans):
    for i in range(len(spans) - 1):
        onesymp[spans[i]] = np.max(onesymp[spans[i]:spans[i + 1]]) if np.any(onesymp[spans[i]:spans[i + 1]] > 0) else 0
    sum += np.where(onesymp>0, 1, 0)
    return onesymp, sum


def creating_nanvalues(df_train):
    sum_symp = np.zeros(len(df_train['patient_id'].data),'float')
    for f in list_symptoms:
        data = df_train[f].data[:]
        data -= 1
        data = np.where(data == -1, 0, data)
        df_train[f].data.clear()
        df_train[f].data.write(data)
        sum_symp+=data
    del df_train['sum_symp']
    df_train.create_numeric('sum_symp', 'float').data.write(sum_symp)
    return df_train


def creating_nanvalues_and_aggregate(df_train):
    sum_symp = np.zeros(len(df_train['patient_id'].data),'float')
    sorted_by_fields_data = np.asarray([df_train[k].data[:] for k in ['patient_id', 'interval_days']])
    spans = ops._get_spans_for_multi_fields(sorted_by_fields_data)
    for f in list_symptoms:
        data = df_train[f].data[:]
        data -= 1
        data = np.where(data == -1, 0, data)
        data, sum_symp = reduce_symptom(data, sum_symp, spans)
        df_train[f].data.clear()
        df_train[f].data.write(data)
    del df_train['sum_symp']
    df_train.create_numeric('sum_symp', 'float').data.write(sum_symp)
    return df_train

# @njit  # no use any more
# def single_interpolate(old_index, old_data, new_index, limit_area='inner'):
#     # expand the index
#     result = np.full(len(new_index), np.nan, 'float')
#     for i in range(len(old_index)):
#         idx = np.argwhere(new_index==old_index[i])
#         result[idx[0][0]] = old_data[i]
#     # interpolate
#     if limit_area=='inner':
#         for i in range(0, len(old_index)-1):
#             if old_index[i+1] - old_index[i] > 1:  # has nan in between
#                 coeff = float(old_data[i+1] - old_data[i])/float(old_index[i+1] - old_index[i])
#                 for j in range(old_index[i]+1, old_index[i+1]):
#                     idx = np.argwhere(new_index == j)[0][0]
#                     result[idx] = old_data[i] + (j-old_index[i])*coeff
#     return result


@njit
def transform_method(data, spans, method):
    result = np.zeros(len(data), data.dtype)
    if method == 'min':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.nanmin(data[spans[i]:spans[i + 1]])
    elif method == 'max':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.nanmax(data[spans[i]:spans[i + 1]])
    elif method == 'sum':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.nansum(data[spans[i]:spans[i + 1]])
    elif method == 'count_nonzero':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.count_nonzero(data[spans[i]:spans[i + 1]])
    elif method == 'first':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = data[spans[i]]
    elif method == 'last':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = data[spans[i+1]-1]
    else:
        print('Unspported method.')
    return result


def expand_single_subject(df_dict, new_idx):
    df = pd.DataFrame(df_dict)
    #df = df.sort_values('interval_days', ascending=False).drop_duplicates('interval_days')
    df_expand = df.set_index('interval_days').reindex(new_idx)
    df_expand['patient_id'] = df_dict['patient_id'][0]
    df_expand['interval_days'] = new_idx

    for f in list_symptoms:
        #df_expand[f] = single_interpolate(df_dict['interval_days'], df_dict[f], new_idx)
        df_expand[f] = df_expand[f].interpolate(method='linear', limit_area='inside')
    #df_expand['created_interp'] = single_interpolate(df_dict['interval_days'], df_dict['created_at'], new_idx)
    df_expand['created_interp'] = df_expand['created_at'].interpolate(method='linear', limit_area='inside')

    df_expand = df_expand.dropna(subset=list_symptoms + ['date_update'], how='all')

    return df_expand




def interpolate_date(df_train, list_symptoms=list_symptoms, col_day='interval_days'):
    sob2 = np.round(df_train['shortness_of_breath'].data[:] * 1.0 / 3, 0)
    df_train.create_numeric('sob2', 'float').data.write(sob2)
    fatigue2 = np.round(df_train['fatigue'].data[:] * 1.0 / 3, 0)
    df_train.create_numeric('fatigue2', 'float').data.write(fatigue2)
    #drop duplicates method 1 - keeping the last
    df_train.sort_values(by=['patient_id', 'interval_days', 'created_at'])
    sorted_by_fields_data = np.asarray([df_train[k].data[:] for k in ['patient_id', 'interval_days']])
    spans = ops._get_spans_for_multi_fields(sorted_by_fields_data)

    global TEMPH5
    df_test_comb = TEMPH5.create_dataframe('df_test_comb')
    df_train.apply_index(spans[:-1], ddf=df_test_comb)
    #drop duplicates method 2 - keeping the first and sum up the symptoms

    # df_test_comb = df_train.sort_values(col_day, ascending=False).drop_duplicates(
    #     [col_day, "patient_id"])
    #df_test_comb = df_test_comb.sort_values(['patient_id', 'interval_days'])

    full_idx = np.arange(df_test_comb['interval_days'].data[:].min(), df_test_comb['interval_days'].data[:].max())

    # for f in list_symptoms + ['fatigue', 'shortness_of_breath']:
    #     df_test_comb[f] = df_test_comb[f].fillna(0)
    #df_test_comb_ind = df_test_comb.set_index('interval_days')

    # df_test2 = df_test_comb_ind.groupby('patient_id', as_index=False).apply(lambda group: group.reindex(full_idx)).reset_index(level=0, drop=True).sort_index()
    expanded_df = TEMPH5.create_dataframe('expanded_df')
    for fld in df_test_comb.keys():
        if fld in list_symptoms:
            expanded_df.create_numeric(fld, 'float')
        else:
            df_test_comb[fld].create_like(expanded_df, fld)
    #expanded_df.create_numeric('interval','float')
    expanded_df.create_timestamp('created_interp')

    spans = df_test_comb['patient_id'].get_spans()
    for i in range(len(spans) - 1):
        if (i % 1000 == 0): print(datetime.now(), i, ' no. of interpolation processed. ', len(spans))
        pddf = dict()
        for fld in df_test_comb.keys():
            pddf[fld] = df_test_comb[fld].data[spans[i]:spans[i + 1]]
        result = expand_single_subject(pddf, full_idx)
        for fld in result.columns:
            expanded_df[fld].data.write(result[fld])

    return expanded_df


def creating_interpolation(df_init):
    print('Applying date range')
    df.copy(df_init['created_at'], df_init, 'date_update')

    interval_days = np.zeros(len(df_init['patient_id']), 'int32')
    spans = df_init['patient_id'].get_spans()
    first_entry_sum = transform_method(df_init['sum_symp'].data[:], spans, 'first')
    last_entry_sum = transform_method(df_init['sum_symp'].data[:], spans, 'last')
    date_update = df_init['date_update'].data[:]
    min_date = transform_method(date_update, spans, 'min')

    for i in range(len(min_date)):
        interval_days[i] = (datetime.fromtimestamp(date_update[i]) - datetime.fromtimestamp(min_date[i])).days

    df_init.create_numeric('first_entry_sum', 'float').data.write(first_entry_sum)
    df_init.create_numeric('last_entry_sum', 'float').data.write(last_entry_sum)
    df_init.create_numeric('interval_days', 'int32').data.write(interval_days)


    df_init = creating_nanvalues_and_aggregate(df_init)

    print('Performing interpolation')
    df_interp = interpolate_date(df_init)

    return df_interp


def define_gh_noimp(dfg, gap_healthy=7):  # still using pandas df
    print(np.unique(dfg['patient_id']), len(dfg.index), len(np.unique(dfg.index)))
    if np.unique(dfg['patient_id']) == b'001991ad5b58bb78f0a22705e3677327':
        print('A')
    id_max = dfg[dfg['sum_symp'] == dfg['max_symp']]['interval_days'].min()
    id_max2 = dfg[dfg['sum_symp'] == dfg['max_symp']]['interval_days'].max()
    id_test = dfg[(dfg['created_at'] <= dfg['date_effective_test'] + 7 * 86400)]['interval_days'].max()
    id_test_ab = dfg[dfg['created_at'] <= dfg['date_effective_test'] - 14 * 86400]['interval_days'].max()
    # Check if id_test
    dfg_before_test = dfg[dfg['interval_days'] <= id_test]
    dfg_before_test_ab = dfg[dfg['interval_days'] <= id_test_ab]
    dfg_after_test = dfg[dfg['interval_days'] >= id_test]
    dfg_before_max = dfg[dfg['interval_days'] <= id_max]
    dfg_after_max = dfg[dfg['interval_days'] >= id_max2]
    max_count_test_nh = dfg_before_test['count_nothealthy'].max()
    max_count_bh = dfg_before_max['count_healthy'].max()
    max_count_ah = dfg_after_max['count_healthy'].max()

    dfg['first_stuh_hg%d' % gap_healthy] = 0
    dfg['latest_first_%d' % gap_healthy] = 0
    dfg['last_stuh_hg%d' % gap_healthy] = 0
    dfg['first_stuh_nhg%d' % gap_healthy] = 0
    dfg['last_stuh_nhg%d' % gap_healthy] = 0
    dfg['date_first_stuh_hg%d' % gap_healthy] = 0
    dfg['date_latest_first_%d' % gap_healthy] = 0

    if dfg_before_test.shape[0] > 0 and dfg['pcr_standard'].max() == 1:
        if max_count_test_nh > 0:
            print('treating test', dfg_before_test.shape[0], max_count_test_nh)

            #min_count_bh = dfg_before_test['count_healthy'].min()
            dfg['first_stuh_hg%d' % gap_healthy] = dfg_before_test[dfg_before_test['count_nothealthy'] == 1][
                'interval_days'].min()
            dfg['latest_first_%d' % gap_healthy] = dfg_before_test[dfg_before_test['count_nothealthy'] == 1][
                'interval_days'].max()
            dfg['date_latest_first_%d' % gap_healthy] = np.where(
                dfg['latest_first_%d' % gap_healthy] == dfg['interval_days'], dfg['created_interp'], np.nan)
            dfg['date_latest_first_%d' % gap_healthy] = dfg['date_latest_first_%d' % gap_healthy].max()
            if dfg['date_latest_first_%d' % gap_healthy].min() < dfg['date_effective_test'].min() - 14 * 86400:
                dfg['first_stuh_hg%d' % gap_healthy] = np.nan
            else:
                print('creating after test', dfg['first_stuh_hg%d' % gap_healthy].max(),
                      dfg['latest_first_%d' % gap_healthy].max())
                dfg_after_first = dfg[dfg['interval_days'] >= dfg['first_stuh_hg%d' % gap_healthy]]
                print(dfg_after_first.shape, 'is shape of after first')
                last_count_ah = dfg_after_first['count_healthy'].tail(1).to_numpy()[0]

                dfg['last_stuh_hg%d' % gap_healthy] = \
                dfg_after_first[dfg_after_first['count_healthy'] == last_count_ah][
                    'interval_days'].max() - last_count_ah

                dfg['first_stuh_nhg%d' % gap_healthy] = dfg_before_test[dfg_before_test['count_nothealthy'] == 1][
                    'interval_days'].min()
                dfg['last_stuh_nhg%d' % gap_healthy] = \
                dfg_after_first[dfg_after_first['count_healthy'] == last_count_ah][
                    'interval_days'].max() - last_count_ah

                dbm_hg = dfg_before_test[(dfg_before_test['count_healthy'] > gap_healthy) &
                                         (dfg_before_test['interval_days'] <= dfg[
                                             'latest_first_%d' % gap_healthy].max())]

                last_healthy_gap = dbm_hg['interval_days'].max()

                print('creating when healthy gaps')
                if dbm_hg.shape[0] > 0:
                    dfg['first_stuh_hg%d' % gap_healthy] = last_healthy_gap + 1
                dfg['date_first_stuh_hg%d' % gap_healthy] = np.where(
                    dfg['first_stuh_hg%d' % gap_healthy] == dfg['interval_days'], dfg['created_interp'], np.nan)
                dfg['date_latest_first_%d' % gap_healthy] = dfg['date_latest_first_%d' % gap_healthy].max()
                if dfg['date_first_stuh_hg%d' % gap_healthy].min() < dfg['date_effective_test'].min() - 14 * 86400:
                    dfg['first_stuh_hg%d' % gap_healthy] = np.nan

            dfg_after_first = dfg[dfg['interval_days'] >= dfg['first_stuh_hg%d' % gap_healthy]]
            print(dfg_after_first.shape, 'for creating healthy gaps')
            if dfg_after_first.shape[0] > 0:
                last_count_ah = dfg_after_first['count_healthy'].tail(1).to_numpy()[0]
                dfg['last_stuh_hg%d' % gap_healthy] = \
                dfg_after_first[dfg_after_first['count_healthy'] == last_count_ah][
                    'interval_days'].max() - last_count_ah
            dam_hg = dfg_after_first[dfg_after_first['count_healthy'] > gap_healthy]

            if dam_hg.shape[0] > 0:
                first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
                dfg['last_stuh_hg%d' % gap_healthy] = first_healthy_gap

            dbm_hg = dfg_before_test[dfg_before_test['count_nans_or_health'] > gap_healthy]

            #print('creating non nans version')
            if dbm_hg.shape[0] > 0:
                last_healthy_gap = dbm_hg['interval_days'].max()
                dfg['first_stuh_nhg%d' % gap_healthy] = last_healthy_gap + 1
            dam_hg = dfg_after_first[dfg_after_first['count_nans_or_health'] > gap_healthy]

            if dam_hg.shape[0] > 0:
                first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
                dfg['last_stuh_nhg%d' % gap_healthy] = first_healthy_gap
    elif dfg_before_test_ab.shape[0] > 0 and dfg['pcr_standard'].max() == 0:
        last_count_ah = dfg['count_healthy'].tail(1).to_numpy()[0]
        #min_count_bh = dfg_before_max['count_healthy'].min()
        dfg['first_stuh_hg%d' % gap_healthy] = dfg_before_test_ab[dfg_before_test_ab['count_nothealthy'] == 1][
            'interval_days'].min()
        dfg['last_stuh_hg%d' % gap_healthy] = dfg[dfg['count_healthy'] == last_count_ah][
                                                  'interval_days'].max() - last_count_ah
        dfg_before_test_ab['first_stuh_hg%d' % gap_healthy] = dfg['first_stuh_hg%d' % gap_healthy]
        possible_firsts = list(
            dfg_before_test_ab[dfg_before_test_ab['count_nothealthy'] == 1]['interval_days'].to_numpy())
        possible_lasts = list((dfg[(dfg['interval_days'] > dfg['first_stuh_hg%d' % gap_healthy])
                                   & (dfg['count_healthy'] == gap_healthy + 1)][
                                   'interval_days'] - gap_healthy).to_numpy())

        possible_lasts.append(dfg['last_stuh_hg%d' % gap_healthy].max())
        possible_firsts.append(dfg['first_stuh_hg%d' % gap_healthy].max())
        #print(list(set(possible_firsts)), list(set(possible_lasts)))

        possible_firsts = np.sort(np.asarray(list(set(possible_firsts))))
        possible_lasts = np.sort(np.asarray(list(set(possible_lasts))))

        checked_first_pos = []
        checked_last_pos = []
        checked_first_pos.append(possible_firsts[0])
        checked_last_pos.append(possible_lasts[0])

        len_pos = []

        for f in possible_firsts:
            if np.min(possible_lasts) < f:
                checked_first_pos.append(f)
        for l in possible_lasts:
            if np.max(checked_first_pos) > l:
                checked_last_pos.append(l)

        for (f, l) in zip(checked_first_pos, checked_last_pos):
            len_pos.append(l + 1 - f)
        ind_max = np.argmax(len_pos)
        dfg['first_stuh_hg%d' % gap_healthy] = checked_first_pos[ind_max]
        dfg['last_stuh_hg%d' % gap_healthy] = checked_last_pos[ind_max]
    else:
        #print('WARNING: No log before test')
        dfg['first_stuh_hg%d' % gap_healthy] = np.nan
        dfg['last_stuh_hg%d' % gap_healthy] = np.nan
    #print(np.max(dfg.index), 'is index max of dfg')

    return dfg


def creating_duration_healthybased(df_interp, saved_name, days=[7], hi=False, force=True):
    if hi == True:
        health_status = np.where(df_interp['sum_symp'].data[:] > 0, 0, np.nan)
        health_status = np.where(df_interp['sum_symp'].data[:] == 0, 1, health_status)
        df_interp.create_numeric('health_status', 'float').data.write(health_status)

        # df_interp['max_symp'] = df_interp.groupby('patient_id')['sum_symp'].transform('max')
        # df_interp = df_interp.groupby('patient_id').apply(lambda group: interpolate_healthy(group))
        health = np.where(df_interp['sum_symp'].data[:]  == 0, 1, np.where(df_interp['sum_symp'].data[:] > 0, 0, np.nan))
        df_interp.create_numeric('health', 'float').data.write(health)

        spans = df_interp['patient_id'].get_spans()
        max_symp = np.zeros(len(df_interp['patient_id'].data), 'float')
        nan_uh = np.zeros(len(df_interp['patient_id'].data), 'float')
        nan_or_health = np.zeros(len(df_interp['patient_id'].data), 'float')
        nan_healthy = np.zeros(len(df_interp['patient_id'].data), 'float')
        health_interp = np.zeros(len(df_interp['patient_id'].data), 'float')
        health_back = np.zeros(len(df_interp['patient_id'].data), 'float')
        count_healthy = np.zeros(len(df_interp['patient_id'].data), 'int32')
        count_nothealthy = np.zeros(len(df_interp['patient_id'].data), 'int32')
        count_nans_or_health = np.zeros(len(df_interp['patient_id'].data), 'int32')
        count_nans = np.zeros(len(df_interp['patient_id'].data), 'int32')

        for i in range(len(spans)-1):
            max_symp[spans[i]:spans[i+1]] = np.nanmax(df_interp['sum_symp'].data[spans[i]:spans[i+1]])
            #interpolate_healthy functions
            health_interp[spans[i]:spans[i+1]] = pd.Series(df_interp['health'].data[spans[i]:spans[i+1]]).ffill()
            health_back[spans[i]:spans[i+1]] = pd.Series(df_interp['health'].data[spans[i]:spans[i+1]]).bfill()
            nan_healthy[spans[i]:spans[i+1]] = np.where(health_interp[spans[i]:spans[i+1]] == 1, np.nan, health_interp[spans[i]:spans[i+1]])
            nan_healthy_pd = pd.Series(nan_healthy[spans[i]:spans[i+1]])
            count_healthy[spans[i]:spans[i+1]] = nan_healthy_pd.isnull().astype(int).groupby( nan_healthy_pd.notnull().astype(int).cumsum()).cumsum()

            nan_uh[spans[i]:spans[i+1]] = np.where(health_interp[spans[i]:spans[i+1]] == 0, np.nan, 1 - health_interp[spans[i]:spans[i+1]])
            nan_uh_pd = pd.Series(nan_uh[spans[i]:spans[i+1]])
            count_nothealthy[spans[i]:spans[i+1]] = nan_uh_pd.isnull().astype(int).groupby( nan_uh_pd.notnull().astype(int).cumsum()).cumsum()

            nan_or_health[spans[i]:spans[i+1]] = np.where(df_interp['health'].data[spans[i]:spans[i+1]] == 0, 1, np.nan)
            nan_or_health_pd = pd.Series(nan_or_health[spans[i]:spans[i+1]])
            count_nans_or_health[spans[i]:spans[i+1]] = nan_or_health_pd.isnull().astype(int).groupby(nan_or_health_pd.notnull().astype(int).cumsum()).cumsum()

            health_status = pd.Series(df_interp['health_status'].data[spans[i]:spans[i+1]])
            count_nans[spans[i]:spans[i+1]] = health_status.isnull().astype(int).groupby(health_status.notnull().astype(int).cumsum()).cumsum()

        df_interp.create_numeric('nan_healthy', 'float').data.write(nan_healthy)
        df_interp.create_numeric('nan_uh', 'float').data.write(nan_uh)
        df_interp.create_numeric('nan_or_health', 'float').data.write(nan_or_health)
        df_interp.create_numeric('max_symp', 'float').data.write(max_symp)
        df_interp.create_numeric('health_interp', 'float').data.write(health_interp)
        df_interp.create_numeric('health_back', 'float').data.write(health_back)
        df_interp.create_numeric('count_healthy', 'int32').data.write(count_healthy)
        df_interp.create_numeric('count_nothealthy', 'int32').data.write(count_nothealthy)
        df_interp.create_numeric('count_nans_or_health', 'int32').data.write(count_nans_or_health)
        df_interp.create_numeric('count_nans', 'int32').data.write(count_nans)

    for f in days:
        if 'first_stuh_hg%d' % f not in df_interp.keys() or force == True:
            print('treating ', f)
            #todo confirm remove dup index
            #df_interp = df_interp.loc[~df_interp.index.duplicated(keep='first')]

            # list_dfg = []
            # list_pat = list(set(df_interp['patient_id']))
            # #list_ind = []
            # for (i, p) in enumerate(list_pat):
            #     dfg = df_interp[df_interp['patient_id'] == p]
            #     dfg_new = define_gh_noimp(dfg, gap_healthy=f)
            #     #list_ind.append(dfg_new.index)
            #     list_dfg.append(dfg_new.loc[~dfg_new.index.duplicated(keep='first')])
            #     print(i, " treated out of ", len(list_pat))
            #
            # df_interp = pd.concat(list_dfg)
            #
            # # df_interp = df_interp.groupby('patient_id').apply(lambda group: define_gh_noimp(group, gap_healthy=f))
            # df_interp.to_csv(saved_name) #  note here csv file is only saved for the last day
            spans = df_interp['patient_id'].get_spans()
            global TEMPH5
            newdf = TEMPH5.create_dataframe('newdf')
            for fld in df_interp.keys():
                df_interp[fld].create_like(newdf, fld)
            for fld in ['first_stuh_hg%d' % f, 'latest_first_%d' % f, 'last_stuh_hg%d' % f,
                      'first_stuh_nhg%d' % f, 'last_stuh_nhg%d' % f]:
                newdf.create_numeric(fld, 'float')
            for fld in ['date_first_stuh_hg%d' % f, 'date_latest_first_%d' % f]:
                newdf.create_timestamp(fld)
            for i in range(
                    len(spans) - 1):  # alter the define_gh_noimp function to exetera is too complicated, try using pandas
                if i % 1000 == 0: print(datetime.now(), i, ' number of define_gh_noimp processed.', len(spans))
                tempd = {}
                for fld in df_interp.keys():
                    tempd[fld] = df_interp[fld].data[spans[i]:spans[i + 1]]
                tempdf = pd.DataFrame(tempd)
                resultdb = define_gh_noimp(tempdf)
                for fld in resultdb.columns:
                    newdf[fld].data.write(resultdb[fld])

            df_interp = newdf
            #df_interp.to_csv(saved_name)

    return df_interp


def determine_meeting_criteria(df_interp, days=[7]):
    from datetime import datetime, timedelta

    #timemin = timedelta(days=-7)
    import time

    #struct_time = datetime.strptime("20 Jul 20", "%d %b %y").date()
    delay = (df_interp['created_at'].data[:] - datetime.strptime("20 Jul 20", "%d %b %y").timestamp())/86400  # delay is in days, todo confirm use created_at
    dropped = np.where(delay <= -7, 1, 0)
    df_interp.create_numeric('delay', 'float').data.write(delay)
    df_interp.create_numeric('dropped', 'int8').data.write(dropped)

    last_entry_sum = np.zeros(len(df_interp['patient_id'].data), 'float')
    spans = df_interp['patient_id'].get_spans()
    for i in range(len(spans)-1):
        last_entry_sum[spans[i]:spans[i+1]] = df_interp['created_at'].data[spans[i+1]-1]
    df_interp['last_entry_sum'].data.clear()
    df_interp['last_entry_sum'].data.write(last_entry_sum)

    # df_interp['count_uh_nans'] = df_interp['count_nans'] * (1 - df_interp['health_interp']) * (
    #         1 - df_interp['health_back'])
    count_uh_nans = df_interp['count_nans'].data[:] * (1 - df_interp['health_interp'].data[:]) * (
            1 - df_interp['health_back'].data[:])
    df_interp.create_numeric('count_uh_nans', 'int16').data.write(count_uh_nans)
    #df_interp['count_utoh_nans'] = df_interp['count_nans'] * (1 - df_interp['health_interp']) * df_interp['health_back']
    count_utoh_nans = df_interp['count_nans'].data[:] * (1 - df_interp['health_interp'].data[:]) * df_interp['health_back'].data[:]
    df_interp.create_numeric('count_utoh_nans', 'int16').data.write(count_utoh_nans)
    # df_interp['count_htouh_nans'] = df_interp['count_nans'] * (df_interp['health_interp']) * (
    #         1 - df_interp['health_back'])
    count_htouh_nans = df_interp['count_nans'].data[:] * (df_interp['health_interp'].data[:]) * (
            1 - df_interp['health_back'].data[:])
    df_interp.create_numeric('count_htouh_nans', 'int16').data.write(count_htouh_nans)
    #df_interp['max_uh_nans'] = df_interp.groupby('patient_id')['count_uh_nans'].transform('max')
    max_uh_nans = np.zeros(len(df_interp['patient_id']), 'int16')
    spans = df_interp['patient_id'].get_spans()
    for i in range(len(spans)-1):
        max_uh_nans[spans[i]:spans[i+1]] = np.max(df_interp['count_uh_nans'].data[spans[i]:spans[i+1]])
    df_interp.create_numeric('max_uh_nans', 'int16').data.write(max_uh_nans)

    for f in days:
        if 'postcrit_aok%d' % f not in df_interp.keys():
            print('Need to treat ', f)
            #timemin = timedelta(days=-f)
            meeting_post_criteria = np.where(
                np.logical_and(df_interp['interval_days'].data[:] > df_interp['last_stuh_hg%d' % f].data[:],
                               df_interp['count_healthy'].data[:] == f), 1, 0)
            df_interp.create_numeric('meeting_post_criteria%d' % f, 'int16').data.write(meeting_post_criteria)

            # df_interp['postcrit_ok%d' % f] = df_interp.groupby('patient_id')['meeting_post_criteria%d' % f].transform(
            #     'max')
            postcrit_ok = np.zeros(len(df_interp['patient_id'].data), 'int16')
            spans = df_interp['patient_id'].get_spans()
            for i in range(len(spans)-1):
                postcrit_ok[spans[i]:spans[i+1]] = np.max(df_interp['meeting_post_criteria%d' % f].data[spans[i]:spans[i+1]])
            df_interp.create_numeric('postcrit_ok%d' % f, 'int16').data.write(postcrit_ok)

            df.copy(df_interp['postcrit_ok%d' % f], df_interp, 'postcrit_aok%d' % f)

            postcrit_aok = np.where( np.logical_and(delay < int(f), df_interp['last_entry_sum'].data[:] == 0), 1,
                df_interp['postcrit_aok%d' % f].data[:])
            df_interp['postcrit_aok%d' % f].data.write(postcrit_aok)
            # df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')

    for f in list_symptoms:
        check_ = np.where(df_interp['health_interp'].data[:] == 0, df_interp[f].data[:], 0)
        df_interp.create_numeric('check_'+f, 'int16').data.write(check_)

        #df_interp['sumcheck_' + f] = df_interp.groupby('patient_id')['check_' + f].transform('sum')
        sumcheck_ = np.zeros(len(df_interp['patient_id'].data), 'float')
        spans = df_interp['patient_id'].get_spans()
        for i in range(len(spans)-1):
            sumcheck_[spans[i]:spans[i+1]] = np.sum(check_[spans[i]:spans[i+1]])
        df_interp.create_numeric('sumcheck_' + f, 'float').data.write(sumcheck_)

    for d in days:
        sickd = np.where(
            np.logical_and(df_interp['interval_days'].data[:] >= df_interp['first_stuh_hg%d' % d].data[:],
                           df_interp['interval_days'].data[:] <= df_interp['last_stuh_hg%d' % d].data[:]), 1, 0)
        df_interp.create_numeric('sick%d' % d, 'int16').data.write(sickd)
        for f in list_symptoms:
            #df_interp['sick%d_' % d + f] = np.where(df_interp['sick%d' % d] == 1, df_interp['check_' + f], 0)
            sickdf = np.where(df_interp['sick%d' % d].data[:] == 1, df_interp['check_' + f].data[:], 0)
            df_interp.create_numeric('sick%d_' % d + f, 'int16').data.write(sickdf)

            # df_interp['day%d_' % d + f] = np.where(
            #     np.logical_and(df_interp['sick%d' % d] == 1, df_interp['check_' + f] > 0.5), df_interp['interval_days'],
            #     np.nan)
            daydf = np.where(np.logical_and(df_interp['sick%d' % d].data[:] == 1, df_interp['check_' + f].data[:] > 0.5), df_interp['interval_days'].data[:], np.nan)
            df_interp.create_numeric('day%d_' % d + f, 'int16').data.write(daydf)

            #df_interp['start%d_' % d + f] = df_interp.groupby('patient_id')['day%d_' % d + f].transform('min')
            #df_interp['sumsick%d_' % d + f] = df_interp.groupby('patient_id')['sick%d_' % d + f].transform('sum')
            spans = df_interp['patient_id'].get_spans()
            startdf = np.zeros(len(df_interp['patient_id'].data), 'float')
            sumsickdf = np.zeros(len(df_interp['patient_id'].data), 'int16')
            for i in range(len(spans) - 1):
                startdf[spans[i]:spans[i + 1]] = np.min(daydf[spans[i]:spans[i + 1]])
                sumsickdf[spans[i]:spans[i + 1]] = np.sum(sickdf[spans[i]:spans[i + 1]])
            df_interp.create_numeric('start%d_' % d + f, 'float').data.write(startdf)
            df_interp.create_numeric('sumsick%d_' % d + f, 'int16').data.write(sumsickdf)


        for f in ['fatigue', 'shortness_of_breath']:
            # df_interp['sick%d_' % d + f + '_mild'] = np.where(df_interp['sick%d_' % d + f] >= 1, 1,
            #                                                   df_interp['sick%d_' % d + f])
            sickmild = np.where(df_interp['sick%d_' % d + f].data[:] >= 1, 1, df_interp['sick%d_' % d + f].data[:])
            df_interp.create_numeric('sick%d_' % d + f + '_mild', 'int16').data.write(sickmild)

            # df_interp['day%d_' % d + f + '_mild'] = np.where( np.logical_and(df_interp['sick%d_' % d + f + '_mild'] > 1,
            #                                                                  df_interp['check_' + f] > 0.5), df_interp['interval_days'], np.nan)
            daymild = np.where( np.logical_and(df_interp['sick%d_' % d + f + '_mild'].data[:] > 1,
                                                                             df_interp['check_' + f].data[:] > 0.5), df_interp['interval_days'].data[:], np.nan)
            df_interp.create_numeric('day%d_' % d + f + '_mild', 'int16').data.write(daymild)
            # df_interp['start%d_' % d + f + '_mild'] = df_interp.groupby('patient_id')[
            #     'day%d_' % d + f + '_mild'].transform(
            #     'min')
            # df_interp['sumsick%d_' % d + f + '_mild'] = df_interp.groupby('patient_id')[
            #     'sick%d_' % d + f + '_mild'].transform('sum')
            # df_interp['end%d_' % d + f + '_mild'] = df_interp.groupby('patient_id')[
            #     'day%d_' % d + f + '_mild'].transform(
            #     'max')

            #df_interp['sick%d_' % d + f + '_severe'] = df_interp['sick%d_' % d + f] / 3
            sicksever = df_interp['sick%d_' % d + f].data[:] / 3
            df_interp.create_numeric('sick%d_' % d + f + '_severe', 'int16').data.write(sicksever)

            # df_interp['day%d_' % d + f + '_severe'] = np.where( np.logical_and(df_interp['sick%d_' % d + f + '_severe'] >= 0.5,
            #                                                                    df_interp['check_' + f] > 1.5), df_interp['interval_days'], np.nan)
            daysever =  np.where( np.logical_and(df_interp['sick%d_' % d + f + '_severe'].data[:] >= 0.5,
                                                 df_interp['check_' + f].data[:] > 1.5), df_interp['interval_days'].data[:], np.nan)
            df_interp.create_numeric('day%d_' % d + f + '_severe', 'int16').data.write(daysever)

            # df_interp['start%d_' % d + f + '_severe'] = df_interp.groupby('patient_id')[
            #     'day%d_' % d + f + '_severe'].transform('min')
            # df_interp['sumsick%d_' % d + f + '_severe'] = df_interp.groupby('patient_id')[
            #     'sick%d_' % d + f + '_severe'].transform('sum')
            # df_interp['end%d_' % d + f + '_severe'] = df_interp.groupby('patient_id')[
            #     'day%d_' % d + f + '_severe'].transform('max')
            startmild = np.zeros(len(df_interp['patient_id']), 'float')
            endmild = np.zeros(len(df_interp['patient_id']), 'float')
            summild = np.zeros(len(df_interp['patient_id']), 'int16')
            startsever = np.zeros(len(df_interp['patient_id']), 'float')
            endsever = np.zeros(len(df_interp['patient_id']), 'float')
            sumsever = np.zeros(len(df_interp['patient_id']), 'int16')
            for i in range(len(spans) - 1):
                startmild[spans[i]:spans[i+1]] = np.min(daymild[spans[i]:spans[i+1]])
                endmild[spans[i]:spans[i+1]] = np.max(daymild[spans[i]:spans[i+1]])
                summild[spans[i]:spans[i+1]] = np.sum(sickmild[spans[i]:spans[i+1]])

                startsever[spans[i]:spans[i+1]] = np.min(daysever[spans[i]:spans[i+1]])
                endsever[spans[i]:spans[i + 1]] = np.max(daysever[spans[i]:spans[i + 1]])
                sumsever[spans[i]:spans[i + 1]] = np.sum(sicksever[spans[i]:spans[i + 1]])

            df_interp.create_numeric('start%d_' % d + f + '_mild', 'float').data.write(startmild)
            df_interp.create_numeric('end%d_' % d + f + '_mild', 'float').data.write(endmild)
            df_interp.create_numeric('sumsick%d_' % d + f + '_mild', 'int16').data.write(summild)

            df_interp.create_numeric('start%d_' % d + f + '_severe', 'float').data.write(startsever)
            df_interp.create_numeric('end%d_' % d + f + '_severe', 'float').data.write(endsever)
            df_interp.create_numeric('sumsick%d_' % d + f + '_severe', 'int16').data.write(sumsever)

    return df_interp


def treat_interp(df_interp):
    print(datetime.now(), ' Final processing...')
    spans = df_interp['patient_id'].get_spans()
    data = df_interp['interval_days'].data[:]
    count_values = transform_method(data, spans, 'count_nonzero')
    length_log = transform_method(data, spans, 'max')
    df_interp.create_numeric('count_values', 'int16').data.write(count_values)
    df_interp.create_numeric('length_log', 'int16').data.write(length_log)

    for d in [7]:  # 14
        duration = df_interp['last_stuh_hg%d'%d].data[:] - df_interp['first_stuh_hg%d'%d].data[:] + 1
        df_interp.create_numeric('duration%d'%d, 'int16').data.write(duration)
        last_status = np.where(df_interp['last_stuh_hg%d'%d].data[:] == df_interp['interval_days'].data[:],df_interp['health_interp'].data[:],np.nan)

        last_status = transform_method(last_status, spans, 'max')
        df_interp.create_numeric('last_status%d'%d, 'float').data.write(last_status)

        duration_new = np.where(df_interp['last_status%d'%d].data[:] == 1, df_interp['duration%d'%d].data[:] - 1, df_interp['duration%d'%d].data[:])
        df_interp.create_numeric('duration%dnew'%d, 'int16').data.write(duration_new)

    # for s in list_symptoms_red:  # todo list_symptoms_red unknown
    #     if 'day7_'+s in df_interp.keys():
    #         spans = df_interp['patient_id'].get_spans()
    #         end7 = np.zeros(len(df_interp['patient_id']), 'int16')
    #         for i in range(len(spans) - 1):
    #             end7[spans[i]:spans[i + 1]] = np.max(df_interp['day7_'+s].data[spans[i]:spans[i + 1]])
    #         df_interp.create_numeric('end7_'+s, 'int16').data.write(end7)

    date_sick = np.where(df_interp['health_interp'].data[:] == 0, df_interp['created_at'].data[:], np.nan)
    df_interp.create_timestamp('date_sick').data.write(date_sick)

    date_sick = df_interp['date_sick'].data[:]
    first_sick = transform_method(date_sick, spans, 'min')
    last_sick = transform_method(date_sick, spans, 'max')
    created_at = df_interp['created_at'].data[:]
    first_log = transform_method(created_at, spans, 'min')
    last_log = transform_method(created_at, spans, 'max')

    df_interp.create_timestamp('first_sick').data.write(first_sick)
    df_interp.create_timestamp('last_sick').data.write(last_sick)
    df_interp.create_timestamp('first_log').data.write(first_log)
    df_interp.create_timestamp('last_log').data.write(last_log)

    h_extent = np.where(df_interp['created_at'].data[:] >= df_interp['first_sick'].data[:], df_interp['health_interp'].data[:], 0)
    h_extent = np.where(df_interp['created_at'].data[:] <= df_interp['last_sick'].data[:], h_extent, 0)
    df_interp.create_numeric('h_extent', 'int16').data.write(h_extent)
    count_h_extent = transform_method(h_extent, spans, 'sum')
    df_interp.create_numeric('count_h_extent', 'int16').data.write(count_h_extent)

    #df_interp['prop_h'] = df_interp['count_healthy'].replace(0, np.nan)
    prop_h = np.where(df_interp['count_healthy'].data[:]==np.nan, 0, df_interp['count_healthy'].data[:])
    df_interp.create_numeric('prop_h', 'int16').data.write(prop_h)
    #df_interp['prop_h'] = df_interp.groupby('patient_id')['prop_h'].ffill()  # shouldn't be useful as no nan left

    first_s = np.where(df_interp['count_nothealthy'].data[:] == 1, df_interp['created_at'].data[:], np.nan)
    df_interp.create_timestamp('first_s').data.write(first_s)

    first_sicki = np.where(df_interp['created_at'].data[:] == df_interp['first_sick'].data[:], df_interp['interval_days'].data[:], np.nan)
    last_sicki = np.where(df_interp['created_at'].data[:] == df_interp['last_sick'].data[:], df_interp['interval_days'].data[:], np.nan)
    first_sicki = transform_method(first_sicki, spans, 'min')
    last_sicki = transform_method(last_sicki, spans, 'min')
    max_nans = transform_method(df_interp['count_nans'].data[:], spans, 'max')
    df_interp.create_numeric('first_sicki', 'int16').data.write(first_sicki)
    df_interp.create_numeric('last_sicki', 'int16').data.write(last_sicki)
    df_interp.create_numeric('max_nans', 'int16').data.write(max_nans)

    print('creating nan values')
    for d in [7]: # 14
        date_firstuh7 = np.where(df_interp['interval_days'].data[:] == df_interp['first_stuh_hg7'].data[:], df_interp['created_at'].data[:], np.nan)
        # date_firstuh14 = np.where(df_interp['interval_days'].data[:] == df_interp['first_stuh_hg14'].data[:],
        #                                        df_interp['created_at'].data[:], np.nan)
        date_lastuh7 = np.where(df_interp['interval_days'].data[:] >= df_interp['last_stuh_hg7'].data[:],
                                             df_interp['created_at'].data[:], np.nan)
        # date_lastuh14 = np.where(df_interp['interval_days'].data[:] >= df_interp['last_stuh_hg14'].data[:],
        #                                       df_interp['created_at'].data[:], np.nan)
        spans = df_interp['patient_id'].get_spans()
        for i in range(len(spans)-1):
            date_firstuh7[spans[i]:spans[i+1]] = np.min(date_firstuh7[spans[i]:spans[i+1]])
            #date_firstuh14[spans[i]:spans[i + 1]] = np.min(date_firstuh14[spans[i]:spans[i + 1]])
            date_lastuh7[spans[i]:spans[i + 1]] = np.min(date_lastuh7[spans[i]:spans[i + 1]])
            #date_lastuh14[spans[i]:spans[i + 1]] = np.min(date_lastuh14[spans[i]:spans[i + 1]])
        date_lastuh7 = np.where(df_interp['last_status7'].data[:] == 1, date_lastuh7 - 86400, date_lastuh7)
        #date_lastuh14 = np.where(df_interp['last_status14'].data[:] == 1, date_lastuh14 - 86400, date_lastuh14)
        df_interp.create_timestamp('date_firstuh7').data.write(date_firstuh7)
        #df_interp.create_timestamp('date_firstuh14').data.write(date_firstuh14)
        df_interp.create_timestamp('date_lastuh7').data.write(date_lastuh7)
        #df_interp.create_timestamp('date_lastuh14').data.write(date_lastuh14)

        last_afs7 = np.where(
            np.logical_and(df_interp['created_at'].data[:] > df_interp['first_sick'].data[:], df_interp['count_healthy'].data[:] >= 7),
            df_interp['created_at'].data[:] - 86400 * df_interp['count_healthy'].data[:], np.nan)
        last_afs7 = np.where(df_interp['created_at'].data[:] >= df_interp['last_sick'].data[:], df_interp['last_sick'].data[:], last_afs7)
        df_interp.create_timestamp('last_afs7').data.write(last_afs7)

        last_afs14 = np.where(
            np.logical_and(df_interp['created_at'].data[:] > df_interp['first_sick'].data[:], df_interp['count_healthy'].data[:] >= 14),
            df_interp['created_at'].data[:] - 86400 * df_interp['count_healthy'].data[:], np.nan)
        last_afs14 = np.where(df_interp['created_at'].data[:] >= df_interp['last_sick'].data[:], df_interp['last_sick'].data[:], last_afs14)
        df_interp.create_timestamp('last_afs14').data.write(last_afs14)

        first_s7 = np.where(
            np.logical_and(df_interp['first_s'].data[:] > df_interp['first_sick'].data[:], df_interp['prop_h'].data[:] < 7),
            np.nan, df_interp['first_s'].data[:])
        df_interp.create_timestamp('first_s7').data.write(first_s7)
        first_s14 = np.where(
            np.logical_and(df_interp['first_s'].data[:] > df_interp['first_sick'].data[:], df_interp['prop_h'].data[:] < 14), np.nan,
            df_interp['first_s'].data[:])
        df_interp.create_timestamp('first_s14').data.write(first_s14)

        spans = df_interp['patient_id'].get_spans()
        last_afs7_f = np.zeros(len(df_interp['last_afs7']), float)
        last_afs14_f = np.zeros(len(df_interp['last_afs14']), float)
        first_s7_f = np.zeros(len(df_interp['first_s7']), float)
        first_s14_f = np.zeros(len(df_interp['first_s14']), float)
        for i in range(len(spans)-1):
            last_afs7_f[spans[i]:spans[i+1]] = pd.Series(df_interp['last_afs7'].data[spans[i]:spans[i+1]]).bfill()
            last_afs14_f[spans[i]:spans[i+1]] = pd.Series(df_interp['last_afs14'].data[spans[i]:spans[i+1]]).bfill()
            first_s7_f[spans[i]:spans[i+1]] = pd.Series(df_interp['first_s7'].data[spans[i]:spans[i+1]]).ffill()
            first_s14_f[spans[i]:spans[i+1]] = pd.Series(df_interp['first_s14'].data[spans[i]:spans[i+1]]).ffill()

        df_interp.create_numeric('last_afs7_f', 'float').data.write(last_afs7_f)
        df_interp.create_numeric('last_afs14_f','float').data.write(last_afs14_f)
        df_interp.create_numeric('first_s7_f', 'float').data.write(first_s7_f)
        df_interp.create_numeric('first_s14_f', 'float').data.write(first_s14_f)

        dur7_art = (df_interp['last_afs7_f'].data[:] - df_interp['first_s7_f'].data[:]) / 86400 + 1
        dur14_art = (df_interp['last_afs14_f'].data[:] - df_interp['first_s14_f'].data[:]) / 86400 + 1
        dur7_artm = np.zeros(len(dur7_art), float)
        dur14_artm = np.zeros(len(dur14_art), float)
        for i in range(len(spans) - 1):
            dur7_artm[spans[i]:spans[i+1]] = np.max(dur7_art[spans[i]:spans[i+1]])
            dur14_artm[spans[i]:spans[i + 1]] = np.max(dur14_art[spans[i]:spans[i + 1]])
        df_interp.create_timestamp('dur7_art').data.write(dur7_art)
        df_interp.create_timestamp('dur14_art').data.write(dur14_art)
        df_interp.create_timestamp('dur7_artm').data.write(dur7_artm)
        df_interp.create_timestamp('dur14_artm').data.write(dur14_artm)

        max_nans_fl7 = np.where(np.logical_and(df_interp['interval_days'].data[:] >= df_interp['first_stuh_hg7'].data[:],
                                                            df_interp['interval_days'].data[:] <= df_interp['last_stuh_hg7'].data[:]),
                                             df_interp['count_nans'].data[:], 0)
        # max_nans_fl14 = np.where(np.logical_and(df_interp['interval_days'].data[:] >= df_interp['first_stuh_hg14'].data[:],
        #                                                      df_interp['interval_days'].data[:] <= df_interp['last_stuh_hg14'].data[:]),
        #                                       df_interp['count_nans'].data[:], 0)
        for i in range(len(spans) - 1):
            max_nans_fl7[spans[i]:spans[i+1]] = np.max(max_nans_fl7[spans[i]:spans[i+1]])
            #max_nans_fl14[spans[i]:spans[i + 1]] = np.max(max_nans_fl14[spans[i]:spans[i + 1]])
        df_interp.create_numeric('max_nans_fl7', 'int16').data.write(max_nans_fl7)
        #df_interp.create_numeric('max_nans_fl14', 'int16').data.write(max_nans_fl14)

        #end 7, 14 loop


    day_test = np.where(np.logical_and(df_interp['created_interp'].data[:] <= df_interp['date_effective_test'].data[:] + 86400,
                                               df_interp['created_interp'].data[:] <= df_interp['date_effective_test'].data[:] - 86400),
                        df_interp['interval_days'].data[:], np.nan)
    before_test = np.where(df_interp['created_interp'].data[:] <= df_interp['date_effective_test'].data[:], 1, 0)
    count_uhbef = df_interp['count_nothealthy'].data[:] * before_test
    count_nanbef = df_interp['count_nans'].data[:] * before_test
    created_aroundtest = np.where(np.logical_and(df_interp['created_at'].data[:] <= df_interp['date_effective_test'].data[:]+8*86400,
                                                         df_interp['created_at'].data[:] >= df_interp['date_effective_test'].data[:]-14*86400), 1, 0)
    sum_symp_at = df_interp['sum_symp'].data[:] * created_aroundtest
    df_interp.create_numeric('day_test', 'int16').data.write(day_test)
    df_interp.create_numeric('before_test', 'int16').data.write(before_test)
    df_interp.create_numeric('count_uhbef', 'int16').data.write(count_uhbef)
    df_interp.create_numeric('count_nanbef', 'int16').data.write(count_nanbef)
    df_interp.create_numeric('created_aroundtest', 'int16').data.write(created_aroundtest)
    df_interp.create_numeric('sum_symp_at', 'int16').data.write(sum_symp_at)


    #max_symp = np.zeros(len(df_interp['sum_symp'].data), 'float')
    max_countuh = np.zeros(len(df_interp['count_nothealthy'].data), 'int16')
    max_countnanbef = np.zeros(len(df_interp['count_nanbef'].data), 'int16')
    max_countuhbef = np.zeros(len(df_interp['count_uhbef'].data), 'int16')
    max_symp_at = np.zeros(len(df_interp['sum_symp_at'].data), 'float')
    numb_at = np.zeros(len(df_interp['created_aroundtest'].data), 'int16')

    for i in range(len(spans) - 1):
        #max_symp[spans[i]:spans[i+1]] = np.max(df_interp['sum_symp'].data[spans[i]:spans[i+1]])
        max_countuh[spans[i]:spans[i+1]] = np.max(df_interp['count_nothealthy'].data[spans[i]:spans[i+1]])
        max_countnanbef[spans[i]:spans[i+1]] = np.max(count_nanbef[spans[i]:spans[i+1]])
        max_countuhbef[spans[i]:spans[i+1]] = np.max(count_uhbef[spans[i]:spans[i+1]])
        max_symp_at[spans[i]:spans[i+1]] = np.max(sum_symp_at[spans[i]:spans[i+1]])
        numb_at[spans[i]:spans[i+1]] = np.sum(created_aroundtest[spans[i]:spans[i+1]])

    #df_interp['max_symp'].data.write(max_symp)
    df_interp.create_numeric('max_countuh', 'int16').data.write(max_countuh)
    df_interp.create_numeric('max_countnanbef', 'int16').data.write(max_countnanbef)
    df_interp.create_numeric('max_countuhbef', 'int16').data.write(max_countuhbef)
    df_interp.create_numeric('max_symp_at', 'float').data.write(max_symp_at)
    df_interp.create_numeric('numb_at', 'int16').data.write(numb_at)

    return df_interp


def treat_interp_short(df_interp):
    print(datetime.now(), ' Treat interpolation short and filtering...')
    d=7
    duration = df_interp['last_stuh_hg%d' % d].data[:] - df_interp['first_stuh_hg%d' % d].data[:] + 1
    df_interp.create_numeric('duration%d' % d, 'float').data.write(duration)
    last_status = np.where(df_interp['last_stuh_hg%d' % d].data[:] == df_interp['interval_days'].data[:],
                                              df_interp['health_interp'].data[:], np.nan)
    spans = df_interp['patient_id'].get_spans()
    last_status = transform_method(last_status, spans, 'max')
    df_interp.create_numeric('last_status%d' % d, 'float').data.write(last_status)

    duration_new = np.where(df_interp['last_status%d' % d].data[:] == 1, df_interp['duration%d' % d].data[:] - 1,
                                              df_interp['duration%d' % d].data[:])
    df_interp.create_numeric('duration%dnew' % d, 'float').data.write(duration_new)

    max_nans_fl7 = np.where(np.logical_and(df_interp['interval_days'].data[:] >= df_interp['first_stuh_hg7'].data[:],
                                                        df_interp['interval_days'].data[:] <= df_interp['last_stuh_hg7'].data[:]),
                                         df_interp['count_nans'].data[:], 0)
    max_nans_fl7 = transform_method(max_nans_fl7, spans, 'max')
    df_interp.create_numeric('max_nans_fl7', 'int32').data.write(max_nans_fl7)

    date_firstuh7 = np.where(df_interp['interval_days'].data[:] == df_interp['first_stuh_hg7'].data[:],
                                          df_interp['created_at'].data[:], np.nan)
    date_firstuh7 = transform_method(date_firstuh7, spans, 'min')
    df_interp.create_numeric('date_firstuh7','float').data.write(date_firstuh7)

    return df_interp


def cal_severity(df_proc):
    """
    Calculate the severity of patient after 7days/28days of test. The severity is the median of sym_sump values.
    :param df_proc:
    :return:
    """
    df_proc.sort_values(by=['patient_id', 'date_effective_test'])
    sorted_by_fields_data = np.asarray([df_proc[k].data[:] for k in ['patient_id', 'date_effective_test']])
    spans = ops._get_spans_for_multi_fields(sorted_by_fields_data)
    severity7 = np.zeros(len(df_proc['patient_id']), 'int16')
    severity28 = np.zeros(len(df_proc['patient_id']), 'int16')
    for i in range(len(spans)-1):
        #get asmts 7days after
        filter = np.logical_and(df_proc['date_effective_test'].data[spans[i]:spans[i+1]]<= df_proc['created_at'].data[spans[i]:spans[i+1]] ,
                 df_proc['created_at'].data[spans[i]:spans[i+1]] <= (df_proc['date_effective_test'].data[spans[i]:spans[i+1]] + 86400*7))
        filter &= ~np.isnan(df_proc['sum_symp'].data[spans[i]:spans[i+1]])
        value7 = np.median(df_proc['sum_symp'].data[spans[i]:spans[i+1]][filter])
        severity7[spans[i]:spans[i+1]] = value7 if value7>0 else 0

        filter = np.logical_and(df_proc['date_effective_test'].data[spans[i]:spans[i + 1]] <= df_proc['created_at'].data[spans[i]:spans[i + 1]]
                 , df_proc['created_at'].data[spans[i]:spans[i + 1]] <= (df_proc['date_effective_test'].data[ spans[i]:spans[i + 1]] + 86400 * 28))
        filter &= ~np.isnan(df_proc['sum_symp'].data[spans[i]:spans[i + 1]])
        value28 = np.median(df_proc['sum_symp'].data[spans[i]:spans[i + 1]][filter])
        severity28[spans[i]:spans[i + 1]] = value28 if value28>0 else 0

    df_proc.create_numeric('severity7', 'int16').data.write(severity7)
    df_proc.create_numeric('severity28', 'int16').data.write(severity28)
    return df_proc


def output_three_files(df_proc):
    '''
    1. One file with patients grouped and with duration and max number of symptoms;
    2. One file with the current results - all patients, with all reported days;
    3. Only patients that pass the criteria with missing data less than 7 days - check column s as
    df_interp['valid_timing'] = np.where(np.logical_and(df_interp['diff_test']>=-14, df_interp['diff_test']<7),1,0)

    '''
    # Additional filtering

    valid_dur = np.where(np.isnan(df_proc['duration7new'].data[:]), 0, 1)
    # no gap of more than 7 days during disease course
    valid_dur = np.where(df_proc['max_nans_fl7'].data[:] > 7, 0, valid_dur)
    # no gap of more than 7 days during disease course
    diff_test = (df_proc['date_firstuh7'].data[:] - df_proc['date_effective_test'].data[:]) / 86400
    valid_timing = np.where(np.logical_and(diff_test >= -14, diff_test < 7), 1, 0)

    df_proc.create_numeric('valid_dur', 'int32').data.write(valid_dur)
    df_proc.create_numeric('valid_timing', 'int32').data.write(valid_timing)

    filter = (valid_dur == 1) & (valid_timing == 1)
    df_proc.to_csv('post_proc.csv', row_filter=filter)

    # Apr 21
    filter2 = filter
    filter2 &= df_proc['date_effective_test'].data[:] > datetime(2021, 4, 21).timestamp()
    df_proc.to_csv('post_april.csv', row_filter=filter2)

    # > 28 days
    filter3 = filter
    filter3 &= df_proc['duration7'].data[:] >= 28
    df_proc.to_csv('post_long.csv', row_filter=filter3)

    return df_proc


def main(argv):
    parser = argparse.ArgumentParser(description='Create interpolation for duration')
    parser.add_argument('-i', dest='input', metavar='input pattern',
                        type=str, default='Data/20201117/PositiveSympStartHealthyAllSymptoms.csv',
                        help='RegExp pattern for the input files')
    parser.add_argument('-a', nargs='+', dest='process', action='store', type=str,
                        default='duration', choices=['nans', 'date',  # todo confirm date, health_int
                                                     'interpolate', 'health_int',
                                                     'duration', 'criteria',
                                                     'hosp',
                                                     'all'])
    parser.add_argument('-t', dest='test_file', metavar='input pattern',
                        type=str, default='20201117/TestedPositiveTestDetails.csv',
                        help='RegExp pattern for the input files')
    parser.add_argument('-o', dest='output', action='store',
                        default='', type=str,
                        help='output path')
    parser.add_argument('-hi', dest='hi', action='store_true', default=True)
    parser.add_argument('-d', dest='days', type=int, nargs='+', action='store',
                        help='days_interval', default=[7])

    try:
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compute_ROI_statistics.py -i <input_image_pattern> -m '
              '<mask_image_pattern> -t <threshold> -mul <analysis_type> '
              '-trans <offset>   ')
        sys.exit(2)

    s = Session()
    src = s.open_dataset(args.input, 'r', 'src')
    global TEMPH5
    TEMPH5 = s.open_dataset('temp.h5','w','temp')
    ds.copy(src['out_pos_hs'], TEMPH5, 'out_pos_hs')
    df_proc = TEMPH5['out_pos_hs']
    df_test = src['out_pos_copy']
    # df_proc = pd.read_csv(args.input)
    # df_proc = df_proc.reset_index(drop=True)

    if 'nans' in args.process:
        print('preparing data')
        #df_proc = creating_nanvalues(df_proc)
    if 'interpolate' in args.process:
        print('performing interpolation')
        df_proc = creating_interpolation(df_proc)
        #df_proc.to_csv('interpolation.csv')
        save_df_to_csv(df_proc, 'interpolation.csv' ,['patient_id','interval_days']+list_symptoms+['created_at','sum_symp','country_code',
                                                                              'location','treatment','updated_at','first_entry_sum','last_entry_sum','date_update','sob2','fatigue2','created_interp'] )
    if 'duration' in args.process:
        print('processing duration')
        if 'date_effective_test' not in df_proc.keys():
            # no need to drop duplicates
            spans = df_test['patient_id'].get_spans()
            for i in range(len(spans)-1):
                spans[i] += np.argmin(df_test['date_effective_test'].data[spans[i]:spans[i+1]])
            test_no_dup = TEMPH5.create_dataframe('test_no_dup')
            df_test.apply_index(spans[:-1], ddf=test_no_dup)
            #test_no_dup.to_csv('tests.csv')
            # merge and perform
            tempdf = TEMPH5.create_dataframe('df_proc')
            df.merge(df_proc, test_no_dup, dest=tempdf, left_on='patient_id', right_on='patient_id', how='left' )
            df_proc = tempdf
            del df_proc['patient_id_l']
            df.move(df_proc['patient_id_r'], df_proc, 'patient_id')
        df_proc = creating_duration_healthybased(df_proc, args.output, days=args.days, hi=args.hi)
        save_df_to_csv(df_proc, 'duration.csv', ['patient_id', 'interval_days'] + list_symptoms + ['created_at','sum_symp', 'country_code','location','treatment','updated_at','first_entry_sum','last_entry_sum','date_update','sob2','fatigue2','created_interp', 'date_effective_test','result','pcr_standard','converted_test','health_status','max_symp','health','health_interp','health_back','nan_healthy','nan_uh','count_healthy','count_nans','count_nothealthy','nan_or_health','count_nans_or_health','first_stuh_hg7','latest_first_7','last_stuh_hg7','first_stuh_nhg7','last_stuh_nhg7','date_first_stuh_hg7','date_latest_first_7'])

    if 'criteria' in args.process:
        print('Checking criteria')
        df_proc = determine_meeting_criteria(df_proc, days=args.days)
        #df_proc.to_csv('criteria.csv')
        save_df_to_csv(df_proc, 'criteria.csv', criteria_output)

    # todo group duration by each patient @Liane/Michela
    # todo output data filtering some column @Liane/Michela
    #calculate severity
    df_proc = cal_severity(df_proc)


    #calculate duration
    df_proc = treat_interp_short(df_proc)  # from notebook
    #df_proc = treat_interp(df_proc)
    output_three_files(df_proc)

    if 'hosp' in args.process:  # not updated
        print('Checking hosp')
        # df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')
        # print(df_interp.shape)

        for d in args.days:
            hosp_valid = (df_proc['interval_days'].data[:] > df_proc['first_stuh_hg%d' % d].data[:]) * ( df_proc['location'].data[:] > 1)
            df_proc.create_numeric('hosp_valid%d' % d, 'int16').data.write(hosp_valid)

            hosp_check = np.zeros(len(df_proc['patient_id'].data), 'int16')
            spans = df_proc['patient_id'].get_spans()
            for i in range(len(spans)-1):
                hosp_check[spans[i]:spans[i+1]] = np.max(hosp_valid[spans[i]:spans[i+1]])
            df_proc.create_numeric('hosp_check%d' % d, 'int16').data.write(hosp_check)

            to_adjust = np.where(
                (df_proc['count_utoh_nans'].data[:] == d) * (df_proc['interval_days'].data[:] > df_proc['first_stuh_hg%d' % d].data[:])
                * (df_proc['interval_days'].data[:] < df_proc['last_stuh_hg%d' % d].data[:]) * (df_proc['hosp_check%d' % d].data[:] == 0) == 1,
                1, 0)
            df_proc.create_numeric('to_adjust%d' % d, 'int16').data.write(to_adjust)

            max_adjust = np.zeros(len(df_proc['patient_id'].data), 'int16')
            for i in range(len(spans) - 1):
                max_adjust[spans[i]:spans[i+1]] = np.max(to_adjust[spans[i]:spans[i+1]])
            df_proc.create_numeric('max_adjust%d' % d, 'int16').data.write(max_adjust)

            day_adjust = np.where(df_proc['to_adjust%d' % d] == 1, df_proc['interval_days'], np.nan)
            df_proc.create_numeric('day_adjust%d' % d, 'int16').data.write(day_adjust)

            maxday_adjust = np.zeros(len(df_proc['patient_id'].data), 'int16')
            for i in range(len(spans) - 1):
                maxday_adjust[spans[i]:spans[i+1]] = np.max(day_adjust[spans[i]:spans[i+1]])
            df_proc.create_numeric('maxday_adjust%d' % d, 'int16').data.write(maxday_adjust)

            last_stuh_hg = np.where(max_adjust == 1, maxday_adjust - d, df_proc['last_stuh_hg%d' % d].data[:])
            df_proc.create_numeric('last_stuh_hg%d_adj' % d, 'int16').data.write(last_stuh_hg)

        df_proc.to_csv('hospitalisation_csv')
    # else:  # not updated
    #     df_proc = creating_nanvalues(df_proc)
    #
    #     print('performing interpolation')
    #     df_proc = creating_interpolation(df_proc)
    #     print('processing duration')
    #     if 'date_effective_test' not in df_proc.columns:
    #         df_test = pd.read_csv(args.test_file)
    #         df_test_min = df_test.drop_duplicates('patient_id')
    #         df_proc = pd.merge(df_proc, df_test_min, on='patient_id', how='left')
    #     df_proc = creating_duration_healthybased(df_proc, args.output, days=args.days, hi=args.hi)
    #     df_proc = determine_meeting_criteria(df_proc, days=args.days)
    #     print('Checking hosp')
    #     # df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')
    #     # print(df_interp.shape)
    #
    #     for d in args.days:
    #         df_proc['hosp_valid%d' % d] = (df_proc['interval_days'] > df_proc['first_stuh_hg%d' % d]) * (
    #                 df_proc['location'] > 1)
    #         df_proc['hosp_check%d' % d] = df_proc.groupby('patient_id')['hosp_valid%d' % d].transform('max')
    #         df_proc['to_adjust%d' % d] = np.where(
    #             (df_proc['count_utoh_nans'] == d) * (df_proc['interval_days'] > df_proc['first_stuh_hg%d' % d])
    #             * (df_proc['interval_days'] < df_proc['last_stuh_hg%d' % d]) * (df_proc['hosp_check%d' % d] == 0) == 1,
    #             1, 0)
    #         df_proc['max_adjust%d' % d] = df_proc.groupby('patient_id')['to_adjust%d' % d].transform('max')
    #         df_proc['max_adjust%d' % d] = df_proc.groupby('patient_id')['to_adjust%d' % d].transform('max')
    #         df_proc['day_adjust%d' % d] = np.where(df_proc['to_adjust%d' % d] == 1, df_proc['interval_days'], np.nan)
    #         df_proc['maxday_adjust%d' % d] = df_proc.groupby('patient_id')['day_adjust%d' % d].transform('max')
    #         df_proc['last_stuh_hg%d_adj' % d] = np.where(df_proc['max_adjust%d' % d] == 1,
    #                                                      df_proc['maxday_adjust%d' % d] - d,
    #                                                      df_proc['last_stuh_hg%d' % d])
    #     df_proc.to_csv(args.output)



if __name__ == '__main__':
    main(sys.argv[1:])
