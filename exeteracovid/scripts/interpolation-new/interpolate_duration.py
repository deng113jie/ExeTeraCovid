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


def creating_nanvalues(df_train):
    sum_symp = np.zeros(len(df_train['patient_id'].data),'int16')
    for f in list_symptoms:
        data = df_train[f].data[:]
        data -= -1
        data = np.where(data == -1, 0, data)
        df_train[f].data.clear()
        df_train[f].data.write(data)
        sum_symp += data
    df_train['sum_symp'].data.clear()
    df_train['sum_symp'].data.write(sum_symp)
    return df_train

@njit
def single_interpolate(old_index, old_data, new_index, limit_area='inner'):
    # expand the index
    result = np.full(len(new_index), np.nan, 'float')
    for i in range(len(old_index)):
        idx = np.argwhere(new_index==old_index[i])
        result[idx[0][0]] = old_data[i]
    # interpolate
    if limit_area=='inner':
        for i in range(0, len(old_index)-1):
            if old_index[i+1] - old_index[i] > 1:  # has nan in between
                coeff = float(old_data[i+1] - old_data[i])/float(old_index[i+1] - old_index[i])
                for j in range(old_index[i]+1, old_index[i+1]):
                    idx = np.argwhere(new_index == j)[0][0]
                    result[idx] = old_data[i] + (j-old_index[i])*coeff
    return result


@njit
def transform_method(data, spans, method):
    result = np.zeros(len(data), data.dtype)
    if method == 'min':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.min(data[spans[i]:spans[i + 1]])
    elif method == 'max':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.max(data[spans[i]:spans[i + 1]])
    elif method == 'sum':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.sum(data[spans[i]:spans[i + 1]])
    elif method == 'count_nonzero':
        for i in range(len(spans) - 1):
            result[spans[i]:spans[i + 1]] = np.count_nonzero(data[spans[i]:spans[i + 1]])
    else:
        print('Unspported method.')
    return result


def expand_single_subject(df_dict, new_idx):
    df = pd.DataFrame(df_dict, index=df_dict['interval_days'])
    df['interval'] = df['date_update'] - df['date_update'].min()
    pid = df.patient_id[0]
    df_expand = df.reindex(new_idx)
    df_expand.patient_id = pid
    df_expand['interval_days'] = new_idx

    for f in list_symptoms:
        df_expand[f] = single_interpolate(df_dict['interval_days'], df_dict[f], new_idx)
    df_expand['created_interp'] = single_interpolate(df_dict['interval_days'], df_dict['created_at'], new_idx)
    df_expand = df_expand.dropna(subset=list_symptoms + ['date_update'], how='all')

    return df_expand




def interpolate_date(df_train, list_symptoms=list_symptoms, col_day='interval_days'):
    sob2 = np.round(df_train['shortness_of_breath'].data[:] * 1.0 / 3, 0)
    df_train.create_numeric('sob2', 'float').data.write(sob2)
    fatigue2 = np.round(df_train['fatigue'].data[:] * 1.0 / 3, 0)
    df_train.create_numeric('fatigue2', 'float').data.write(fatigue2)
    #drop duplicates method 1 - keeping the last
    df_train.sort_values(by=['patient_id', col_day])
    sorted_by_fields_data = np.asarray([df_train[k].data[:] for k in ['patient_id', col_day]])
    spans = ops._get_spans_for_multi_fields(sorted_by_fields_data)
    filter = [spans[i+1]-1 for i in range(len(spans)-1)]
    global TEMPH5
    df_test_comb = TEMPH5.create_dataframe('df_test_comb')
    df_train.apply_index(np.array(filter),ddf=df_test_comb)
    #drop duplicates method 2 - keeping the first and sum up the symptoms

    # df_test_comb = df_train.sort_values(col_day, ascending=False).drop_duplicates(
    #     [col_day, "patient_id"])
    #df_test_comb = df_test_comb.sort_values(['patient_id', 'interval_days'])

    full_idx = np.arange(df_test_comb['interval_days'].data[:].min(), df_test_comb['interval_days'].data[:].max()+1)

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
    expanded_df.create_numeric('interval','float')
    expanded_df.create_timestamp('created_interp')

    spans = df_test_comb['patient_id'].get_spans()
    with ThreadPoolExecutor(max_workers=1) as executor:
        thread_list = []
        for i in range(len(spans) - 1):
            if (i % 1000 == 0): print(datetime.now(), i, ' no. of interpolation processed.')
            if (i > 10): break
            pddf = dict()
            for fld in df_test_comb.keys():
                pddf[fld] = df_test_comb[fld].data[spans[i]:spans[i+1]]
            t = executor.submit(expand_single_subject, pddf, full_idx)
            thread_list.append(t)
    for t in as_completed(thread_list):
        result = t.result()
        for fld in result.columns:
            expanded_df[fld].data.write(result[fld])


    # for i in range(len(spans)-1):
    #     #write patient_id, idx
    #     pid = df_test_comb['patient_id'].data[spans[i]]
    #     expanded_df['patient_id'].data.write([pid for _ in range(len(full_idx))])
    #     expanded_df['interval_days'].data.write(full_idx)
    #     idx = df_test_comb['interval_days'].data[spans[i]:spans[i+1]]
    #     created_at_old = df_test_comb['created_at'].data[spans[i]:spans[i+1]]
    #     expanded_df['created_at'].data.write(pd.Series(created_at_old, index=idx).reindex(full_idx))
    #     # todo perform filter on gap of assessments
    #     # todo use np (memory array) to reduce io and speed up
    #     for f in list_symptoms:
    #         data = df_test_comb[f].data[spans[i]:spans[i+1]]
    #         # pdseries = pd.Series(data, index=idx)
    #         # pdseries = pdseries.reindex(full_idx)
    #         # pdseries = pdseries.interpolate(method='linear', limit_area='inside')
    #         # expanded_df[f].data.write(pdseries.values)
    #         result = single_interpolate(idx, data, full_idx)
    #         expanded_df[f].data.write(result)
    #     created_interp = single_interpolate(idx, df_test_comb['created_at'].data[spans[i]:spans[i+1]], full_idx)
    #     expanded_df['created_interp'].data.write(created_interp)
    # #drop na, rename axis, drop column, reset_index
    # filter = np.isnan(expanded_df['fatigue'].data[:])
    # for f in list_symptoms:
    #     filter &= np.isnan(expanded_df[f].data[:])
    # expanded_df.apply_filter(~filter)
    return expanded_df


def creating_interpolation(df_init):
    print('Applying date range')
    # df_init['first_entry_sum'] = df_init.groupby('patient_id')['sum_symp'].transform('first')
    # df_init['last_entry_sum'] = df_init.groupby('patient_id')['sum_symp'].transform('last')
    df.copy( df_init['created_at'], df_init, 'date_update')
    first_entry_sum = np.zeros(len(df_init['patient_id']), 'int16')
    last_entry_sum = np.zeros(len(df_init['patient_id']), 'int16')
    interval_days = np.zeros(len(df_init['patient_id']), 'int16')
    spans = df_init['patient_id'].get_spans()
    for i in range(len(spans)-1):
        first_entry_sum[spans[i]:spans[i+1]] = df_init['sum_symp'].data[spans[i]]
        last_entry_sum[spans[i]:spans[i + 1]] = df_init['sum_symp'].data[spans[i+1]-1]
        min_date = df_init['date_update'].data[spans[i]:spans[i+1]].min()
        interval = df_init['date_update'].data[spans[i]:spans[i+1]] - min_date
        interval_days[spans[i]:spans[i+1]] = [int(d/86400) for d in interval]
    df_init.create_numeric('first_entry_sum', 'int16').data.write(first_entry_sum)
    df_init.create_numeric('last_entry_sum', 'int16').data.write(last_entry_sum)
    df_init.create_numeric('interval_days', 'int16').data.write(interval_days)

    print('Performing interpolation')
    df_interp = interpolate_date(df_init)

    return df_interp


# def interpolate_healthy(dfg):
#     dfg['health'] = np.nan
#     dfg['health'] = np.where(dfg['sum_symp'] == 0, 1, dfg['health'])
#     dfg['health'] = np.where(dfg['sum_symp'] > 0, 0, dfg['health'])
#     dfg['health_interp'] = dfg['health'].ffill()
#     dfg['health_back'] = dfg['health'].bfill()
#     dfg['nan_healthy'] = np.where(dfg['health_interp'] == 1, np.nan, dfg['health_interp'])
#     dfg['nan_uh'] = np.where(dfg['health_interp'] == 0, np.nan, 1 - dfg['health_interp'])
#     dfg['count_healthy'] = dfg.nan_healthy.isnull().astype(int).groupby(
#         dfg.nan_healthy.notnull().astype(int).cumsum()).cumsum()
#     dfg['count_nans'] = dfg.health_status.isnull().astype(int).groupby(
#         dfg.health_status.notnull().astype(int).cumsum()).cumsum()
#
#     dfg['count_nothealthy'] = dfg.nan_uh.isnull().astype(int).groupby(
#         dfg.nan_uh.notnull().astype(int).cumsum()).cumsum()
#
#     dfg['nan_or_health'] = np.where(dfg['health'] == 0, 1, np.nan)
#     dfg['count_nans_or_health'] = dfg.nan_or_health.isnull().astype(int).groupby(
#         dfg.nan_or_health.notnull().astype(int).cumsum()).cumsum()
#
#     # max_cnans = dfg['count_healthy'].max()
#     # idx_cnans = dfg['count_healthy'].idxmax()
#     return dfg

def define_gh_noimp(dfg, gap_healthy=7):  # still using pandas df
    print(np.unique(dfg['patient_id']), len(dfg.index), len(np.unique(dfg.index)))
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
        # df_interp['health_status'] = np.nan
        # df_interp['health_status'] = np.where(df_interp['sum_symp'] > 0, 0, df_interp['health_status'])
        # df_interp['health_status'] = np.where(df_interp['sum_symp'] == 0, 1, df_interp['health_status'])
        #todo confirm re-cal sum_symp rename to sum_symp_interpo
        sum_symp = np.zeros(len(df_interp['patient_id'].data), 'float')
        for k in list_symptoms:
            sum_symp += df_interp[k].data[:]
        df_interp['sum_symp'].data.write(sum_symp)

        health_status = np.where(df_interp['sum_symp'].data[:] > 0, 0,
                                 np.where(df_interp['sum_symp'].data[:]== 0,1, np.nan))
        df_interp.create_numeric('health_status', 'int8').data.write(health_status)

        # df_interp['max_symp'] = df_interp.groupby('patient_id')['sum_symp'].transform('max')
        # df_interp = df_interp.groupby('patient_id').apply(lambda group: interpolate_healthy(group))
        health = np.where(df_interp['sum_symp'].data[:]  == 0, 1, np.where(df_interp['sum_symp'].data[:] > 0, 0, np.nan))
        df_interp.create_numeric('health', 'int8').data.write(health)
        spans = df_interp['patient_id'].get_spans()
        max_symp = np.zeros(len(df_interp['patient_id'].data), 'float')
        health_interp = np.zeros(len(df_interp['patient_id'].data), 'int8')
        health_back = np.zeros(len(df_interp['patient_id'].data), 'int8')
        count_healthy = np.zeros(len(df_interp['patient_id'].data), 'int8')
        count_nothealthy = np.zeros(len(df_interp['patient_id'].data), 'int8')
        count_nans_or_health = np.zeros(len(df_interp['patient_id'].data), 'int8')
        count_nans = np.zeros(len(df_interp['patient_id'].data), 'int8')
        for i in range(len(spans)-1):
            max_symp[spans[i]:spans[i+1]] = np.max(df_interp['sum_symp'].data[spans[i]:spans[i+1]])
            #interpolate_healthy functions
            health_interp[spans[i]:spans[i+1]] = pd.Series(df_interp['health'].data[spans[i]:spans[i+1]]).ffill()
            health_back[spans[i]:spans[i+1]] = pd.Series(df_interp['health'].data[spans[i]:spans[i+1]]).bfill()
            nan_healthy = np.where(health_interp[spans[i]:spans[i+1]] == 1, np.nan, health_interp[spans[i]:spans[i+1]])
            nan_healthy = pd.Series(nan_healthy)
            count_healthy[spans[i]:spans[i+1]] = nan_healthy.isnull().astype(int).groupby( nan_healthy.notnull().astype(int).cumsum()).cumsum()

            nan_uh = np.where(health_interp[spans[i]:spans[i+1]] == 0, np.nan, 1 - health_interp[spans[i]:spans[i+1]])
            nan_uh = pd.Series(nan_uh)
            count_nothealthy[spans[i]:spans[i+1]] = nan_uh.isnull().astype(int).groupby( nan_uh.notnull().astype(int).cumsum()).cumsum()

            nan_or_health = np.where(df_interp['health'].data[spans[i]:spans[i+1]] == 0, 1, np.nan)
            nan_or_health = pd.Series(nan_or_health)
            count_nans_or_health[spans[i]:spans[i+1]] = nan_or_health.isnull().astype(int).groupby(nan_or_health.notnull().astype(int).cumsum()).cumsum()

            health_status = pd.Series(df_interp['health_status'].data[spans[i]:spans[i+1]])
            count_nans[spans[i]:spans[i+1]] = health_status.isnull().astype(int).groupby(health_status.notnull().astype(int).cumsum()).cumsum()

        df_interp.create_numeric('max_symp', 'float').data.write(max_symp)
        df_interp.create_numeric('health_interp', 'int8').data.write(health_interp)
        df_interp.create_numeric('health_back', 'int8').data.write(health_back)
        df_interp.create_numeric('count_healthy', 'int8').data.write(count_healthy)
        df_interp.create_numeric('count_nothealthy', 'int8').data.write(count_nothealthy)
        df_interp.create_numeric('count_nans_or_health', 'int8').data.write(count_nans_or_health)
        df_interp.create_numeric('count_nans', 'int8').data.write(count_nans)

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
                      'first_stuh_nhg%d' % f, 'last_stuh_nhg%d' % f, 'date_first_stuh_hg%d' % f, 'date_latest_first_%d' % f]:
                newdf.create_numeric(fld, 'int8')
            with ThreadPoolExecutor(max_workers=1) as executor:
                threadset = set()
                for i in range(len(spans) - 1):  # alter the define_gh_noimp function to exetera is too complicated, try using pandas
                    if i % 1000 == 0: print(datetime.now(), i, ' number of define_gh_noimp processed.')
                    tempd = {}
                    for fld in df_interp.keys():
                        tempd[fld] = df_interp[fld].data[spans[i]:spans[i + 1]]
                    tempdf = pd.DataFrame(tempd, index=tempd['interval_days'])
                    t = executor.submit(define_gh_noimp, tempdf)
                    threadset.add(t)
                for t in as_completed(threadset):
                    resultdb = t.result()
                    for fld in resultdb.columns:
                        newdf[fld].data.write(resultdb[fld])


            df_interp = newdf
            #df_interp.to_csv(saved_name)

    return df_interp


def determine_meeting_criteria(df_interp, days=[7]):
    from datetime import datetime, timedelta

    timemin = timedelta(days=-7)
    import time

    #struct_time = datetime.strptime("20 Jul 20", "%d %b %y").date()
    delay = (df_interp['created_at'].data[:] - datetime.strptime("20 Jul 20", "%d %b %y").timestamp())/86400  # delay is in days, todo confirm use created_at
    #df_interp['dropped'] = np.where(df_interp['delay'] <= timemin, 1, 0)
    last_entry_sum = np.zeros(len(df_interp['patient_id'].data), 'float')
    spans = df_interp['patient_id'].get_spans()
    for i in range(len(spans)-1):
        last_entry_sum[spans[i]:spans[i+1]] = df_interp['created_at'].data[spans[i+1]-1]
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


# df_init = pd.read_csv('/home/csudre/MountedSpace/Covid/SympImp_ForInterp_1.csv')
# for c in df_init.columns:
#     print(c)
# df_init = creating_nanvalues(df_init)
# df_interp = creating_interpolation(df_init)
# print('Saving interpolated ')
# df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedImpSHSymp_1.csv')
#
# print('loading file')
# #df_test = pd.read_csv('/home/csudre/MountedSpace/Covid/TestedPositiveTestDetails.csv')
# df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedImpSHSymp_1.csv')
#
#
# print('processing')
# df_interp = creating_duration_healthybased(df_interp,hi=False)
# df_interp = determine_meeting_criteria(df_interp)
# print('sving meeing crite')
# df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedImpSHSymp_1.csv')
# # Checking hosp
# print('Checking hosp')
# # df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')
# # print(df_interp.shape)
# df_interp['hosp_valid'] = (df_interp['interval_days']>df_interp['first_stuh_hg7']) * (df_interp['location']>1)
# df_interp['hosp_check'] = df_interp.groupby('patient_id')['hosp_valid'].transform('max')
# #
# df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedImpSHSymp_1.csv')
# df_unique = df_interp.sort_values(['patient_id','sum_symp'],ascending=False).drop_duplicates('patient_id')
# df_pat = pd.read_csv('/home/csudre/MountedSpace/Covid/PositiveSympStartHealthy_PatDetails.csv')
#
# df_merge = pd.merge(df_unique,df_pat, left_on='patient_id',right_on='id')
# df_merge['age'] = 2020 - df_merge['year_of_birth']
# df_merge.to_csv('/home/csudre/MountedSpace/Covid/UniqueForDuration.csv')
# for f in list_symptoms:
#     df_interp[f] = np.where(df_interp['health_interp']==1,0,df_interp[f])

def treat_interp(df_interp):
    print(datetime.now(), ' Final processing...')
    spans = df_interp['patient_id'].get_spans()
    count_values = np.zeros(len(df_interp['patient_id']), 'int16')
    length_log = np.zeros(len(df_interp['patient_id']), 'int16')
    for i in range(len(spans)-1):
        data = df_interp['interval_days'].data[spans[i]:spans[i+1]]
        count_values[spans[i]:spans[i+1]] = np.count_nonzero(data)
        length_log[spans[i]:spans[i+1]] = np.max(data)
    df_interp.create_numeric('count_values', 'int16').data.write(count_values)
    df_interp.create_numeric('length_log', 'int16').data.write(length_log)

    for d in [7]:  # 14
        duration = df_interp['last_stuh_hg%d'%d].data[:] - df_interp['first_stuh_hg%d'%d].data[:] + 1
        df_interp.create_numeric('duration%d'%d, 'int16').data.write(duration)
        last_status = np.where(df_interp['last_stuh_hg%d'%d].data[:] == df_interp['interval_days'].data[:],df_interp['health_interp'].data[:],np.nan)
        spans = df_interp['patient_id'].get_spans()
        for i in range(len(spans)-1):
            last_status[spans[i]:spans[i+1]] = np.max(last_status[spans[i]:spans[i+1]])
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

    spans = df_interp['patient_id'].get_spans()
    first_sick = np.zeros(len(df_interp['patient_id']), 'float')
    last_sick = np.zeros(len(df_interp['patient_id']), 'float')
    first_log = np.zeros(len(df_interp['patient_id']), 'float')
    last_log = np.zeros(len(df_interp['patient_id']), 'float')
    for i in range(len(spans) - 1):
        first_sick[spans[i]:spans[i+1]] = np.min(df_interp['date_sick'].data[spans[i]:spans[i+1]])
        last_sick[spans[i]:spans[i + 1]] = np.max(df_interp['date_sick'].data[spans[i]:spans[i + 1]])
        first_log[spans[i]:spans[i + 1]] = np.min(df_interp['created_at'].data[spans[i]:spans[i + 1]])
        last_log[spans[i]:spans[i + 1]] = np.max(df_interp['created_at'].data[spans[i]:spans[i + 1]])
    df_interp.create_timestamp('first_sick').data.write(first_sick)
    df_interp.create_timestamp('last_sick').data.write(last_sick)
    df_interp.create_timestamp('first_log').data.write(first_log)
    df_interp.create_timestamp('last_log').data.write(last_log)

    h_extent = np.where(df_interp['created_at'].data[:] >= df_interp['first_sick'].data[:], df_interp['health_interp'].data[:], 0)
    h_extent = np.where(df_interp['created_at'].data[:] <= df_interp['last_sick'].data[:], h_extent, 0)
    df_interp.create_numeric('h_extent', 'int16').data.write(h_extent)
    count_h_extent= np.zeros(len(df_interp['patient_id']), 'int')
    for i in range(len(spans) - 1):
        count_h_extent[spans[i]:spans[i+1]] = np.sum(df_interp['h_extent'].data[spans[i]:spans[i+1]])
    df_interp.create_numeric('count_h_extent', 'int16').data.write(count_h_extent)

    #df_interp['prop_h'] = df_interp['count_healthy'].replace(0, np.nan)
    prop_h = np.where(df_interp['count_healthy'].data[:]==np.nan, 0, df_interp['count_healthy'].data[:])
    df_interp.create_numeric('prop_h', 'int16').data.write(prop_h)
    #df_interp['prop_h'] = df_interp.groupby('patient_id')['prop_h'].ffill()  # shouldn't be useful as no nan left

    first_s = np.where(df_interp['count_nothealthy'].data[:] == 1, df_interp['created_at'].data[:], np.nan)
    df_interp.create_timestamp('first_s').data.write(first_s)
    first_sicki = np.where(df_interp['created_at'].data[:] == df_interp['first_sick'].data[:], df_interp['interval_days'].data[:], np.nan)
    last_sicki = np.where(df_interp['created_at'].data[:] == df_interp['last_sick'].data[:], df_interp['interval_days'].data[:], np.nan)


    max_nans = np.zeros(len(df_interp['count_nans'].data),'int16')
    spans = df_interp['patient_id'].get_spans()
    count_nans = df_interp['count_nans'].data[:]
    for i in range(len(spans)-1):
        first_sicki[spans[i]:spans[i+1]] = np.min(first_sicki[spans[i]:spans[i+1]])
        last_sicki[spans[i]:spans[i+1]] = np.min(last_sicki[spans[i]:spans[i+1]])
        max_nans[spans[i]:spans[i+1]] = np.max(count_nans[spans[i]:spans[i+1]])

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


    max_symp = np.zeros(len(df_interp['sum_symp'].data), 'float')
    max_countuh = np.zeros(len(df_interp['count_nothealthy'].data), 'int16')
    max_countnanbef = np.zeros(len(df_interp['count_nanbef'].data), 'int16')
    max_countuhbef = np.zeros(len(df_interp['count_uhbef'].data), 'int16')
    max_symp_at = np.zeros(len(df_interp['sum_symp_at'].data), 'float')
    numb_at = np.zeros(len(df_interp['created_aroundtest'].data), 'int16')

    for i in range(len(spans) - 1):
        max_symp[spans[i]:spans[i+1]] = np.max(df_interp['sum_symp'].data[spans[i]:spans[i+1]])
        max_countuh[spans[i]:spans[i+1]] = np.max(df_interp['count_nothealthy'].data[spans[i]:spans[i+1]])
        max_countnanbef[spans[i]:spans[i+1]] = np.max(count_nanbef[spans[i]:spans[i+1]])
        max_countuhbef[spans[i]:spans[i+1]] = np.max(count_uhbef[spans[i]:spans[i+1]])
        max_symp_at[spans[i]:spans[i+1]] = np.max(sum_symp_at[spans[i]:spans[i+1]])
        numb_at[spans[i]:spans[i+1]] = np.sum(created_aroundtest[spans[i]:spans[i+1]])

    df_interp['max_symp'].data.write(max_symp)
    df_interp.create_numeric('max_countuh', 'int16').data.write(max_countuh)
    df_interp.create_numeric('max_countnanbef', 'int16').data.write(max_countnanbef)
    df_interp.create_numeric('max_countuhbef', 'int16').data.write(max_countuhbef)
    df_interp.create_numeric('max_symp_at', 'float').data.write(max_symp_at)
    df_interp.create_numeric('numb_at', 'int16').data.write(numb_at)

    return df_interp


def output_three_files(df_proc):
    '''
    1. One file with patients grouped and with duration and max number of symptoms;
    2. One file with the current results - all patients, with all reported days;
    3. Only patients that pass the criteria with missing data less than 7 days - check column s as
    df_interp['valid_timing'] = np.where(np.logical_and(df_interp['diff_test']>=-14, df_interp['diff_test']<7),1,0)

    '''
    pass


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
        df_proc = creating_nanvalues(df_proc)
    if 'interpolate' in args.process:
        print('performing interpolation')
        df_proc = creating_interpolation(df_proc)
        #df_proc.to_csv('interpolation.csv')
        save_df_to_csv(df_proc, 'interpolation.csv' ,['patient_id','interval_days']+list_symptoms+['created_at','sum_symp','country_code',
                                                                              'location','treatment','updated_at','first_entry_sum','last_entry_sum','date_update','interval','sob2','fatigue2','created_interp'] )
    if 'duration' in args.process:
        print('processing duration')
        if 'date_effective_test' not in df_proc.keys():
            tempdf = TEMPH5.create_dataframe('df_proc')
            df.merge(df_proc, df_test, dest=tempdf, left_on='patient_id', right_on='patient_id' )
            #todo drop duplicates
            df_proc = tempdf
            del df_proc['patient_id_l']
            df_proc['patient_id'] = df_proc['patient_id_r']

        # if 'date_effective_test' not in df_proc.columns:
        #     df_test = pd.read_csv(args.test_file)
        #     df_test_min = df_test.drop_duplicates('patient_id')
        #     df_proc = pd.merge(df_proc, df_test_min, on='patient_id', how='left')
        df_proc = creating_duration_healthybased(df_proc, args.output, days=args.days, hi=args.hi)
        df_proc.to_csv('duration.csv')
    if 'criteria' in args.process:
        print('Checking criteria')
        df_proc = determine_meeting_criteria(df_proc, days=args.days)
        df_proc.to_csv('criteria.csv')

    exit(0)
    # todo group duration by each patient @Liane/Michela
    # todo output data filtering some column @Liane/Michela
    output_three_files(df_proc)

    #calculate duration
    treat_interp(df_proc)  # from notebook

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
