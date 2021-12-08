import math

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys

from exetera.core.session import Session
import exetera.core.dataframe as df
import exetera.core.dataset as ds
import exetera.core.operations as ops

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


def interpolate_date(df_train, list_symptoms=list_symptoms, col_day='interval_days'):
    # df_train['sob2'] = np.round(df_train['shortness_of_breath'] * 1.0 / 3, 0)
    # df_train['fatigue2'] = np.round(df_train['fatigue'] * 1.0 / 3, 0)
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

    full_idx = np.arange(df_test_comb['interval_days'].data[:].min(), df_test_comb['interval_days'].data[:].max())

    # for f in list_symptoms + ['fatigue', 'shortness_of_breath']:
    #     df_test_comb[f] = df_test_comb[f].fillna(0)
    #df_test_comb_ind = df_test_comb.set_index('interval_days')

    # df_test2 = df_test_comb_ind.groupby('patient_id', as_index=False).apply(lambda group: group.reindex(full_idx)).reset_index(level=0, drop=True).sort_index()
    expanded_df = TEMPH5.create_dataframe('expanded_df')
    df_test_comb['patient_id'].create_like(expanded_df, 'patient_id')
    df_test_comb['interval_days'].create_like(expanded_df, 'interval_days')
    for f in list_symptoms:
        expanded_df.create_numeric(f,'float')

    spans = df_test_comb['patient_id'].get_spans()
    for i in range(len(spans)-1):
        #write patient_id, idx
        pid = df_test_comb['patient_id'].data[spans[i]]
        expanded_df['patient_id'].data.write([pid for _ in range(len(full_idx))])
        expanded_df['interval_days'].data.write(full_idx)
        idx = df_test_comb['interval_days'].data[spans[i]:spans[i+1]]
        for f in list_symptoms:
            data = df_test_comb[f].data[spans[i]:spans[i+1]]
            pdseries = pd.Series(data, index=idx)
            pdseries = pdseries.reindex(full_idx)
            pdseries = pdseries.interpolate(method='linear', limit_area='inside')
            expanded_df[f].data.write(pdseries.values)
    #drop na, rename axis, drop column, reset_index
    filter = np.isnan(expanded_df['fatigue'].data[:])
    for f in list_symptoms:
        filter &= np.isnan(expanded_df[f].data[:])
    expanded_df.apply_filter(filter)
    return expanded_df


    # def f_inter(x):
    #     #         full_idx = np.arange(x['interval_days'].min(), x['interval_days'].max())
    #     x = x.reindex(full_idx)
    #     for f in list_symptoms:
    #         x[f] = x[f].replace('False', 0)
    #         x[f] = x[f].replace('True', 1)
    #         x[f] = x[f].astype(float)
    #         x[f] = x[f].interpolate(method='linear', limit_area='inside')
    #     x = x.dropna(subset=list_symptoms + ['date_update'], how='all')
    #     return (x)
    #
    # df_interp = df_test_comb_ind.groupby('patient_id').apply(f_inter).rename_axis(('patient_id', 'interval_days')).drop(
    #     'patient_id', 1).reset_index()
    # return df_interp





def check_succession_healthy(dfg, limit):
    dfg['sum_nans'] = np.where(dfg['sum_symp'] == 0, np.nan, dfg['sum_symp'])
    dfg['count_healthy'] = dfg.sum_nans.isnull().astype(int).groupby(
        dfg.sum_nans.notnull().astype(int).cumsum()).cumsum()
    dfg['not_healthy'] = dfg.sum_nans.notnull().astype(int).groupby(dfg.sum_nans.isnull().astype(int).cumsum()).cumsum()
    dfg['interval_interp'] = np.arange(0, dfg.shape[0])
    dfg = dfg.reset_index(drop=True)
    dfg['interp_sum'] = dfg['sum_symp'].interpolate('linear')
    max_cnans = dfg['count_healthy'].max()
    idx_cnans = dfg['count_healthy'].idxmax()
    #     dfg['interval_interp'] = dfg['interval_days'].interpolate(method='linear',limit_area='inside')
    dfg['ln'] = dfg['count_healthy'].ge(limit)
    dfg['days_max'] = 0
    value_max_e = dfg.sort_values(['sum_symp', 'interval_interp'], ascending=False).head(1)['interval_interp']
    value_max_b = dfg.sort_values(['sum_symp'], ascending=False).head(1)['interval_interp']

    dfg['days_max_b'] = np.asarray(value_max_b)[0]
    dfg['days_max_e'] = np.asarray(value_max_e)[0]
    print(value_max_e, value_max_b)
    dfg = dfg.sort_values('interval_interp').reset_index(drop=True)
    # print(dfg['days_max'].min())
    dfg['lne'] = dfg['ln'].astype(float) * dfg['interval_interp'].gt(dfg['days_max_e']).astype(float) * dfg[
        'interp_sum'].lt(2).astype(float)
    dfg['lnb'] = dfg['ln'].astype(float) * dfg['interval_interp'].lt(dfg['days_max_b']).astype(float)
    dfg_temp = dfg
    if dfg['lnb'].max() > 0:
        print("begin issue")
        dfg_lnb_last = dfg_temp[dfg_temp['lnb'] > 0].last_valid_index()
        print(np.asarray(dfg_temp.loc[dfg_lnb_last]['interp_sum']))
        dfg_temp = dfg_temp.loc[dfg_lnb_last:]
        dfg_temp = dfg_temp.reset_index(drop=True)
    if dfg['lne'].max() > 0:
        print("end issue")
        dfg_lne_first = dfg_temp[dfg_temp['lne'] > 0].first_valid_index()
        #         print(dfg_temp.iloc[dfg_temp['lne'].idxmax()]['interval_interp'])
        dfg_temp = dfg_temp[
            dfg_temp['interval_interp'] < dfg_temp.iloc[dfg_temp['lne'].idxmax()]['interval_interp'] - limit + 2]
    #         print(np.asarray(max_cnans), np.asarray(idx_cnans))
    #         dfg_temp = dfg.iloc[:np.asarray(idx_cnans)-np.asarray(max_cnans)]
    print(dfg_temp.shape, dfg.shape, dfg_temp[
        ['interval_interp', 'days_max_e', 'days_max_b', 'sum_symp', 'interp_sum', 'duration', 'ln', 'lnb', 'lne']])
    return dfg


def define_gh_noimp(dfg, gap_healthy=7):
    print(np.unique(dfg['patient_id']), len(dfg.index), len(np.unique(dfg.index)))
    #id_max = dfg[dfg['sum_symp'] == dfg['max_symp']]['interval_days'].min()
    #id_max2 = dfg[dfg['sum_symp'] == dfg['max_symp']]['interval_days'].max()
    id_test = dfg[(dfg['created_at'] <= dfg['date_effective_test'] + 7 * 86400)]['interval_days'].max()
    id_test_ab = dfg[dfg['created_at'] <= dfg['date_effective_test'] - 14 * 86400]['interval_days'].max()
    # Check if id_test
    dfg_before_test = dfg[dfg['interval_days'] <= id_test]
    dfg_before_test_ab = dfg[dfg['interval_days'] <= id_test_ab]
    #dfg_after_test = dfg[dfg['interval_days'] >= id_test]
    #dfg_before_max = dfg[dfg['interval_days'] <= id_max]
    #dfg_after_max = dfg[dfg['interval_days'] >= id_max2]
    max_count_test_nh = dfg_before_test['count_nothealthy'].max()
    #max_count_bh = dfg_before_max['count_healthy'].max()
    #max_count_ah = dfg_after_max['count_healthy'].max()
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

            print('creating non nans version')
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
        print(list(set(possible_firsts)), list(set(possible_lasts)))

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
        print('WARNING: No log before test')
        dfg['first_stuh_hg%d' % gap_healthy] = np.nan
        dfg['last_stuh_hg%d' % gap_healthy] = np.nan
    print(np.max(dfg.index), 'is index max of dfg')

    return dfg


def define_gh_imp(dfg, gap_healthy=7):
    id_max = dfg[dfg['imp'] > 0]['interval_days'].min()
    id_max2 = dfg[dfg['imp'] > 0]['interval_days'].max()

    dfg_before_imp = dfg[dfg['interval_days'] <= id_max]
    dfg_after_imp = dfg[dfg['interval_days'] >= id_max2]
    max_count_test_nh = dfg_before_imp['count_nothealthy'].max()
    #max_count_bh = dfg_before_imp['count_healthy'].max()
    #max_count_ah = dfg_after_imp['count_healthy'].max()

    if dfg_before_imp.shape[0] > 0:
        if max_count_test_nh > 0:
            print('treating test', dfg_before_imp.shape[0], max_count_test_nh)
            last_count_ah = dfg_after_imp['count_healthy'].tail(1).to_numpy()[0]
            #min_count_bh = dfg_before_imp['count_healthy'].min()
            dfg['first_stuh_hg%d' % gap_healthy] = dfg_before_imp[dfg_before_imp['count_nothealthy'] == 1][
                'interval_days'].min()
            dfg['last_stuh_hg%d' % gap_healthy] = dfg_after_imp[dfg_after_imp['count_healthy'] == last_count_ah][
                                                      'interval_days'].max() - last_count_ah

            dfg['first_stuh_nhg%d' % gap_healthy] = dfg_before_imp[dfg_before_imp['count_nothealthy'] == 1][
                'interval_days'].min()
            dfg['last_stuh_nhg%d' % gap_healthy] = dfg_after_imp[dfg_after_imp['count_healthy'] == last_count_ah][
                                                       'interval_days'].max() - last_count_ah

            dbm_hg = dfg_before_imp[dfg_before_imp['count_healthy'] > gap_healthy]
            dam_hg = dfg_after_imp[dfg_after_imp['count_healthy'] > gap_healthy]

            last_healthy_gap = dbm_hg['interval_days'].max()
            first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
            #     print(last_healthy_gap)
            if dbm_hg.shape[0] > 0:
                dfg['first_stuh_hg%d' % gap_healthy] = last_healthy_gap + 1
            if dam_hg.shape[0] > 0:
                dfg['last_stuh_hg%d' % gap_healthy] = first_healthy_gap

            dbm_hg = dfg_before_imp[dfg_before_imp['count_nans_or_health'] > gap_healthy]
            dam_hg = dfg_after_imp[dfg_after_imp['count_nans_or_health'] > gap_healthy]
            last_healthy_gap = dbm_hg['interval_days'].max()
            first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
            #     print(last_healthy_gap)
            if dbm_hg.shape[0] > 0:
                dfg['first_stuh_nhg%d' % gap_healthy] = last_healthy_gap + 1
            if dam_hg.shape[0] > 0:
                dfg['last_stuh_nhg%d' % gap_healthy] = first_healthy_gap
    return dfg


# def date_range(dfg, c='date_update'):
#     dfg['interval'] = dfg[c] - dfg[c].min()
#     return dfg


def creating_interpolation(df_init):
    print('Applying date range')
    # df_init['first_entry_sum'] = df_init.groupby('patient_id')['sum_symp'].transform('first')
    # df_init['last_entry_sum'] = df_init.groupby('patient_id')['sum_symp'].transform('last')
    df_init['date_update'] = df_init['created_at']
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
    #todo confirm first_entry_sum, last_entry_sum, interval_days not used?

    print('Performing interpolation')
    df_interp = interpolate_date(df_init)
    #df_interp['created_interp'] = df_interp['created_at'].interpolate('linear') todo confirm
    df_interp['created_interp'] = df_interp['created_at']
    return df_interp


def interpolate_healthy(dfg):
    dfg['health'] = np.nan
    dfg['health'] = np.where(dfg['sum_symp'] == 0, 1, dfg['health'])
    dfg['health'] = np.where(dfg['sum_symp'] > 0, 0, dfg['health'])
    dfg['health_interp'] = dfg['health'].ffill()
    dfg['health_back'] = dfg['health'].bfill()
    dfg['nan_healthy'] = np.where(dfg['health_interp'] == 1, np.nan, dfg['health_interp'])
    dfg['nan_uh'] = np.where(dfg['health_interp'] == 0, np.nan, 1 - dfg['health_interp'])
    dfg['count_healthy'] = dfg.nan_healthy.isnull().astype(int).groupby(
        dfg.nan_healthy.notnull().astype(int).cumsum()).cumsum()
    dfg['count_nans'] = dfg.health_status.isnull().astype(int).groupby(
        dfg.health_status.notnull().astype(int).cumsum()).cumsum()

    dfg['count_nothealthy'] = dfg.nan_uh.isnull().astype(int).groupby(
        dfg.nan_uh.notnull().astype(int).cumsum()).cumsum()

    dfg['nan_or_health'] = np.where(dfg['health'] == 0, 1, np.nan)
    dfg['count_nans_or_health'] = dfg.nan_or_health.isnull().astype(int).groupby(
        dfg.nan_or_health.notnull().astype(int).cumsum()).cumsum()

    # max_cnans = dfg['count_healthy'].max()
    # idx_cnans = dfg['count_healthy'].idxmax()
    return dfg

def creating_duration_healthybased(df_interp, saved_name, days=[7], hi=False, force=True):
    if hi == True:
        # df_interp['health_status'] = np.nan
        # df_interp['health_status'] = np.where(df_interp['sum_symp'] > 0, 0, df_interp['health_status'])
        # df_interp['health_status'] = np.where(df_interp['sum_symp'] == 0, 1, df_interp['health_status'])
        health_status = np.where(df_interp['sum_symp'].data[:] > 0, 0,
                                 np.where(df_interp['sum_symp'].data[:]== 0,1, np.nan))
        df_interp.create_numeric('health_status', 'int8').data.write(health_status)

        # df_interp['max_symp'] = df_interp.groupby('patient_id')['sum_symp'].transform('max')
        # df_interp = df_interp.groupby('patient_id').apply(lambda group: interpolate_healthy(group))
        health = np.where(dfg['sum_symp'] == 0, 1, np.where(dfg['sum_symp'] > 0, 0, np.nan))
        df_interp.create_numeric('health', 'int8').data.write(health)
        spans = df_interp['patient_id'].get_spans()
        max_symp = np.zeros(len(df_interp['patient_id'].data), 'int8')
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

            nan_uh = np.where(df_interp['health_interp'].data[spans[i]:spans[i+1]] == 0, np.nan, 1 - df_interp['health_interp'].data[spans[i]:spans[i+1]])
            nan_uh = pd.Series(nan_uh)
            count_nothealthy[spans[i]:spans[i+1]] = nan_uh.isnull().astype(int).groupby( nan_uh.notnull().astype(int).cumsum()).cumsum()

            nan_or_health = np.where(df_interp['health'].data[spans[i]:spans[i+1]] == 0, 1, np.nan)
            nan_or_health = pd.Series(nan_or_health)
            count_nans_or_health[spans[i]:spans[i+1]] = nan_or_health.isnull().astype(int).groupby(nan_or_health.notnull().astype(int).cumsum()).cumsum()

            health_status = pd.Series(df_interp['health_status'].data[spans[i]:spans[i+1]])
            count_nans[spans[i]:spans[i+1]] = health_status.isnull().astype(int).groupby(health_status.notnull().astype(int).cumsum()).cumsum()

        df_interp.create_numeric('count_healthy', 'int8').data.write(hecount_healthyalth)
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
            # df_interp.to_csv(saved_name) # todo here csv file is only saved for the last day
            spans = df_interp['patient_id'].get_spans()
            global TEMPH5
            newdf = TEMPH5.create_dataframe('newdf')
            for f in df_interp.keys():
                df_interp[f].create_like(newdf, f)
            for f in ['first_stuh_hg%d' % gap_healthy, 'latest_first_%d' % gap_healthy, 'last_stuh_hg%d' % gap_healthy,
                      'first_stuh_nhg%d' % gap_healthy, 'last_stuh_nhg%d' % gap_healthy, 'date_first_stuh_hg%d' % gap_healthy, 'date_latest_first_%d' % gap_healthy]:
                newdf.create_numeric(f, 'int8')
            for i in range(len(spans)-1):  # alter the define_gh_noimp function to exetera is too complicated, try using pandas
                tempd = {}
                for f in df_interp.keys():
                    tempd[f] = df_interp[f].data[spans[i]:spans[i+1]]
                tempdf = pd.DataFrame(tempd, index=tempd['interval_days'])
                resultdb = define_gh_noimp(tempdf, gap_healthy=f)
                for f in resultdb.columns:
                    newdf[f].data.write(resultdb[f].values)

            df_interp = newdf
            df_interp.to_csv(saved_name)

    return df_interp


def determine_meeting_criteria(df_interp, days=[7]):
    from datetime import datetime, timedelta

    timemin = timedelta(days=-7)
    import time

    struct_time = datetime.strptime("20 Jul 20", "%d %b %y").date()
    df_interp['delay'] = pd.to_datetime(df_interp['date_update'], infer_datetime_format=True).dt.date - struct_time
    delay = (df_interp['date_update'] - datetime.strptime("20 Jul 20", "%d %b %y").timestamp())/86400
    #df_interp['dropped'] = np.where(df_interp['delay'] <= timemin, 1, 0)
    # todo confirm last_entry_sum should be there already
    # df_interp['last_entry_sum'] = df_interp.groupby('patient_id')['sum_symp'].transform('last')
    # last_entry_sum = np.zeros(len(df_interp['patient_id'].data), 'int16')
    # spans = df_interp['patient_id'].get_spans()
    # for i in range(len(spans)-1):
    #     last_entry_sum[spans[i]:spans[i+1]] = df_interp['date_update'].data[spans[i+1]-1]


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

            df_interp['postcrit_aok%d' % f].data.write( np.where(
                np.logical_and(delay < 86400*int(f), df_interp['last_entry_sum'].data[:] == 0), 1,
                df_interp['postcrit_aok%d' % f]))
            # df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')

    for f in list_symptoms:
        check_ = np.where(df_interp['health_interp'].data[:] == 0, df_interp[f].data[:], 0)
        df_interp.create_numeric('check_'+f, 'int16').data.write(check_)

        #df_interp['sumcheck_' + f] = df_interp.groupby('patient_id')['check_' + f].transform('sum')
        sumcheck_ = np.zeros(len(df_interp['patient_id'].data), 'int16')
        spans = df_interp['patient_id'].get_spans()
        for i in range(len(spans)-1):
            sumcheck_[spans[i]:spans[i+1]] = np.sum(check_[spans[i]:spans[i+1]])
        df_interp.create_numeric('sumcheck_' + f, 'int16').data.write(sumcheck_)

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
            startdf = np.zeros(len(df_interp['patient_id'].data), 'int16')
            sumsickdf = np.zeros(len(df_interp['patient_id'].data), 'int16')
            for i in range(len(spans) - 1):
                startdf[spans[i]:spans[i + 1]] = np.min(daydf[spans[i]:spans[i + 1]])
                sumsickdf[spans[i]:spans[i + 1]] = np.sum(sickdf[spans[i]:spans[i + 1]])
            df_interp.create_numeric('start%d_' % d + f, 'int16').data.write(startdf)
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
            startmild = np.zeros(len(df_interp['patient_id']), 'int16')
            endmild = np.zeros(len(df_interp['patient_id']), 'int16')
            summild = np.zeros(len(df_interp['patient_id']), 'int16')
            startsever = np.zeros(len(df_interp['patient_id']), 'int16')
            endsever = np.zeros(len(df_interp['patient_id']), 'int16')
            sumsever = np.zeros(len(df_interp['patient_id']), 'int16')
            for i in range(len(spans) - 1):
                startmild[spans[i]:spans[i+1]] = np.min(daymild[spans[i]:spans[i+1]])
                endmild[spans[i]:spans[i+1]] = np.max(daymild[spans[i]:spans[i+1]])
                summild[spans[i]:spans[i+1]] = np.sum(sickmild[spans[i]:spans[i+1]])

                startsever[spans[i]:spans[i+1]] = np.min(daysever[spans[i]:spans[i+1]])
                endsever[spans[i]:spans[i + 1]] = np.max(daysever[spans[i]:spans[i + 1]])
                sumsever[spans[i]:spans[i + 1]] = np.sum(sicksever[spans[i]:spans[i + 1]])

            df_interp.create_numeric('start%d_' % d + f + '_mild', 'int16').data.write(startmild)
            df_interp.create_numeric('end%d_' % d + f + '_mild', 'int16').data.write(endmild)
            df_interp.create_numeric('sumsick%d_' % d + f + '_mild', 'int16').data.write(summild)

            df_interp.create_numeric('start%d_' % d + f + '_severe', 'int16').data.write(startsever)
            df_interp.create_numeric('end%d_' % d + f + '_severe', 'int16').data.write(endsever)
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
        df_proc.to_csv('interpolation.csv')
    if 'duration' in args.process:
        print('processing duration')
        if 'date_effective_test' not in df_proc.keys():
            tempdf = TEMPH5.create_dataframe('df_proc')
            df.merge(df_proc, df_test, dest=tempdf, left_on='patient_id', right_on='patient_id' )
            #todo drop duplicates
            df_proc = tempdf

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
    #if 'hosp' in args.process:  # not updated
        # print('Checking hosp')
        # # df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')
        # # print(df_interp.shape)
        #
        # for d in args.days:
        #     df_proc['hosp_valid%d' % d] = (df_proc['interval_days'] > df_proc['first_stuh_hg%d' % d]) * (
        #             df_proc['location'] > 1)
        #     df_proc['hosp_check%d' % d] = df_proc.groupby('patient_id')['hosp_valid%d' % d].transform('max')
        #     df_proc['to_adjust%d' % d] = np.where(
        #         (df_proc['count_utoh_nans'] == d) * (df_proc['interval_days'] > df_proc['first_stuh_hg%d' % d])
        #         * (df_proc['interval_days'] < df_proc['last_stuh_hg%d' % d]) * (df_proc['hosp_check%d' % d] == 0) == 1,
        #         1, 0)
        #     df_proc['max_adjust%d' % d] = df_proc.groupby('patient_id')['to_adjust%d' % d].transform('max')
        #     df_proc['max_adjust%d' % d] = df_proc.groupby('patient_id')['to_adjust%d' % d].transform('max')
        #     df_proc['day_adjust%d' % d] = np.where(df_proc['to_adjust%d' % d] == 1, df_proc['interval_days'], np.nan)
        #     df_proc['maxday_adjust%d' % d] = df_proc.groupby('patient_id')['day_adjust%d' % d].transform('max')
        #     df_proc['last_stuh_hg%d_adj' % d] = np.where(df_proc['max_adjust%d' % d] == 1,
        #                                                  df_proc['maxday_adjust%d' % d] - d,
        #                                                  df_proc['last_stuh_hg%d' % d])
        #
        # df_proc.to_csv('hospitalisation_csv')
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
