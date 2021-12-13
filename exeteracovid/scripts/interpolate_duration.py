import math
from datetime import datetime, timedelta
import argparse
import sys

from numba import njit
import pandas as pd
import numpy as np


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

    full_idx = np.arange(df_test_comb['interval_days'].data[:].min(), df_test_comb['interval_days'].data[:].max()+1)

    # for f in list_symptoms + ['fatigue', 'shortness_of_breath']:
    #     df_test_comb[f] = df_test_comb[f].fillna(0)
    #df_test_comb_ind = df_test_comb.set_index('interval_days')

    # df_test2 = df_test_comb_ind.groupby('patient_id', as_index=False).apply(lambda group: group.reindex(full_idx)).reset_index(level=0, drop=True).sort_index()
    expanded_df = TEMPH5.create_dataframe('expanded_df')
    df_test_comb['patient_id'].create_like(expanded_df, 'patient_id')
    df_test_comb['interval_days'].create_like(expanded_df, 'interval_days')
    df_test_comb['created_at'].create_like(expanded_df, 'created_at')
    for f in list_symptoms:
        expanded_df.create_numeric(f,'float')

    spans = df_test_comb['patient_id'].get_spans()
    for i in range(len(spans)-1):
        if (i%1000==0): print(datetime.now(), i, ' no. of interpolation processed.')
        #write patient_id, idx
        pid = df_test_comb['patient_id'].data[spans[i]]
        expanded_df['patient_id'].data.write([pid for _ in range(len(full_idx))])
        expanded_df['interval_days'].data.write(full_idx)
        idx = df_test_comb['interval_days'].data[spans[i]:spans[i+1]]
        # todo perform filter on gap of assessments
        # todo use np (memory array) to reduce io and speed up
        for f in list_symptoms:
            data = df_test_comb[f].data[spans[i]:spans[i+1]]
            # pdseries = pd.Series(data, index=idx)
            # pdseries = pdseries.reindex(full_idx)
            # pdseries = pdseries.interpolate(method='linear', limit_area='inside')
            # expanded_df[f].data.write(pdseries.values)
            result = single_interpolate(idx, data, full_idx)
            expanded_df[f].data.write(result)
        created_interp = single_interpolate(idx, df_test_comb['created_at'].data[spans[i]:spans[i+1]], full_idx)
        expanded_df['created_at'].data.write(created_interp)
    #drop na, rename axis, drop column, reset_index
    filter = np.isnan(expanded_df['fatigue'].data[:])
    for f in list_symptoms:
        filter &= np.isnan(expanded_df[f].data[:])
    expanded_df.apply_filter(filter)
    return expanded_df


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
    #todo confirm first_entry_sum, last_entry_sum, interval_days not used?

    print('Performing interpolation')
    df_interp = interpolate_date(df_init)
    #df_interp['created_interp'] = df_interp['created_at'].interpolate('linear') todo confirm created_at should be local per patient

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
        #todo confirm re-cal sum_symp rename to sum_symp_interpo
        sum_symp = np.zeros(len(df_train['patient_id'].data), 'int16')
        for k in list_symptoms:
            sum_symp+=df_interp[k].data[:]
        df_interp.create_numeric('df_interp', 'int16').data.write(sum_symp)

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

def treat_interp(df_interp):
    df_interp['count_values'] = df_interp.groupby('patient_id')['interval_days'].transform('count')
    df_interp['length_log'] = df_interp.groupby('patient_id')['interval_days'].transform('max')
    for d in [7,14]:
        df_interp['duration%d'%d] = df_interp['last_stuh_hg%d'%d] - df_interp['first_stuh_hg%d'%d] + 1
        df_interp['last_status%d'%d] = np.where(df_interp['last_stuh_hg%d'%d]==df_interp['interval_days'],df_interp['health_interp'],np.nan)
        df_interp['last_status%d'%d] = df_interp.groupby('patient_id')['last_status%d'%d].transform('max')
        df_interp['duration%dnew'%d] = np.where(df_interp['last_status%d'%d]==1,df_interp['duration%d'%d]-1,df_interp['duration%d'%d])
    for s in list_symptoms_red:
        if 'day7_'+s in df_interp.columns:
            df_interp['end7_'+s] = df_interp.groupby('patient_id')['day7_'+s].transform('max')
    df_interp['date_firstuh7'] = np.where(df_interp['interval_days']==df_interp['first_stuh_hg7'],df_interp['created_at'],np.nan)
    df_interp['date_firstuh7'] = df_interp.groupby('patient_id')['date_firstuh7'].transform('min')

    df_interp['date_firstuh14'] = np.where(df_interp['interval_days']==df_interp['first_stuh_hg14'],df_interp['created_at'],np.nan)
    df_interp['date_firstuh14'] = df_interp.groupby('patient_id')['date_firstuh14'].transform('min')

    df_interp['date_lastuh7'] = np.where(df_interp['interval_days']>=df_interp['last_stuh_hg7'],df_interp['created_at'],np.nan)
    df_interp['date_lastuh7'] = df_interp.groupby('patient_id')['date_lastuh7'].transform('min')
    df_interp['date_lastuh7'] = np.where(df_interp['last_status7']==1,df_interp['date_lastuh7']-86400,df_interp['date_lastuh7'])
    df_interp['date_lastuh14'] = np.where(df_interp['interval_days']>=df_interp['last_stuh_hg14'],df_interp['created_at'],np.nan)
    df_interp['date_lastuh14'] = df_interp.groupby('patient_id')['date_lastuh14'].transform('min')
    df_interp['date_lastuh14'] = np.where(df_interp['last_status14']==1,df_interp['date_lastuh14']-86400,df_interp['date_lastuh14'])
    df_interp['date_sick'] = np.where(df_interp['health_interp']==0, df_interp['created_at'],np.nan)
    df_interp['first_sick'] = df_interp.groupby('patient_id')['date_sick'].transform('min')
    df_interp['last_sick'] = df_interp.groupby('patient_id')['date_sick'].transform('max')
    df_interp['last_log'] = df_interp.groupby('patient_id')['created_at'].transform('max')
    df_interp['first_log'] = df_interp.groupby('patient_id')['created_at'].transform('min')
    df_interp['h_extent'] = np.where(df_interp['created_at']>=df_interp['first_sick'],df_interp['health_interp'],0)
    df_interp['h_extent'] = np.where(df_interp['created_at']<=df_interp['last_sick'],df_interp['h_extent'],0)
    df_interp['count_h_extent'] = df_interp.groupby('patient_id')['h_extent'].transform('sum')
    df_interp['last_afs7'] = np.where(np.logical_and(df_interp['created_at']>df_interp['first_sick'], df_interp['count_healthy']>=7),df_interp['created_at']-86400*df_interp['count_healthy'],np.nan)
    df_interp['last_afs14'] = np.where(np.logical_and(df_interp['created_at']>df_interp['first_sick'], df_interp['count_healthy']>=14),df_interp['created_at']-86400*df_interp['count_healthy'],np.nan)
    df_interp['last_afs7'] = np.where(df_interp['created_at']>=df_interp['last_sick'],df_interp['last_sick'],df_interp['last_afs7'])
    df_interp['last_afs14'] = np.where(df_interp['created_at']>=df_interp['last_sick'],df_interp['last_sick'],df_interp['last_afs14'])
    df_interp['prop_h'] = df_interp['count_healthy'].replace(0,np.nan)
    df_interp['prop_h'] = df_interp.groupby('patient_id')['prop_h'].ffill()
    df_interp['first_s'] = np.where(df_interp['count_nothealthy']==1,df_interp['created_at'],np.nan)
    df_interp['first_s7'] = np.where(np.logical_and(df_interp['first_s']>df_interp['first_sick'], df_interp['prop_h']<7),np.nan, df_interp['first_s'])
    df_interp['first_s14'] = np.where(np.logical_and(df_interp['first_s']>df_interp['first_sick'], df_interp['prop_h']<14),np.nan, df_interp['first_s'])

    df_interp['last_afs7_f'] = df_interp.groupby('patient_id')['last_afs7'].bfill()
    df_interp['last_afs14_f'] = df_interp.groupby('patient_id')['last_afs14'].bfill()
    df_interp['first_s7_f'] = df_interp.groupby('patient_id')['first_s7'].ffill()
    df_interp['first_s14_f'] = df_interp.groupby('patient_id')['first_s14'].ffill()
    df_interp['dur7_art'] = (df_interp['last_afs7_f'] - df_interp['first_s7_f'])/86400 +1
    df_interp['dur14_art'] = (df_interp['last_afs14_f'] - df_interp['first_s14_f'])/86400 +1

    df_interp['dur7_artm'] = df_interp.groupby('patient_id')['dur7_art'].transform('max')
    df_interp['dur14_artm'] = df_interp.groupby('patient_id')['dur14_art'].transform('max')
    df_interp['first_sicki'] = np.where(df_interp['created_at']==df_interp['first_sick'],df_interp['interval_days'],np.nan)
    df_interp['last_sicki'] = np.where(df_interp['created_at']==df_interp['last_sick'],df_interp['interval_days'],np.nan)
    df_interp['first_sicki'] = df_interp.groupby('patient_id')['first_sicki'].transform('min')
    df_interp['last_sicki'] = df_interp.groupby('patient_id')['last_sicki'].transform('min')
    print('creating nan values')
    df_interp['max_nans'] = df_interp.groupby('patient_id')['count_nans'].transform('max')
    df_interp['max_nans_fl7'] = np.where(np.logical_and(df_interp['interval_days']>=df_interp['first_stuh_hg7'],
                                                        df_interp['interval_days']<=df_interp['last_stuh_hg7']), df_interp['count_nans'],0)

    df_interp['max_nans_fl14'] = np.where(np.logical_and(df_interp['interval_days']>=df_interp['first_stuh_hg14'],
                                                        df_interp['interval_days']<=df_interp['last_stuh_hg14']), df_interp['count_nans'],0)

    df_interp['max_nans_fl14'] = df_interp.groupby('patient_id')['max_nans_fl14'].transform('max')
    df_interp['max_nans_fl7'] = df_interp.groupby('patient_id')['max_nans_fl7'].transform('max')
    df_interp['max_symp'] = df_interp.groupby('patient_id')['sum_symp'].transform('max')
    df_interp['max_countuh']  = df_interp.groupby('patient_id')['count_nothealthy'].transform('max')
    df_interp['day_test'] = np.where(np.logical_and(df_interp['created_interp']<=df_interp['date_effective_test']+86400,
                                               df_interp['created_interp']<=df_interp['date_effective_test']-86400), df_interp['interval_days'],np.nan)
    df_interp['before_test'] = np.where(df_interp['created_interp']<=df_interp['date_effective_test'],1,0)
    df_interp['count_uhbef'] = df_interp['count_nothealthy'] * df_interp['before_test']
    df_interp['count_nanbef'] = df_interp['count_nans'] * df_interp['before_test']
    df_interp['max_countnanbef'] =  df_interp.groupby('patient_id')['count_nanbef'].transform('max')
    df_interp['max_countuhbef'] = df_interp.groupby('patient_id')['count_uhbef'].transform('max')
    df_interp['created_aroundtest'] = np.where(np.logical_and(df_interp['created_at']<=df_interp['date_effective_test']+8*86400,
                                                         df_interp['created_at']>=df_interp['date_effective_test']-14*86400),1,0)
    df_interp['sum_symp_at'] = df_interp['sum_symp'] * df_interp['created_aroundtest']
    df_interp['max_symp_at'] = df_interp.groupby('patient_id')['sum_symp_at'].transform('max')
    df_interp['numb_at'] = df_interp.groupby('patient_id')['created_aroundtest'].transform('sum')
    return df_interp


def determine_meeting_criteria(df_interp, days=[7]):
    from datetime import datetime, timedelta

    timemin = timedelta(days=-7)
    import time

    struct_time = datetime.strptime("20 Jul 20", "%d %b %y").date()
    df_interp['delay'] = pd.to_datetime(df_interp['date_update'], infer_datetime_format=True).dt.date - struct_time
    delay = (df_interp['date_update'] - datetime.strptime("20 Jul 20", "%d %b %y").timestamp())/86400
    #df_interp['dropped'] = np.where(df_interp['delay'] <= timemin, 1, 0)
    # todo confirm last_entry_sum should be there already - refresh after interpolation
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
