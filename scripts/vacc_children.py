import os
import time
from datetime import datetime, timedelta
import csv

import numpy as np

from exetera.core.session import Session
import exetera.core.dataset as ds
import exetera.core.dataframe as df
import exetera.core.operations as ops
import exetera.core.fields as fld

list_symptoms = ['abdominal_pain', 'altered_smell', 'blisters_on_feet', 'brain_fog',
                     'chest_pain', 'chills_or_shivers', 'delirium', 'diarrhoea',
                     'diarrhoea_frequency', 'dizzy_light_headed', 'ear_ringing', 'earache',
                     'eye_soreness', 'fatigue', 'feeling_down', 'fever', 'hair_loss',
                     'headache', 'headache_frequency', 'hoarse_voice',
                     'irregular_heartbeat', 'loss_of_smell', 'nausea', 'persistent_cough', 'rash',
                     'red_welts_on_face_or_lips', 'runny_nose',
                     'shortness_of_breath', 'skin_burning', 'skipped_meals', 'sneezing',
                     'sore_throat', 'swollen_glands', 'typical_hayfever', 'unusual_muscle_pains']

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
                if isinstance(df[k], fld.TimestampField):
                    batch.append([get_ts_str(d) for d in df[k].data[current_row:torow]])
                else:
                    batch.append(df[k].data[current_row:torow])
            writer.writerows(list(zip(*batch)))

def get_vacc_in_childern(src_filename, dst_filename, vacc_date):
    """
    patient age 12-17, vaccined after 23.8.2021, proxy reorted
    vaccine_dose: first dose, Pfizer
    symptoms: 8 days after vaccination
    vaccine_symptoms: 8 days after vaccination
    some dates to discade: xx.xx.2021 - xx.xx.2021
    """
    with Session() as s:
        src = s.open_dataset(src_filename, 'r', 'src')
        dst = s.open_dataset(dst_filename, 'w', 'dst')
        src_patients = src['patients']
        filter = (src_patients['age'].data[:] >= 11) & (src_patients['age'].data[:] <= 16)
        filter &=  src_patients['country_code'].data[:] == b'GB'
        filter &= src_patients['reported_by_another'].data[:] == 1
        d_patients = dst.create_dataframe('patients')
        src_patients.apply_filter(filter, ddf=d_patients)
        del d_patients['created_at']
        del d_patients['updated_at']
        print(len(d_patients['id'].data), ' number of children found.')

        p_vacc = dst.create_dataframe('p_vacc')
        df.merge(d_patients, src['vaccine_doses'], dest=p_vacc, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['sequence', 'date_taken_specific', 'brand'])
        filter = p_vacc['date_taken_specific'].data[:] > datetime.strptime(vacc_date, '%Y%m%d').timestamp()  # vaccine date
        filter &= p_vacc['brand'].data[:] == 2
        filter &= p_vacc['sequence'].data[:] == 1
        p_vacc.apply_filter(filter)

        print(len(p_vacc['id'].data), ' number of vaccined children found.')

        p_vacc_lsptm = dst.create_dataframe('p_vacc_lsptm')
        df.merge(p_vacc, src['vaccine_symptoms'], dest=p_vacc_lsptm, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['created_at', 'updated_at', 'pain', 'redness', 'swelling', 'swollen_armpit_glands', 'warmth',
                               'itch', 'tenderness', 'bruising', 'other'])
        filter = (p_vacc_lsptm['date_taken_specific'].data[:] < p_vacc_lsptm['created_at'].data[:]) \
                 & (p_vacc_lsptm['date_taken_specific'].data[:] > p_vacc_lsptm['created_at'].data[:] - 8*24*3600)
        p_vacc_lsptm.apply_filter(filter)

        print(len(p_vacc_lsptm['id'].data), ' number of local symptoms found.')

        p_vacc_ssptm = dst.create_dataframe('p_vacc_ssptm')
        df.merge(p_vacc_lsptm, src['assessments'], dest=p_vacc_ssptm, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['created_at', 'updated_at']+list_symptoms)
        filter = (p_vacc_ssptm['date_taken_specific'].data[:] < p_vacc_ssptm['created_at_r'].data[:]) \
                 & (p_vacc_ssptm['date_taken_specific'].data[:] > p_vacc_ssptm['created_at_r'].data[:] - 8*24*3600)
        p_vacc_ssptm.apply_filter(filter)
        print(len(p_vacc_ssptm['id_l'].data), ' number of systematic symptoms found.')

        save_df_to_csv(p_vacc_ssptm,'vacc_children.csv', list(p_vacc_ssptm.keys()))

if __name__=="__main__":
    srcfile='/nvme0_mounts/nvme0lv01/exetera/recent/ds_20211121_full.hdf5'
    dstfile='vacc_children.hdf5'
    get_vacc_in_childern(srcfile, dstfile, '20210823')

