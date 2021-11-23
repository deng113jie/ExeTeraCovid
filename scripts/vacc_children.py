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

patient_fields = ['id', 'age', 'country_code', 'bmi', 'ethnicity', 'gender',
                  'is_pregnant', 'is_smoker', 'lsoa11cd', 'reported_by_another',
                  'has_asthma',   'has_eczema','has_hayfever',
                  'has_cancer', 'has_diabetes', 'has_heart_disease', 'has_kidney_disease', 'has_lung_disease', 'does_chemotherapy', 'takes_immunosuppressants']

def get_ts_str(d):
    if d > 0:
        return datetime.fromtimestamp(d).strftime("%Y-%m-%d")
    else:
        return '0'

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

def get_vacc_in_childern_dup(src_filename, dst_filename, vacc_date):
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
        print(datetime.now(), len(np.unique(d_patients['id'].data[:])), ' number of unique children found.')

        p_vacc = dst.create_dataframe('p_vacc')
        df.merge(d_patients, src['vaccine_doses'], dest=p_vacc, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['sequence', 'date_taken_specific', 'brand'])
        filter = p_vacc['date_taken_specific'].data[:] > datetime.strptime(vacc_date, '%Y%m%d').timestamp()  # vaccine date
        filter &= p_vacc['brand'].data[:] == 2
        filter &= p_vacc['sequence'].data[:] == 1
        p_vacc.apply_filter(filter)

        print(datetime.now(), len(np.unique(p_vacc['id'].data)), ' vaccined children found.')

        p_vacc_lsptm = dst.create_dataframe('p_vacc_lsptm')
        df.merge(p_vacc, src['vaccine_symptoms'], dest=p_vacc_lsptm, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['created_at', 'updated_at', 'pain', 'redness', 'swelling', 'swollen_armpit_glands', 'warmth',
                               'itch', 'tenderness', 'bruising', 'other'])
        filter = (p_vacc_lsptm['date_taken_specific'].data[:] < p_vacc_lsptm['created_at'].data[:]) \
                 & (p_vacc_lsptm['date_taken_specific'].data[:] > p_vacc_lsptm['created_at'].data[:] - 8*24*3600)
        p_vacc_lsptm.apply_filter(filter)
        print(datetime.now(), len(np.unique(p_vacc_lsptm['id'].data)), ' children with ',
              len(p_vacc_lsptm['id'].data), ' local records found.')

        p_vacc_ssptm = dst.create_dataframe('p_vacc_ssptm')
        df.merge(p_vacc_lsptm, src['assessments'], dest=p_vacc_ssptm, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['created_at', 'updated_at']+list_symptoms)
        filter = (p_vacc_ssptm['date_taken_specific'].data[:] < p_vacc_ssptm['created_at_r'].data[:]) \
                 & (p_vacc_ssptm['date_taken_specific'].data[:] > p_vacc_ssptm['created_at_r'].data[:] - 8*24*3600)
        p_vacc_ssptm.apply_filter(filter)
        print(datetime.now(), len(np.unique(p_vacc_ssptm['id'].data)), ' children with ',
              len(p_vacc_ssptm['id'].data), ' systemic records found.')

        #output to csv
        save_df_to_csv(p_vacc_ssptm,'vacc_children.csv', list(p_vacc_ssptm.keys()))


def get_vacc_in_childern_uniq(src_filename, dst_filename, vacc_date):
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
        d_patients = dst.create_dataframe('patients')
        for f in patient_fields:
            df.copy(src_patients[f], d_patients, f)
        filter = (d_patients['age'].data[:] >= 11) & (d_patients['age'].data[:] <= 16)
        filter &=  d_patients['country_code'].data[:] == b'GB'
        filter &= d_patients['reported_by_another'].data[:] == 1
        d_patients.apply_filter(filter)
        print(datetime.now(), len(np.unique(d_patients['id'].data[:])), ' number of unique children found.')

        #nr_comorbidities
        nr_comorbidities = np.zeros(len(d_patients['has_diabetes'].data))
        for k in ['has_diabetes', 'has_heart_disease', 'has_lung_disease', 'does_chemotherapy', 'has_kidney_disease',
                  'has_cancer', 'takes_immunosuppressants']:
            nr_comorbidities += np.where(d_patients[k].data[:] == 2, 1, 0)
        d_patients.create_numeric('nr_comorbidities', 'int32')
        d_patients['nr_comorbidities'].data.write(nr_comorbidities)

        has_comorbidities = np.where(nr_comorbidities == 0, 0, 1)
        d_patients.create_numeric('has_comorbidities', 'int8').data.write(has_comorbidities)

        #vaccine
        p_vacc = dst.create_dataframe('p_vacc')
        df.merge(d_patients, src['vaccine_doses'], dest=p_vacc, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['id', 'sequence', 'date_taken_specific', 'brand'])
        filter = p_vacc['date_taken_specific'].data[:] > datetime.strptime(vacc_date, '%Y%m%d').timestamp()  # vaccine date
        filter &= p_vacc['brand'].data[:] == 2
        filter &= p_vacc['sequence'].data[:] == 1
        p_vacc.apply_filter(filter)
        df.move(p_vacc['id_l'], p_vacc, 'id')
        df.move(p_vacc['id_r'], p_vacc, 'vaccine_id')
        print(datetime.now(), len(np.unique(p_vacc['id'].data)), ' children with ',
              len(np.unique(p_vacc['vaccine_id'].data[:])), ' vaccine records found.')

        #local symptom
        p_vacc_lsptm = dst.create_dataframe('p_vacc_lsptm')
        df.merge(p_vacc, src['vaccine_symptoms'], dest=p_vacc_lsptm, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['id', 'created_at', 'updated_at', 'pain', 'redness', 'swelling', 'swollen_armpit_glands', 'warmth',
                               'itch', 'tenderness', 'bruising', 'other'])
        filter = (p_vacc_lsptm['date_taken_specific'].data[:] < p_vacc_lsptm['created_at'].data[:]) \
                 & (p_vacc_lsptm['date_taken_specific'].data[:] > p_vacc_lsptm['created_at'].data[:] - 8*24*3600)
        p_vacc_lsptm.apply_filter(filter)
        df.move(p_vacc_lsptm['id_l'], p_vacc_lsptm, 'id')
        df.move(p_vacc_lsptm['id_r'], p_vacc_lsptm, 'lsymptom_id')
        print(datetime.now(), len(np.unique(p_vacc_lsptm['id'].data)), ' children with ',
              len(np.unique(p_vacc_lsptm['lsymptom_id'].data[:])), ' local symptoms records found.')

        #one local symptom a day
        lsymp_doy = [datetime.fromtimestamp(i).timetuple().tm_yday for i in p_vacc_lsptm['created_at'].data[:]]
        p_vacc_lsptm.create_numeric('lsymp_doy', 'int32').data.write(lsymp_doy)
        p_vacc_lsptm.sort_values(by=['id', 'lsymp_doy'])

        lsymptom_max = np.zeros(len(p_vacc_lsptm['pain'].data), np.int16)
        for f in ['pain', 'redness', 'swelling', 'swollen_armpit_glands', 'warmth', 'itch', 'tenderness', 'bruising']:
            lsymptom_max+=p_vacc_lsptm[f].data[:]
        p_vacc_lsptm.create_numeric('lsymp_max', 'int32').data.write(lsymptom_max)

        filter = np.zeros(len(p_vacc_lsptm['pain'].data), bool)
        span_data = np.asarray([p_vacc_lsptm[k].data[:] for k in ['id', 'lsymp_doy']])
        spans = ops._get_spans_for_multi_fields(span_data)
        for i in range(len(spans)-1):
            max_record = np.argmax(p_vacc_lsptm['lsymp_max'].data[spans[i]:spans[i+1]])
            filter[spans[i]+max_record] = True
        p_vacc_lsptm.apply_filter(filter)
        print(datetime.now(), len(np.unique(p_vacc_lsptm['id'].data)), ' children with ',
              len(np.unique(p_vacc_lsptm['lsymptom_id'].data[:])), ' local symptoms records left.')

        # systematic symptoms
        p_vacc_ssptm = dst.create_dataframe('p_vacc_ssptm')
        df.merge(p_vacc_lsptm, src['assessments'], dest=p_vacc_ssptm, how='inner', left_on='id', right_on='patient_id',
                 right_fields=['id', 'created_at', 'updated_at']+list_symptoms)
        filter = (p_vacc_ssptm['date_taken_specific'].data[:] < p_vacc_ssptm['created_at_r'].data[:]) \
                 & (p_vacc_ssptm['date_taken_specific'].data[:] > p_vacc_ssptm['created_at_r'].data[:] - 8*24*3600)
        p_vacc_ssptm.apply_filter(filter)
        df.move(p_vacc_ssptm['id_l'],p_vacc_ssptm, 'id')
        df.move(p_vacc_ssptm['id_r'], p_vacc_ssptm, 'ssymptom_id')
        print(datetime.now(), len(np.unique(p_vacc_ssptm['id'].data)), ' children with ',
              len(np.unique(p_vacc_ssptm['lsymptom_id'].data[:])), ' systematic symptoms records found.')

        #filter one symptom per day
        ssymp_doy = [datetime.fromtimestamp(i).timetuple().tm_yday for i in p_vacc_ssptm['created_at_r'].data[:]]
        p_vacc_ssptm.create_numeric('ssymp_doy', 'int32').data.write(ssymp_doy)
        p_vacc_ssptm.sort_values(by=['id', 'ssymp_doy'])

        ssymptom_max = np.zeros(len(p_vacc_ssptm['abdominal_pain'].data), np.int16)
        for f in  list_symptoms:
            ssymptom_max += np.where(p_vacc_ssptm[f].data[:]>1,1,0)
        p_vacc_ssptm.create_numeric('ssymp_max', 'int32').data.write(ssymptom_max)

        filter = np.zeros(len(p_vacc_ssptm['abdominal_pain'].data), bool)
        span_data = np.asarray([p_vacc_ssptm[k].data[:] for k in ['id', 'ssymp_doy']])
        spans = ops._get_spans_for_multi_fields(span_data)
        for i in range(len(spans) - 1):
            max_record = np.argmax(p_vacc_ssptm['ssymp_max'].data[spans[i]:spans[i + 1]])
            filter[spans[i] + max_record] = True
        p_vacc_ssptm.apply_filter(filter)
        print(datetime.now(), len(np.unique(p_vacc_ssptm['id'].data)), ' children with ',
              len(np.unique(p_vacc_ssptm['ssymptom_id'].data[:])), ' systematic symptoms records left.')

        #output to csv
        save_df_to_csv(p_vacc_ssptm,'vacc_children.csv', list(p_vacc_ssptm.keys()))

if __name__=="__main__":
    srcfile='/nvme0_mounts/nvme0lv01/exetera/recent/ds_20211121_full.hdf5'
    dstfile='vacc_children.hdf5'
    get_vacc_in_childern_uniq(srcfile, dstfile, '20210823')

