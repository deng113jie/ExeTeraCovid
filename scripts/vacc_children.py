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
        print(len(d_patients['id'].data), ' number of children found.')

        p_vacc = dst.create_dataframe('p_vacc')
        df.merge(d_patients, src['vaccine_doses'], dest=p_vacc, how='inner', left_on='id', right_on='patient_id')
        filter = p_vacc['created_at_r'].data[:] > datetime.strptime(vacc_date, '%Y%m%d').timestamp()  # vaccine date
        filter &= p_vacc['brand'].data[:] == 2
        filter &= p_vacc['sequence'].data[:] == 1
        p_vacc.apply_filter(filter)
        p_vacc['created_at_r'] = p_vacc['vaccine_date']
        print(len(p_vacc['id'].data), ' number of vaccined children found.')

        p_vacc_lsptm = dst.create_dataframe('p_vacc_lsptm')
        df.merge(p_vacc, src['vaccine_symptoms'], dest=p_vacc_lsptm, how='inner', left_on='id', right_on='patient_id')
        filter = (p_vacc_lsptm['vaccine_date'].data[:] < p_vacc_lsptm['created_at'].data[:]) \
                 & (p_vacc_lsptm['vaccine_date'].data[:] > p_vacc_lsptm['created_at'].data[:] - 8*24*3600)
        p_vacc_lsptm.apply_filter(filter)
        p_vacc_lsptm['created_at'] = p_vacc_lsptm['lsymptom_date']
        print(len(p_vacc_lsptm['id'].data), ' number of local symptoms found.')

        p_vacc_ssptm = dst.create_dataframe('p_vacc_ssptm')
        df.merge(p_vacc_lsptm, src['assessments'], dest=p_vacc_ssptm, how='inner', left_on='id', right_on='patient_id')
        filter = (p_vacc_ssptm['vaccine_date'].data[:] < p_vacc_ssptm['created_at'].data[:]) \
                 & (p_vacc_ssptm['vaccine_date'].data[:] > p_vacc_ssptm['created_at'].data[:] - 8*24*3600)
        p_vacc_ssptm.apply_filter(filter)
        p_vacc_ssptm['created_at'] = p_vacc_ssptm['ssymptom_date']
        print(len(p_vacc_ssptm['id'].data), ' number of systematic symptoms found.')









if __name__=="__main__":
    srcfile='/nvme0_mounts/nvme0lv01/exetera/recent/ds_20211121_full.hdf5'
    dstfile='vacc_children.hdf5'
    get_vacc_in_childern(srcfile, dstfile, 20210823)

