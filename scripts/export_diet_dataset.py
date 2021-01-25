#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd

from exetera.core import fields, utils
from exetera.core.session import Session
from exeteracovid.algorithms.healthy_diet_index import healthy_diet_index


def export_diet_dataset(s, src_data, geo_data, dest_data, csv_file):

    src_ptnts = src_data['patients']
    src_diet = src_data['diet']
    geo_ptnts = geo_data['patients']

    ffq_questions = ('ffq_chips', 'ffq_crisps_snacks', 'ffq_eggs', 'ffq_fast_food',
                     'ffq_fibre_rich_breakfast', 'ffq_fizzy_pop', 'ffq_fruit',
                     'ffq_fruit_juice', 'ffq_ice_cream', 'ffq_live_probiotic_fermented',
                     'ffq_oily_fish', 'ffq_pasta', 'ffq_pulses', 'ffq_red_meat',
                     'ffq_red_processed_meat', 'ffq_refined_breakfast', 'ffq_rice',
                     'ffq_salad', 'ffq_sweets', 'ffq_vegetables', 'ffq_white_bread',
                     'ffq_white_fish', 'ffq_white_fish_battered_breaded', 'ffq_white_meat',
                     'ffq_white_processed_meat', 'ffq_wholemeal_bread')

    ffq_dict = {k: s.get(src_diet[k]).data[:] for k in ffq_questions}
    scores = healthy_diet_index(ffq_dict)

    p_ids = s.get(src_ptnts['id']).data[:]
    d_pids = s.get(src_diet['patient_id']).data[:]

    g_pids = s.get(geo_ptnts['id']).data[:]

    if not np.array_equal(p_ids, g_pids):
        print("src_data and geo_data do not match")
        exit()

    unique_d_pids = set(d_pids)
    p_filter = np.zeros(len(p_ids), np.bool)
    for i in range(len(p_ids)):
        p_filter[i] = p_ids[i] in unique_d_pids

    patient_fields = ('110_to_220_cm', '15_to_55_bmi', '16_to_90_years', '40_to_200_kg',
                      'a1c_measurement_mmol', 'a1c_measurement_mmol_valid', 'a1c_measurement_percent', 'a1c_measurement_percent_valid',
                      'activity_change', 'age', 'age_filter', 'alcohol_change', 'already_had_covid',
                      'assessment_count', 'blood_group', 'bmi', 'bmi_clean', 'bmi_valid', 'cancer_clinical_trial_site',
                      'cancer_type', 'classic_symptoms', 'classic_symptoms_days_ago', 'classic_symptoms_days_ago_valid',
                      'clinical_study_institutions', 'clinical_study_names', 'clinical_study_nct_ids',
                      'contact_additional_studies', 'contact_health_worker', 'country_code', 'created_at',
                      'created_at_day', 'diabetes_diagnosis_year', 'diabetes_diagnosis_year_valid', 'diabetes_oral_biguanide',
                      'diabetes_oral_dpp4', 'diabetes_oral_meglitinides', 'diabetes_oral_other_medication',
                      'diabetes_oral_sglt2', 'diabetes_oral_sulfonylurea', 'diabetes_oral_thiazolidinediones',
                      'diabetes_treatment_basal_insulin', 'diabetes_treatment_insulin_pump', 'diabetes_treatment_lifestyle',
                      'diabetes_treatment_none', 'diabetes_treatment_other_injection', 'diabetes_treatment_other_oral',
                      'diabetes_treatment_pfnts', 'diabetes_treatment_rapid_insulin', 'diabetes_type',
                      'diabetes_uses_cgm', 'diet_change', 'diet_counts', 'does_chemotherapy', 'ethnicity', 'ever_had_covid_test',
                      'first_assessment_day', 'gender', 'has_asthma', 'has_cancer', 'has_diabetes', 'has_eczema', 'has_hayfever',
                      'has_heart_disease',
                      'has_kidney_disease', 'has_lung_disease', 'has_lung_disease_only',
                      'health_worker_with_contact',
                      'healthcare_professional', 'height_cm', 'height_cm_clean', 'height_cm_valid', 'help_available',
                      'housebound_problems', 'ht_combined_oral_contraceptive_pill', 'ht_depot_injection_or_implant', 'ht_hormone_treatment_therapy',
                      'ht_mirena_or_other_coil', 'ht_none', 'ht_oestrogen_hormone_therapy', 'ht_pfnts', 'ht_progestone_only_pill',
                      'ht_testosterone_hormone_therapy', 'id',
                      'interacted_patients_with_covid',
                      'interacted_with_covid', 'is_carer_for_community',
                      'is_pregnant',
                      'is_smoker',
                      'last_assessment_day', 'lifestyle_version', 'limited_activity',
                      'lsoa11cd',
                      'max_assessment_test_result',
                      'max_test_result', 'mobility_aid',
                      'need_inside_help', 'need_outside_help',
                      'needs_help', 'never_used_shortage', 'on_cancer_clinical_trial',
                      'period_frequency', 'period_status', 'period_stopped_age', 'period_stopped_age_valid', 'pregnant_weeks',
                      'pregnant_weeks_valid',
                      'race_is_other', 'race_is_prefer_not_to_say', 'race_is_uk_asian', 'race_is_uk_black', 'race_is_uk_chinese',
                      'race_is_uk_middle_eastern', 'race_is_uk_mixed_other', 'race_is_uk_mixed_white_black', 'race_is_uk_white',
                      'race_is_us_asian', 'race_is_us_black', 'race_is_us_hawaiian_pacific', 'race_is_us_indian_native', 'race_is_us_white',
                      'race_other', 'reported_by_another', 'same_household_as_reporter', 'se_postcode', 'smoked_years_ago',
                      'smoked_years_ago_valid', 'smoker_status', 'snacking_change', 'sometimes_used_shortage',
                      'still_have_past_symptoms', 'takes_any_blood_pressure_medications', 'takes_aspirin', 'takes_blood_pressure_medications_pril',
                      'takes_blood_pressure_medications_sartan', 'takes_corticosteroids', 'takes_immunosuppressants', 'test_count',
                      'vs_asked_at_set', 'vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3', 'vs_other', 'vs_pftns',
                      'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc', 'weight_change', 'weight_change_kg',
                      'weight_change_kg_valid', 'weight_change_pounds', 'weight_change_pounds_valid', 'weight_kg',
                      'weight_kg_clean', 'weight_kg_valid', 'year_of_birth', 'year_of_birth_valid')
    patient_geo_fields = ('has_imd_data', 'imd_rank', 'imd_decile', 'ruc11cd')

    flt_ptnts = dest_data.create_group('patients')
    for k in patient_fields:
        r = s.get(src_ptnts[k])
        w = r.create_like(flt_ptnts, k)
        s.apply_filter(p_filter, r, w)

    for k in patient_geo_fields:
        r = s.get(geo_ptnts[k])
        w = r.create_like(flt_ptnts, k)
        s.apply_filter(p_filter, r, w)

    p_dict = {'id': s.apply_filter(p_filter, p_ids)}
    for k in flt_ptnts.keys():
        if "weight" in k or "height" in k:
            pkey = "patient_{}".format(k)
        else:
            pkey = k
        p_dict[pkey] = s.get(flt_ptnts[k]).data[:]

    pdf = pd.DataFrame(p_dict)

    d_dict = {'diet_id': s.get(src_diet['id']).data[:],
              'patient_id': s.get(src_diet['patient_id']).data[:]}
    d_dict.update({
        k: s.get(src_diet[k]).data[:] for k in src_diet.keys() if k not in ('id', 'patient_id')
    })
    d_dict.update({'scores': scores})
    ddf = pd.DataFrame(d_dict)

    tdf = pd.merge(left=ddf, right=pdf, left_on='patient_id', right_on='id')
    for k in tdf.keys():
        if k[-2:] == "_x" or k[-2:] == "_y":
            print(k)

    print(tdf)
    tdf.to_csv(csv_file, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='The dataset containing the patient and diet data')
    parser.add_argument('-g', '--geodata', required=True, help='The dataset containing patient-level geocode data')
    parser.add_argument('-o', '--output', required=True, help='The output dataset that contains the output data')
    parser.add_argument('-c', '--csvoutput', required=True, help='The csv file that contains the output data')
    args = parser.parse_args()
    with Session() as s:
        src = s.open_dataset(args.source, 'r', 'src')
        geo = s.open_dataset(args.geodata, 'r', 'geo')
        output = s.open_dataset(args.output, 'w', 'output')
        export_diet_dataset(s, src, geo, output, args.csvoutput)
