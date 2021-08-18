import numpy as np

import exetera.core.persistence as prst

def at_least_one_symptom_v1(assessment_df):
    """
    Filter the rows with at least one symptom reported.

    :param assessment_df: The assessment dataframe.
    :return: A numpy array of bool marking matched rows.
    """
    list_symptoms = ['fatigue', 'abdominal_pain', 'chest_pain', 'sore_throat', 'shortness_of_breath',
                     'skipped_meals', 'loss_of_smell', 'unusual_muscle_pains', 'headache', 'hoarse_voice', 'delirium',
                     'diarrhoea',
                     'fever', 'persistent_cough', 'dizzy_light_headed', 'eye_soreness', 'red_welts_on_face_or_lips',
                     'blisters_on_feet']
    if len(assessment_df.keys()) == 0:
        raise ValueError("The assessment dataframe contains no data.")
    filter = np.zeros(len(assessment_df[list(assessment_df.keys())[0]]), dtype=bool)
    for symptom in list_symptoms:
        if symptom not in assessment_df.keys():
            raise ValueError("Field ", symptom, " not in the assessment dataframe.")
        else:
            if symptom == 'fatigue' or symptom == 'shortness_of_breath':
                filter |= assessment_df[symptom].data[:] > 2  # has symptom
            else:
                filter |= assessment_df[symptom].data[:] > 1  # has symptom

    return filter

def sum_up_symptons_v1(asmt_df):
    sum_symp = np.zeros(len(asmt_df['patient_id'].data))
    list_symptoms = ['fatigue', 'abdominal_pain', 'chest_pain', 'sore_throat', 'shortness_of_breath',
                     'skipped_meals', 'loss_of_smell', 'unusual_muscle_pains', 'headache', 'hoarse_voice', 'delirium',
                     'diarrhoea',
                     'fever', 'persistent_cough', 'dizzy_light_headed', 'eye_soreness', 'red_welts_on_face_or_lips',
                     'blisters_on_feet']
    for k in list_symptoms:
        values = asmt_df[k].data[:]
        if k == 'fatigue' or k == 'shortness_of_breath':
            values = np.where(values > 2, np.ones_like(values), np.zeros_like(values))
        else:
            values = np.where(values > 1, np.ones_like(values), np.zeros_like(values))
        sum_symp += values

    return sum_symp

def nhs_symptom_v1(assessment_df):
    """
    Filter the rows with symptoms match with NHS announced main Covid symptoms: high temperature, cough and loss of smell.

    :param assessment_df: The assessment dataframe.
    :return: A numpy array of bool marking matched rows.
    """
    list_symptoms = ['loss_of_smell', 'fever', 'persistent_cough']
    if len(assessment_df.keys()) == 0:
        raise ValueError("The assessment dataframe contains no data.")
    filter = np.zeros(len(assessment_df[list(assessment_df.keys())[0]]), dtype=bool)
    for symptom in list_symptoms:
        if symptom not in assessment_df.keys():
            raise ValueError("Field ", symptom, " not in the assessment dataframe.")
        else:
            filter |= assessment_df[symptom].data[:] > 1  # has symptom

    return filter

def filter_symptoms_v1(assessment_df, symp_dict):
    """
    Filter the rows in assessment dataframe based on given symptom values.

    :param assessment_df: The assessment dataframe.
    :param symp_dict: A dictionary of symptoms to filter, with the symptom name as key and bool or integer as value.
    :return: A numpy array of bool marking rows matching symp_dict.
    """
    if len(assessment_df.keys()) == 0:
        raise ValueError("The assessment dataframe contains no data.")
    filter = np.zeros(len(assessment_df[list(assessment_df.keys())[0]]), dtype=bool)
    for symptom in symp_dict.keys():
        if symptom not in assessment_df.keys():
            raise ValueError("Field ", symptom, " not in the assessment dataframe.")
        else:
            if isinstance(symp_dict[symptom], list):  # of list of integers
                filter |= np.isin(assessment_df[symptom].data[:], symp_dict[symptom])
            elif isinstance(symp_dict[symptom], int):  # integer
                filter |= assessment_df[symptom].data[:] == symp_dict[symptom]
            elif isinstance(symp_dict[symptom], bool):  # boolean
                if symp_dict[symptom] is True:
                    filter |= assessment_df[symptom].data[:] > 1
                else:
                    filter |= assessment_df[symptom].data[:] <= 1
    return filter

def filter_asymp_and_firstnz_v1(s, out_pos):
    """
    Filter symptom and return two index: 1) the patients without any symptom recorded; and 2) the patients have a first
    full-zero symptom and then recorded some symptom.

    :param s: The exetera Session instance.
    :param out_post: The dataframe contains sum_symp, patient_id fields.
    :return: spans_asymp -- the patients without any symptom;
             filt_sel --  patients with a first full-zero and then some symptom.
    """
    symp_flat = np.where(out_pos['sum_symp'].data[:] < 1, 0, 1)
    spans = out_pos['patient_id'].get_spans()
    print('Number definitie positive is', len(spans) - 1)

    # Get the first index at which the hct field is maximum
    firstnz_symp_ind = s.apply_spans_index_of_max(spans, symp_flat)
    max_symp_check = symp_flat[firstnz_symp_ind]
    # Get the index of first element of patient_id when sorted

    filt_asymptomatic = max_symp_check == 0
    print('Number asymptomatic is ', len(spans) - 1 - np.sum(max_symp_check), np.sum(filt_asymptomatic))

    first_symp_ind = spans[:-1]
    not_healthy_first = first_symp_ind != firstnz_symp_ind
    print('Number not healthy first is ', len(spans) - 1 - np.sum(not_healthy_first))

    spans_valid = s.apply_filter(not_healthy_first, first_symp_ind)
    #pat_sel = s.apply_indices(spans_valid, out_pos['patient_id'].data[:])
    pat_sel = out_pos['patient_id'].apply_index(spans_valid)
    filt_sel = prst.foreign_key_is_in_primary_key(pat_sel.data[:], out_pos['patient_id'].data[:])

    spans_asymp = s.apply_filter(filt_asymptomatic, first_symp_ind)
    return spans_asymp, filt_sel
