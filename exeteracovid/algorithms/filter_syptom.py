import numpy as np

def at_least_one_symptom(assessment_df):
    """
    Filter the rows with at least one symptom reported.

    :param assessment_df: The assessment dataframe.
    :returns: A numpy array of bool marking matched rows.
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
            filter |= assessment_df[symptom].data[:] > 1  # has symptom

    return filter


def nhs_symptom(assessment_df):
    """
    Filter the rows with symptoms match with NHS announced main Covid symptoms: high temperature, cough and loss of smell.

    :param assessment_df: The assessment dataframe.
    :returns: A numpy array of bool marking matched rows.
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

def filter_symptoms(assessment_df, symp_dict):
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
