import numpy as np

"""
An example to infer the positiveness based on bio information and symptoms using logistic regression.
"""

dict_coef = {
    'intercept': np.float64(-1.32167356),
    'persistent_cough': np.float64(0.30872189),
    'skipped_meals': np.float64(0.38691672),
    'loss_of_smell': np.float64(1.74711853),
    'gender': np.float64(0.43576578),
    'age': np.float64(-0.00716455),
    'fatigue': np.float64(0.49534113)
}
dict_sd = {
    'intercept': np.float64(0.080211442),
    'persistent_cough': np.float64(0.041716425),
    'skipped_meals': np.float64(0.046133888),
    'loss_of_smell': np.float64(0.042320229),
    'gender': np.float64(0.047851708),
    'age': np.float64(0.001729339),
    'fatigue': np.float64(0.052359487)
}


def logit_prediction(df, dict_coef, index):
    """
    Compute the logged index value using bio information and symptoms.

    :param df: The dataset contains bio information and symptoms.
    :param dict_coef: The model coefficient parameter.
    :param index: The index of patient to calculate the value.
    :return: The value computed.
    """
    temp_sum = np.float64(0)
    for k in dict_coef.keys():
        if k != 'intercept':
            data = df[k]
            # print(f"df[{k}] range: min={df[k].min()}, max={df[k].max()}")
            # print(f"type of dict_coef item: {type(dict_coef[k][index])}, {dict_coef[k][index]}")
            temp_sum += data * dict_coef[k][index]
    temp_sum += dict_coef['intercept'][index]
    # print(f"temp_sum type: {type(temp_sum)}")
    #temp_logit = 1.0/(1+np.exp(-1.0*temp_sum.astype(float)))
    # print(f"temp_sum min={temp_sum.min()}, max={temp_sum.max()}")
    temp_logit = 1.0 / (1 + np.exp(-1.0 * temp_sum))
    return temp_logit


def imputation_pos_all(df_umerge, distr, samplecount, threshold=None):
    """
    Infer the positiveness.

    :param df_umerge: The dataset contains bio information and symptoms.
    :param distr: The distribution parameters.
    :param samplecount: Number of samples in this batch.
    :param threshold: The threshold used for making decision.
    :return: The result of likelihood of positiveness for each patient.
    """
    # df_umerge = pd.merge(df_assess, df_patred, left_on='patient_id', right_on='id', how='left')
    # df_umerge = df_umerge.drop_duplicates(['patient_id', 'interval_days']) #not needed due to daily assessments
    list_temp = []

    df_umerge = {k: v[:] for k, v in df_umerge.items()}

    results = logit_prediction(df_umerge, distr, 0)
    for i in range(1, samplecount):
        print(f"  imputation iteration {i}")
        results += logit_prediction(df_umerge, distr, i)
    average_decision = results / np.float64(samplecount)
    # average_decision = np.mean(np.asarray(list_temp).astype(float), 0)
    # this could be a threshold of
    # print(average_decision.shape, len(average_decision > threshold), df_umerge.shape, df_assess.shape)
    # df_assess['imputed'] = np.where(average_decision > threshold, 1, 0) # this goes to a writer
    # return df_assess
    if threshold is not None:
        return np.where(average_decision > threshold, np.uint8(1), np.uint8(0))
    else:
        return average_decision


def nature_medicine_model_1(persistent_cough, skipped_meals, loss_of_smell, fatigue,
                            age_in_assessment_space, gender_in_assessment_space,
                            threshold=None):
    """
    Infer the positiveness based on bio information and symptoms using logistic regression.

    :param persistent_cough: The 'persistent_cough' column from assessments dataframe.
    :param skipped_meals: The 'skipped_meals' column from assessments dataframe.
    :param loss_of_smell: The 'loss_of_smell' column from assessments dataframe.
    :param fatigue: The 'fatigue' column from assessments dataframe.
    :param age_in_assessment_space: The 'age' column of each patient.
    :param gender_in_assessment_space: The 'gender' column of each patient.
    :param threshold: The threshold used for making decision.
    :return: The result of likelihood of positiveness for each patient.
    """
    dataset = {
        'persistent_cough': persistent_cough,
        'skipped_meals': skipped_meals,
        'loss_of_smell': loss_of_smell,
        'fatigue': fatigue,
        'age': age_in_assessment_space,
        'gender': gender_in_assessment_space
    }

    samplecount = 100
    distr = {}
    for k in dict_coef.keys():
        distr[k] = dict_coef[k] + dict_sd[k] * np.random.randn(samplecount)
    # patients (age , gender)

    results = imputation_pos_all(dataset, distr, samplecount, threshold)
    return results

