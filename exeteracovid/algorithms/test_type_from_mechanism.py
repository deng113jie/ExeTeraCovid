import numpy as np

from exetera.core import validation as val


def test_type_from_mechanism_v1(datastore, mechanism, mechanism_free,
                                pcr_standard_answers, pcr_strong_inferred, pcr_weak_inferred,
                                antibody_standard_answers, antibody_strong_inferred, antibody_weak_inferred):
    """
    Classify the test mechanism by using data from the 'mechanism' field and user filled free text.

    :param datastore: The Exetera session instance.
    :param mechanism: The 'mechanism' column from the tests dataframe.
    :param mechanism_free: The 'mechanism_free' column from the tests dataframe.
    :param pcr_standard_answers: The field to indicate a standard pcr test performed.
    :param pcr_strong_inferred: The field to indicate a strong pcr test performed.
    :param pcr_weak_inferred: The field to indicate a weak pcr test performed.
    :param antibody_standard_answers: The field to indicate a standard antibody test performed.
    :param antibody_strong_inferred: The field to indicate a strong antibody test performed.
    :param antibody_weak_inferred: The field to indicate a weak antibody test performed.
    """

    def search_for_substring(text_entries, pattern):
        filt = np.zeros(len(text_entries), np.bool)
        for ie, e in enumerate(text_entries):
            if pattern in e.lower():
                filt[ie] = True
        return filt

    antigen_exclusions = ('nose_throat_swab', 'throat_swab', 'nose_swab', 'spit_tube')
    antibody_exclusions = ('blood_sample', 'blood_sample_finger_prick', 'blood_sample_needle_draw')

    antibody_strong = ('blood', 'antib', 'anti b', 'anti-b', 'prick', 'antikro', 'anti kro', 'blod', 'all three', 'all tests', 'all of')
    antibody_weak = ('prick', 'stick i f', 'finger')
    pcr_strong = ('swab', 'swap', 'swob', 'swan', 'tonsil', 'nose', 'throat', 'näsa', 'svalg', 'oral', 'nasoph',
                      'saliva', 'all three', 'all tests', 'all of', 'plasma', 'drive t', 'drivet')
    pcr_weak = ('self test', 'self admin', 'home test', 'home admin', 'self', 'home', 'post', 'i did it', 'drive', 'hemma', 'private')


    r_mechanism = val.raw_array_from_parameter(datastore, 'mechanism', mechanism)

    f_pcr_cat = np.isin(r_mechanism, (1, 2, 3, 4))
    if isinstance(pcr_standard_answers, np.ndarray):
        pcr_standard_answers[:] = f_pcr_cat
    else:
        pcr_standard_answers.data.write(f_pcr_cat)

    f_atb_cat = np.isin(r_mechanism, (5, 6, 7))
    if isinstance(antibody_standard_answers, np.ndarray):
        antibody_standard_answers[:] = f_atb_cat
    else:
        antibody_standard_answers.data.write(f_atb_cat)

    r_mechanism_free = val.raw_array_from_parameter(datastore, 'mechanism_free', mechanism_free)

    f_pcr_strong = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in pcr_strong:
        filt = search_for_substring(r_mechanism_free, p)
        f_pcr_strong = f_pcr_strong | filt
    if isinstance(pcr_strong_inferred, np.ndarray):
        pcr_strong_inferred[:] = f_pcr_strong
    else:
        pcr_strong_inferred.data.write(f_pcr_strong)

    f_pcr_weak = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in pcr_weak:
        filt = search_for_substring(r_mechanism_free, p)
        f_pcr_weak = f_pcr_weak | filt
    if isinstance(pcr_weak_inferred, np.ndarray):
        pcr_weak_inferred[:] = f_pcr_weak
    else:
        pcr_weak_inferred.data.write(f_pcr_weak)

    f_antibody_strong = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in antibody_strong:
        filt = search_for_substring(r_mechanism_free, p)
        f_antibody_strong = f_antibody_strong | filt
    if isinstance(antibody_strong_inferred, np.ndarray):
        antibody_strong_inferred[:] = f_antibody_strong
    else:
        antibody_strong_inferred.data.write(f_antibody_strong)

    f_antibody_weak = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in antibody_weak:
        filt = search_for_substring(r_mechanism_free, p)
        f_antibody_weak = f_antibody_weak | filt
    if isinstance(antibody_weak_inferred, np.ndarray):
        antibody_weak_inferred[:] = f_antibody_weak
    else:
        antibody_weak_inferred.data.write(f_antibody_weak)

    # count_in_exclusion = 0
    # for r in r_mechanism_free:
    #     if r.lower in antigen_exclusions:
    #         count_in_exclusion += 1
    # print(count_in_exclusion)


def pcr_standard_summarize_v1(session, test_df):
    reader_pcr_sa = test_df['pcr_standard_answers']
    reader_pcr_si = test_df['pcr_strong_inferred']
    reader_pcr_wi = test_df['pcr_weak_inferred']

    pcr_standard = reader_pcr_si + reader_pcr_sa + reader_pcr_wi
    pcr_standard = np.where(pcr_standard.data[:] > 0, np.ones_like(pcr_standard.data[:]),
                            np.zeros_like(pcr_standard.data[:]))

    write = session.create_numeric(test_df, 'pcr_standard', 'bool')
    write.data.write(pcr_standard)


def test_type_from_mechanism_v1_standard_input(datastore, test_df):
    mechanism = test_df['mechanism']
    mechanism_free = test_df['mechanism_freetext']
    pcr_standard_answers = test_df.create_numeric('pcr_standard_answers', 'bool').data
    pcr_strong_inferred = test_df.create_numeric('pcr_strong_inferred', 'bool').data
    pcr_weak_inferred = test_df.create_numeric('pcr_weak_inferred', 'bool').data
    antibody_standard_answers = test_df.create_numeric('antibody_standard_answers', 'bool').data
    antibody_strong_inferred = test_df.create_numeric('antibody_strong_inferred', 'bool').data
    antibody_weak_inferred = test_df.create_numeric('antibody_weak_inferred', 'bool').data

    def search_for_substring(text_entries, pattern):
        filt = np.zeros(len(text_entries), np.bool)
        for ie, e in enumerate(text_entries):
            if pattern in e.lower():
                filt[ie] = True
        return filt

    antigen_exclusions = ('nose_throat_swab', 'throat_swab', 'nose_swab', 'spit_tube')
    antibody_exclusions = ('blood_sample', 'blood_sample_finger_prick', 'blood_sample_needle_draw')

    antibody_strong = ('blood', 'antib', 'anti b', 'anti-b', 'prick', 'antikro', 'anti kro', 'blod', 'all three', 'all tests', 'all of')
    antibody_weak = ('prick', 'stick i f', 'finger')
    pcr_strong = ('swab', 'swap', 'swob', 'swan', 'tonsil', 'nose', 'throat', 'näsa', 'svalg', 'oral', 'nasoph',
                      'saliva', 'all three', 'all tests', 'all of', 'plasma', 'drive t', 'drivet')
    pcr_weak = ('self test', 'self admin', 'home test', 'home admin', 'self', 'home', 'post', 'i did it', 'drive', 'hemma', 'private')


    r_mechanism = val.raw_array_from_parameter(datastore, 'mechanism', mechanism)

    f_pcr_cat = np.isin(r_mechanism, (1, 2, 3, 4))
    if isinstance(pcr_standard_answers, np.ndarray):
        pcr_standard_answers[:] = f_pcr_cat
    else:
        pcr_standard_answers.write(f_pcr_cat)

    f_atb_cat = np.isin(r_mechanism, (5, 6, 7))
    if isinstance(antibody_standard_answers, np.ndarray):
        antibody_standard_answers[:] = f_atb_cat
    else:
        antibody_standard_answers.write(f_atb_cat)

    r_mechanism_free = val.raw_array_from_parameter(datastore, 'mechanism_free', mechanism_free)

    f_pcr_strong = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in pcr_strong:
        filt = search_for_substring(r_mechanism_free, p)
        f_pcr_strong = f_pcr_strong | filt
    if isinstance(pcr_strong_inferred, np.ndarray):
        pcr_strong_inferred[:] = f_pcr_strong
    else:
        pcr_strong_inferred.write(f_pcr_strong)

    f_pcr_weak = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in pcr_weak:
        filt = search_for_substring(r_mechanism_free, p)
        f_pcr_weak = f_pcr_weak | filt
    if isinstance(pcr_weak_inferred, np.ndarray):
        pcr_weak_inferred[:] = f_pcr_weak
    else:
        pcr_weak_inferred.write(f_pcr_weak)

    f_antibody_strong = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in antibody_strong:
        filt = search_for_substring(r_mechanism_free, p)
        f_antibody_strong = f_antibody_strong | filt
    if isinstance(antibody_strong_inferred, np.ndarray):
        antibody_strong_inferred[:] = f_antibody_strong
    else:
        antibody_strong_inferred.write(f_antibody_strong)

    f_antibody_weak = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in antibody_weak:
        filt = search_for_substring(r_mechanism_free, p)
        f_antibody_weak = f_antibody_weak | filt
    if isinstance(antibody_weak_inferred, np.ndarray):
        antibody_weak_inferred[:] = f_antibody_weak
    else:
        antibody_weak_inferred.write(f_antibody_weak)

    # count_in_exclusion = 0
    # for r in r_mechanism_free:
    #     if r.lower in antigen_exclusions:
    #         count_in_exclusion += 1
    # print(count_in_exclusion)