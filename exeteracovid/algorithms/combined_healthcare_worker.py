import warnings

import numpy as np

from exetera.core import validation as val


def combined_hcw_with_contact(datastore,
                              healthcare_professional, contact_health_worker,
                              is_carer_for_community,
                              group, name):
    """
    Deprecated, please use combined_healthcare_worker.combined_hcw_with_contact_v1().
    """
    warnings.warn("deprecated", DeprecationWarning)
    raw_hcp = val.raw_array_from_parameter(datastore, 'healthcare_professional',
                                           healthcare_professional)
    filter_ = np.where(raw_hcp == 0,
                       0,
                       np.where(raw_hcp == 1,
                                1,
                                np.where(raw_hcp < 4,
                                         2,
                                         3)))
    raw_chw = val.raw_array_from_parameter(datastore, 'contact_health_worker',
                                           contact_health_worker)
    filter_ = np.maximum(filter_, np.where(raw_chw == 2, 3, raw_chw))

    raw_icfc = val.raw_array_from_parameter(datastore, 'is_carer_for_community',
                                            is_carer_for_community)
    filter_ = np.maximum(filter_,
                         np.where(raw_icfc == 2, 3, raw_icfc))
    key = {'': 0, 'no': 1, 'yes_no_contact': 2, 'yes_contact': 3}
    hccw = datastore.get_categorical_writer(group, name, categories=key)
    hccw.write(filter_)
    return hccw


def combined_hcw_with_contact_v1(session,
                                 healthcare_professional, contact_health_worker,
                                 is_carer_for_community,
                                 group, name):
    """
    Identify the users in Covid dataset who are health workers with contact history.

    :param healthcare_professional: The healthcare_professional column from dataset.
    :param contact_health_worker: The contact_health_worker column from dataset.
    :param is_carer_for_community: The is_carer_for_community column from dataset.
    :param group: The dataframe to store the result field to.
    :param name: The name of the result field.
    :return: The categorical field which identifying health workers with contact history.
    """
    raw_hcp = val.raw_array_from_parameter(session, 'healthcare_professional',
                                           healthcare_professional)
    filter_ = np.where(raw_hcp == 0,
                       0,
                       np.where(raw_hcp == 1,
                                1,
                                np.where(raw_hcp < 4,
                                         2,
                                         3)))
    raw_chw = val.raw_array_from_parameter(session, 'contact_health_worker',
                                           contact_health_worker)
    filter_ = np.maximum(filter_, np.where(raw_chw == 2, 3, raw_chw))

    raw_icfc = val.raw_array_from_parameter(session, 'is_carer_for_community',
                                            is_carer_for_community)
    filter_ = np.maximum(filter_,
                         np.where(raw_icfc == 2, 3, raw_icfc))
    key = {'': 0, 'no': 1, 'yes_no_contact': 2, 'yes_contact': 3}
    hccw = session.create_categorical(group, name, 'int8', key)
    hccw.data.write(filter_)
    return hccw
