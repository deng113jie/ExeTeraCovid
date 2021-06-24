# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

import numpy as np

from exetera.core.utils import check_input_lengths
from exetera.core import persistence


class CheckInconsistentSymptoms:
    def __init__(self, f_healthy_but_symptoms, f_not_healthy_but_no_symptoms):
        warnings.warn("deprecated", DeprecationWarning)
        self.f_healthy_but_symptoms = f_healthy_but_symptoms
        self.f_not_healthy_but_no_symptoms = f_not_healthy_but_no_symptoms

    def __call__(self, healthy, symptoms, flags, i_healthy, i_not_healthy):
        check_input_lengths(('healthy', 'symptoms', 'flags'), (healthy, symptoms, flags))
        for i_r in range(len(healthy)):
            if healthy[i_r] == i_healthy and symptoms[i_r]:
                flags[i_r] |= self.f_healthy_but_symptoms
            elif healthy[i_r] == i_not_healthy and not symptoms[i_r]:
                flags[i_r] |= self.f_not_healthy_but_no_symptoms


def check_inconsistent_symptoms_1(datastore, src_assessments, dest_assessments, timestamp=None):
    """
    Deprecated, please use inconsistent_symptoms.check_inconsistent_symptoms_v1().
    """
    warnings.warn("deprecated", DeprecationWarning)
    if timestamp is None:
        timestamp = datastore.timestamp

    generated_health_fields = ()
    # generated_health_fields = ('has_temperature',)

    # TODO: check that all fields are leaky booleans
    health_check_fields = (
        'fever', 'persistent_cough', 'fatigue', 'shortness_of_breath', 'diarrhoea',
        'delirium', 'skipped_meals', 'abdominal_pain', 'chest_pain', 'hoarse_voice',
        'loss_of_smell', 'headache', 'chills_or_shivers', 'eye_soreness', 'nausea',
        'dizzy_light_headed', 'red_welts_on_face_or_lips', 'blisters_on_feet', 'sore_throat',
        'unusual_muscle_pains'
    )

    health_status = datastore.get_reader(src_assessments['health_status'])
    health_status_array = health_status[:]

    combined_results = np.zeros(len(health_status), dtype=np.bool)

    for h in health_check_fields:
        if h not in src_assessments.keys():
            print(f"warning: field {h} is not present in this dataset")
        else:
            f = datastore.get_reader(src_assessments[h])
            combined_results = combined_results & (f[:] != 2)

    for h in generated_health_fields:
        if h not in src_assessments.keys():
            print(f"warning: field {h} is not present in this dataset")
        else:
            f = datastore.get_reader(src_assessments[h])
            combined_results = combined_results and f[:]

    inconsistent_healthy =\
        datastore.get_numeric_writer(dest_assessments,
                                     'inconsistent_healthy', 'bool', timestamp)
    inconsistent_not_healthy = \
        datastore.get_numeric_writer(dest_assessments,
                                     'inconsistent_not_healthy', 'bool', timestamp)

    # TODO: replace DataStore with Session
    # inconsistent_healthy =\
    #     datastore.create_numeric(dest_assessments, 'inconsistent_healthy', 'bool', timestamp)
    # inconsistent_not_healthy = \
    #     datastore.create_numeric(dest_assessments, 'inconsistent_not_healthy', 'bool', timestamp)

    inconsistent_healthy.write((health_status_array == 1) & (combined_results is False))
    inconsistent_not_healthy.write((health_status_array == 2) & (combined_results is True))


def check_inconsistent_symptoms_v1(session, src_assessments, dest_assessments, timestamp=None):
    """
    Checking if the health status recorded by user is consistent over different columns, for example in 'health_status'
        reported healthy but also reported 'fever'.

    :param session: The Exetera session instance.
    :param src_assessments: The assessments dataframe contains reported data.
    :param dest_assessments: The destination dataframe to write the result columns to.
    :param timestamp: The timestamp to use when create destination fields.
    """
    generated_health_fields = ()
    # generated_health_fields = ('has_temperature',)

    # TODO: check that all fields are leaky booleans
    health_check_fields = (
        'fever', 'persistent_cough', 'fatigue', 'shortness_of_breath', 'diarrhoea',
        'delirium', 'skipped_meals', 'abdominal_pain', 'chest_pain', 'hoarse_voice',
        'loss_of_smell', 'headache', 'chills_or_shivers', 'eye_soreness', 'nausea',
        'dizzy_light_headed', 'red_welts_on_face_or_lips', 'blisters_on_feet', 'sore_throat',
        'unusual_muscle_pains'
    )

    health_status = session.get(src_assessments['health_status'])
    health_status_array = health_status.data[:]

    combined_results = np.zeros(len(health_status), dtype=np.bool)

    for h in health_check_fields:
        if h not in src_assessments.keys():
            print(f"warning: field {h} is not present in this dataset")
        else:
            f = session.get(src_assessments[h])
            combined_results = combined_results & (f.data[:] != 2)

    for h in generated_health_fields:
        if h not in src_assessments.keys():
            print(f"warning: field {h} is not present in this dataset")
        else:
            f = session.get(src_assessments[h])
            combined_results = combined_results and f.data[:]

    inconsistent_healthy =\
        session.create_numeric(dest_assessments, 'inconsistent_healthy', 'bool', timestamp)
    inconsistent_not_healthy = \
        session.create_numeric(dest_assessments, 'inconsistent_not_healthy', 'bool', timestamp)

    # TODO: replace DataStore with Session
    # inconsistent_healthy =\
    #     datastore.create_numeric(dest_assessments, 'inconsistent_healthy', 'bool', timestamp)
    # inconsistent_not_healthy = \
    #     datastore.create_numeric(dest_assessments, 'inconsistent_not_healthy', 'bool', timestamp)

    inconsistent_healthy.data.write((health_status_array == 1) & (combined_results is False))
    inconsistent_not_healthy.data.write((health_status_array == 2) & (combined_results is True))
