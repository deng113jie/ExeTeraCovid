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

from exetera.core import fields
from exetera.core.utils import check_input_lengths
from exetera.core import persistence as persist


def calculate_age_from_year_of_birth_fast(datastore, min_age, max_age,
                                          year_of_birth, year_of_birth_filter,
                                          age, age_filter, age_range_filter, year,
                                          chunksize=None, timestamp=None):
    warnings.warn("deprecated", DeprecationWarning)
    yob_v = datastore.get_reader(year_of_birth)
    yob_f = datastore.get_reader(year_of_birth_filter)
    raw_ages = year - yob_v[:]
    raw_age_filter = yob_f[:]
    raw_age_range_filter = raw_age_filter & (min_age <= raw_ages) & (raw_ages <= max_age)
    age.write_part(raw_ages)
    age_filter.write_part(raw_age_filter)
    age_range_filter.write_part(raw_age_range_filter)
    age.flush()
    age_filter.flush()
    age_range_filter.flush()


def calculate_age_from_year_of_birth_v1(year_of_birth, year_of_birth_filter,
                                        min_age, max_age,
                                        age, age_filter, age_range_filter, year):
    if isinstance(year_of_birth, fields.Field):
        yob_v = year_of_birth.data[:]
    else:
        yob_v = year_of_birth

    if isinstance(year_of_birth_filter, fields.Field):
        yob_f = year_of_birth_filter.data[:]
    else:
        yob_f = year_of_birth_filter

    raw_ages = year - yob_v
    raw_age_filter = yob_f
    raw_age_range_filter = raw_age_filter & (raw_ages >= min_age) & (raw_ages <= max_age)

    if isinstance(age, fields.Field):
        age.data.write(raw_ages)
    else:
        age[:] = raw_ages

    if isinstance(age_filter, fields.Field):
        age_filter.data.write(raw_age_filter)
    else:
        age_filter[:] = raw_age_filter

    if isinstance(age_range_filter, fields.Field):
        age_range_filter.data.write(raw_age_range_filter)
    else:
        age_range_filter[:] = raw_age_range_filter
