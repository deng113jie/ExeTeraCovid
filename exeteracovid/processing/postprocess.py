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

from datetime import datetime
import time
from collections import defaultdict

import numpy as np
import numba

from exeteracovid.algorithms.age_from_year_of_birth import calculate_age_from_year_of_birth_v1
from exeteracovid.algorithms.weight_height_bmi import weight_height_bmi_v1
from exeteracovid.algorithms.inconsistent_symptoms import check_inconsistent_symptoms_v1
from exeteracovid.algorithms.temperature import validate_temperature_v1
from exeteracovid.algorithms.combined_healthcare_worker import combined_hcw_with_contact_v1
from exeteracovid.algorithms.effective_test_date import effective_test_date_v1
from exetera.core import persistence
from exetera.core import operations as ops
from exetera.core.session import Session
from exetera.core.dataset import Dataset
from exetera.core import readerwriter as rw
from exetera.core import fields, utils
from exetera.core import dataframe


def log(*a, **kwa):
    print(*a, **kwa)


def postprocess_v1(s: Session,
                   src_dataset: Dataset,
                   dest_dataset: Dataset,
                   timestamp=None,
                   flags=None):

    if flags is None:
        flags = set()

    do_daily_asmts = 'daily' in flags
    has_patients = 'patients' in src_dataset.keys()
    has_assessments = 'assessments' in src_dataset.keys()
    has_tests = 'tests' in src_dataset.keys()
    has_diet = 'diet' in src_dataset.keys()
    print("has_patients: {}".format(has_patients))
    print("has_assessments: {}".format(has_assessments))
    print("has_tests: {}".format(has_tests))
    print("has_diet: {}".format(has_diet))

    sort_enabled = lambda x: True
    process_enabled = lambda x: True

    sort_patients = sort_enabled(flags) and True
    sort_assessments = sort_enabled(flags) and True
    sort_tests = sort_enabled(flags) and True
    sort_diet = sort_enabled(flags) and True

    make_assessment_patient_id_fkey = process_enabled(flags) and True
    year_from_age = process_enabled(flags) and True
    clean_weight_height_bmi = process_enabled(flags) and True
    health_worker_with_contact = process_enabled(flags) and True
    clean_temperatures = process_enabled(flags) and True
    check_symptoms = process_enabled(flags) and True
    create_daily = process_enabled(flags) and do_daily_asmts
    make_patient_level_assessment_metrics = process_enabled(flags) and True
    make_patient_level_daily_assessment_metrics = process_enabled(flags) and do_daily_asmts
    make_new_test_level_metrics = process_enabled(flags) and True
    make_diet_level_metrics = True
    make_healthy_diet_index = True

    # ds = DataStore(timestamp=timestamp)

    # patients ================================================================

    sorted_patients_src = None

    if has_patients:
        patients_src = src_dataset['patients']

        write_mode = 'write'

        if 'patients' not in dest_dataset.keys():
            patients_dest = s.get_or_create_group(dest_dataset, 'patients')
            sorted_patients_src = patients_dest

            # Patient sort
            # ============
            if sort_patients:
                duplicate_filter = \
                    persistence.filter_duplicate_fields(s.get(patients_src['id']).data[:])

                for k in patients_src.keys():
                    t0 = time.time()
                    r = s.get(patients_src[k])
                    w = r.create_like(patients_dest, k)
                    s.apply_filter(duplicate_filter, r, w)
                    print(f"'{k}' filtered in {time.time() - t0}s")

                print(np.count_nonzero(duplicate_filter == True),
                      np.count_nonzero(duplicate_filter == False))
                sort_keys = ('id',)
                s.sort_on(
                    patients_dest, patients_dest, sort_keys, write_mode='overwrite')

            # Patient processing
            # ==================
            if year_from_age:
                log("year of birth -> age; 18 to 90 filter")
                t0 = time.time()
                yobs = s.get(patients_dest['year_of_birth'])
                yob_filter = s.get(patients_dest['year_of_birth_valid'])
                age = s.create_numeric(patients_dest, 'age', 'uint32')
                age_filter = s.create_numeric(patients_dest, 'age_filter', 'bool')
                age_16_to_90 = s.create_numeric(patients_dest, '16_to_90_years', 'bool')
                print('year_of_birth:', patients_dest['year_of_birth'])
                # for k in patients_dest['year_of_birth'].attrs.keys():
                #     print(k, patients_dest['year_of_birth'].attrs[k])
                calculate_age_from_year_of_birth_v1(
                    yobs, yob_filter, 16, 90, age, age_filter, age_16_to_90, 2020)
                log(f"completed in {time.time() - t0}")

                # print('age_filter count:', np.sum(patients_dest['age_filter']['values'][:]))
                # print('16_to_90_years count:', np.sum(patients_dest['16_to_90_years']['values'][:]))

            if clean_weight_height_bmi:
                log("height / weight / bmi; standard range filters")
                t0 = time.time()

                weights_clean = s.create_numeric(patients_dest, 'weight_kg_clean', 'float32')
                weights_filter = s.create_numeric(patients_dest, '40_to_200_kg', 'bool')
                heights_clean = s.create_numeric(patients_dest, 'height_cm_clean', 'float32')
                heights_filter = s.create_numeric(patients_dest, '110_to_220_cm', 'bool')
                bmis_clean = s.create_numeric(patients_dest, 'bmi_clean', 'float32')
                bmis_filter = s.create_numeric(patients_dest, '15_to_55_bmi', 'bool')

                weight_height_bmi_v1(s, 40, 200, 110, 220, 15, 55,
                                     None, None, None, None,
                                     patients_dest['weight_kg'], patients_dest['weight_kg_valid'],
                                     patients_dest['height_cm'], patients_dest['height_cm_valid'],
                                     patients_dest['bmi'], patients_dest['bmi_valid'],
                                     weights_clean, weights_filter, None,
                                     heights_clean, heights_filter, None,
                                     bmis_clean, bmis_filter, None)
                log(f"completed in {time.time() - t0}")

            if health_worker_with_contact:
                with utils.Timer("health_worker_with_contact field"):
                    #writer = ds.get_categorical_writer(patients_dest, 'health_worker_with_contact', 'int8')
                    combined_hcw_with_contact_v1(s,
                                              s.get(patients_dest['healthcare_professional']),
                                              s.get(patients_dest['contact_health_worker']),
                                              s.get(patients_dest['is_carer_for_community']),
                                              patients_dest, 'health_worker_with_contact')

    # assessments =============================================================

    sorted_assessments_src = None
    if has_assessments:
        assessments_src = src_dataset['assessments']
        print(assessments_src.keys())
        if 'assessments' not in dest_dataset.keys():
            assessments_dest = s.get_or_create_group(dest_dataset, 'assessments')
            sorted_assessments_src = assessments_dest

            if sort_assessments:
                sort_keys = ('patient_id', 'created_at')
                with utils.Timer("sorting assessments"):
                    s.sort_on(
                        assessments_src, assessments_dest, sort_keys)

            if has_patients:
                if make_assessment_patient_id_fkey:
                    print("creating 'assessment_patient_id_fkey' foreign key index for 'patient_id'")
                    t0 = time.time()
                    patient_ids = s.get(sorted_patients_src['id'])
                    assessment_patient_ids =\
                        s.get(sorted_assessments_src['patient_id'])
                    assessment_patient_id_fkey =\
                        s.create_numeric(assessments_dest, 'assessment_patient_id_fkey', 'int64')
                    s.get_index(patient_ids.data[:], assessment_patient_ids.data[:], assessment_patient_id_fkey)
                    print(f"completed in {time.time() - t0}s")

            if clean_temperatures:
                print("clean temperatures")
                t0 = time.time()
                temps = s.get(sorted_assessments_src['temperature'])
                temp_units = s.get(sorted_assessments_src['temperature_unit'])
                temps_valid = s.get(sorted_assessments_src['temperature_valid'])
                dest_temps = temps.create_like(assessments_dest, 'temperature_c_clean')
                dest_temps_valid = temps_valid.create_like(assessments_dest, 'temperature_35_to_42_inclusive')
                dest_temps_modified = temps_valid.create_like(assessments_dest, 'temperature_modified')
                validate_temperature_v1(s, 35.0, 42.0,
                                       temps, temp_units, temps_valid,
                                       dest_temps, dest_temps_valid, dest_temps_modified)
                print(f"temperature cleaning done in {time.time() - t0}")

            if check_symptoms:
                print('check inconsistent health_status')
                t0 = time.time()
                check_inconsistent_symptoms_v1(s, sorted_assessments_src, assessments_dest)
                print(time.time() - t0)

    # tests ===================================================================

    if has_tests:
        if sort_tests:
            tests_src = src_dataset['tests']
            tests_dest = s.get_or_create_group(dest_dataset, 'tests')
            sort_keys = ('patient_id', 'created_at')
            s.sort_on(tests_src, tests_dest, sort_keys)

    # diet ====================================================================

    if has_diet:
        diet_src = src_dataset['diet']
        if 'diet' not in dest_dataset.keys():
            diet_dest = s.get_or_create_group(dest_dataset, 'diet')
            sorted_diet_src = diet_dest
            if sort_diet:
                sort_keys = ('patient_id', 'display_name', 'id')
                s.sort_on(diet_src, diet_dest, sort_keys)


    if has_assessments:
        if do_daily_asmts:
            daily_assessments_dest = s.get_or_create_group(dest_dataset, 'daily_assessments')


    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(src_dataset['assessments'].keys())
    print(src_dataset['tests'].keys())

    # write_mode = 'overwrite'
    write_mode = 'write'


    # Daily assessments
    # =================

    if has_assessments:
        if create_daily:
            print(sorted_assessments_src.keys())
            print("generate daily assessments")
            patient_ids = s.get(sorted_assessments_src['patient_id'])
            created_at_days = s.get(sorted_assessments_src['created_at_day'])
            raw_created_at_days = created_at_days.data[:]

            if 'assessment_patient_id_fkey' in assessments_src.keys():
                patient_id_index = assessments_src['assessment_patient_id_fkey']
            else:
                patient_id_index = assessments_dest['assessment_patient_id_fkey']
            patient_id_indices = s.get(patient_id_index)
            raw_patient_id_indices = patient_id_indices.data[:]


            print("Calculating patient id index spans")
            t0 = time.time()
            patient_id_index_spans = s.get_spans(fields=(raw_patient_id_indices,
                                                         raw_created_at_days))
            print(f"Calculated {len(patient_id_index_spans)-1} spans in {time.time() - t0}s")


            print("Applying spans to 'health_status'")
            t0 = time.time()
            default_behavour_overrides = {
                'id': s.apply_spans_last,
                'patient_id': s.apply_spans_last,
                'patient_index': s.apply_spans_last,
                'created_at': s.apply_spans_last,
                'created_at_day': s.apply_spans_last,
                'updated_at': s.apply_spans_last,
                'updated_at_day': s.apply_spans_last,
                'version': s.apply_spans_max,
                'country_code': s.apply_spans_first,
                'date_test_occurred': None,
                'date_test_occurred_guess': None,
                'date_test_occurred_day': None,
                'date_test_occurred_set': None,
            }
            for k in sorted_assessments_src.keys():
                t1 = time.time()
                reader = s.get(sorted_assessments_src[k])
                if k in default_behavour_overrides:
                    apply_span_fn = default_behavour_overrides[k]
                    if apply_span_fn is not None:
                        apply_span_fn(patient_id_index_spans, reader,
                                      reader.create_like(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    else:
                        print(f"  Skipping field {k}")
                else:
                    if isinstance(reader, fields.CategoricalField):
                        s.apply_spans_max(
                            patient_id_index_spans, reader,
                            reader.create_like(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    elif isinstance(reader, rw.IndexedStringReader):
                        s.apply_spans_concat(
                            patient_id_index_spans, reader,
                            reader.create_like(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    elif isinstance(reader, rw.NumericReader):
                        s.apply_spans_max(
                            patient_id_index_spans, reader,
                            reader.create_like(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    else:
                        print(f"  No function for {k}")

            print(f"apply_spans completed in {time.time() - t0}s")

    if has_patients and has_assessments:
            if make_patient_level_assessment_metrics:
                if 'assessment_patient_id_fkey' in assessments_dest:
                    src = assessments_dest['assessment_patient_id_fkey']
                else:
                    src = assessments_src['assessment_patient_id_fkey']
                assessment_patient_id_fkey = s.get(src)
                # generate spans from the assessment-space patient_id foreign key
                spans = s.get_spans(field=assessment_patient_id_fkey.data[:])

                ids = s.get(patients_dest['id'])

                print('calculate assessment counts per patient')
                t0 = time.time()
                writer = s.create_numeric(patients_dest, 'assessment_count', 'uint32')
                aggregated_counts = s.apply_spans_count(spans)
                s.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
                print(f"calculated assessment counts per patient in {time.time() - t0}")

                print('calculate first assessment days per patient')
                t0 = time.time()
                reader = s.get(sorted_assessments_src['created_at_day'])
                writer = s.create_fixed_string(patients_dest, 'first_assessment_day', 10)
                aggregated_counts = s.apply_spans_first(spans, reader)
                s.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
                print(f"calculated first assessment days per patient in {time.time() - t0}")

                print('calculate last assessment days per patient')
                t0 = time.time()
                reader = s.get(sorted_assessments_src['created_at_day'])
                writer = s.create_fixed_string(patients_dest, 'last_assessment_day', 10)
                aggregated_counts = s.apply_spans_last(spans, reader)
                s.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
                print(f"calculated last assessment days per patient in {time.time() - t0}")

                print('calculate maximum assessment test result per patient')
                t0 = time.time()
                reader = s.get(sorted_assessments_src['tested_covid_positive'])
                writer = reader.create_like(patients_dest, 'max_assessment_test_result')
                max_result_value = s.apply_spans_max(spans, reader)
                s.join(ids, assessment_patient_id_fkey, max_result_value, writer, spans)
                print(f"calculated maximum assessment test result in {time.time() - t0}")


    if has_assessments and do_daily_asmts and make_patient_level_daily_assessment_metrics:
        print("creating 'daily_assessment_patient_id_fkey' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = s.get(sorted_patients_src['id'])
        daily_assessment_patient_ids =\
            s.get(daily_assessments_dest['patient_id'])
        daily_assessment_patient_id_fkey =\
            s.create_numeric(daily_assessments_dest, 'daily_assessment_patient_id_fkey', 'int64')
        s.get_index(patient_ids, daily_assessment_patient_ids,
                    daily_assessment_patient_id_fkey)
        print(f"completed in {time.time() - t0}s")

        spans = s.get_spans(
            field=s.get(daily_assessments_dest['daily_assessment_patient_id_fkey']))

        print('calculate daily assessment counts per patient')
        t0 = time.time()
        writer = s.create_numeric(patients_dest, 'daily_assessment_count', 'uint32')
        aggregated_counts = s.apply_spans_count(spans)
        daily_assessment_patient_id_fkey =\
            s.get(daily_assessments_dest['daily_assessment_patient_id_fkey'])
        s.join(ids, daily_assessment_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated daily assessment counts per patient in {time.time() - t0}")


    if has_tests and make_new_test_level_metrics:
        print("creating 'test_patient_id_fkey' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = s.get(sorted_patients_src['id'])
        test_patient_ids = s.get(tests_dest['patient_id'])
        test_patient_id_fkey = s.create_numeric(tests_dest, 'test_patient_id_fkey', 'int64')
        s.get_index(patient_ids, test_patient_ids, test_patient_id_fkey)
        test_patient_id_fkey = s.get(tests_dest['test_patient_id_fkey'])
        spans = s.get_spans(field=test_patient_id_fkey)
        print(f"completed in {time.time() - t0}s")

        print('calculate test_counts per patient')
        t0 = time.time()
        writer = s.create_numeric(patients_dest, 'test_count', 'uint32')
        aggregated_counts = s.apply_spans_count(spans)
        s.join(ids, test_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated test counts per patient in {time.time() - t0}")

        print('calculate test_result per patient')
        t0 = time.time()
        test_results = s.get(tests_dest['result'])
        writer = test_results.create_like(patients_dest, 'max_test_result')
        aggregated_results = s.apply_spans_max(spans, test_results)
        s.join(ids, test_patient_id_fkey, aggregated_results, writer, spans)
        print(f"calculated max_test_result per patient in {time.time() - t0}")

    if has_diet and make_diet_level_metrics:
        with utils.Timer("Making patient-level diet questions count", new_line=True):
            d_pids_ = s.get(diet_dest['patient_id']).data[:]
            d_pid_spans = s.get_spans(d_pids_)
            d_distinct_pids = s.apply_spans_first(d_pid_spans, d_pids_)
            d_pid_counts = s.apply_spans_count(d_pid_spans)
            p_diet_counts = s.create_numeric(patients_dest, 'diet_counts', 'int32')
            s.merge_left(left_on=s.get(patients_dest['id']).data[:], right_on=d_distinct_pids,
                         right_fields=(d_pid_counts,), right_writers=(p_diet_counts,))


def postprocess_v2(s: Session,
                   src_file, temp_file, dest_file, timestamp=None, flags=None):
    if flags is None:
        flags = dict()

    src_dataset = s.open_dataset(src_file, 'r', 'src')
    temp_dataset = s.open_dataset(temp_file, 'w', 'temp')
    dest_dataset = s.open_dataset(dest_file, 'w', 'dest')

    has_patients = 'patients' in src_dataset
    has_assessments = 'assessments' in src_dataset
    has_tests = 'tests' in src_dataset
    has_vaccine_doses = 'vaccine_doses' in src_dataset
    has_vaccine_symptoms = 'vaccine_symptoms' in src_dataset
    has_vaccine_hesitancy = 'vaccine_hesitancy' in src_dataset
    has_mental_health = 'mental_health' in src_dataset


    # sort and generate fields based on patients

    if has_patients:
        src_patients = src_dataset['patients']
        temp_patients = temp_dataset.create_dataframe('patients')
        dest_patients = dest_dataset.create_dataframe('patients')

        # sort on 'id'
        with utils.Timer("calculating patient sorted index"):
            indices = s.dataset_sort_index((src_patients['id'], src_patients['created_at']))

        with utils.Timer("applying patient sorted index to patients"):
            src_patients.apply_index(indices, temp_patients)
        # src_patients.sort_on(temp_patients, ('id', 'created_at'), verbose=True)

        with utils.Timer("Calculate age and age filters"):
            calculate_age_from_year_of_birth_v1(temp_patients['year_of_birth'],
                                                temp_patients['year_of_birth_valid'],
                                                16,
                                                90,
                                                temp_patients.create_numeric('age', 'int32'),
                                                temp_patients.create_numeric('age_filter', 'bool'),
                                                temp_patients.create_numeric('16_to_90_years', 'bool'),
                                                2020)

        with utils.Timer("Clean height/weight/bmi and calculate filters"):
            weight_height_bmi_v1(s, 40, 200, 110, 220, 15, 55,
                                 None, None, None, None,
                                 temp_patients['weight_kg'],
                                 temp_patients['weight_kg_valid'],
                                 temp_patients['height_cm'],
                                 temp_patients['height_cm_valid'],
                                 temp_patients['bmi'],
                                 temp_patients['bmi_valid'],
                                 temp_patients.create_numeric('weight_kg_clean', 'float32'),
                                 temp_patients.create_numeric('40_to_200_kg', 'bool'),
                                 None,
                                 temp_patients.create_numeric('height_cm_clean', 'float32'),
                                 temp_patients.create_numeric('110_to_220_cm', 'bool'),
                                 None,
                                 temp_patients.create_numeric('bmi_clean', 'float32'),
                                 temp_patients.create_numeric('15_to_55_bmi', 'bool'),
                                 None)

        with utils.Timer("Calculate health_care_worker_with_contact"):
            combined_hcw_with_contact_v1(s,
                                         temp_patients['healthcare_professional'],
                                         temp_patients['contact_health_worker'],
                                         temp_patients['is_carer_for_community'],
                                         temp_patients, 'health_worker_with_contact')

        # finally, filter out the duplicate patient entries
        with utils.Timer("Calculate duplicates"):
            duplicate_filter = persistence.filter_duplicate_fields(temp_patients['id'])

        with utils.Timer("Filter patient duplicates and write to the destination dataset"):
            temp_patients.apply_filter(duplicate_filter, dest_patients)


    # sort and generate fields based on assessments

    if has_assessments:
        s_assessments = src_dataset['assessments']
        d_assessments = dest_dataset.create_dataframe('assessments')

        if 'up_to_assessments' in flags:
            keep_first = 10000000
            for k in s_assessments.keys():
                with utils.Timer("Trimming {}".format(k)):
                    s_assessments[k].create_like(d_assessments, k)
                    if s_assessments[k].indexed:
                        d_assessments[k].indices.write(s_assessments[k].indices[:keep_first+1])
                        d_assessments[k].values.write(
                            s_assessments[k].values[:d_assessments[k].indices[keep_first]])
                    else:
                        d_assessments[k].data.write(s_assessments[k].data[:keep_first])
            with utils.Timer("Sort assessments"):
                s.sort_on(d_assessments, d_assessments, ('patient_id', 'created_at'),
                          verbose=True)
        else:
            s.sort_on(s_assessments, d_assessments, ('patient_id', 'created_at'),
                      verbose=True)

        # with utils.Timer("Sort assessments"):
        #     s.sort_on(s_assessments, d_assessments, ('patient_id', 'created_at'), verbose=True)

        with utils.Timer("Clean assessment temperatures"):
            validate_temperature_v1(s, 35.0, 42.0,
                                    d_assessments['temperature'],
                                    d_assessments['temperature_unit'],
                                    d_assessments['temperature_valid'],
                                    d_assessments.create_numeric('temperature_c_clean', 'float32'),
                                    d_assessments.create_numeric('temperature_35_to_42_inclusive',
                                                                 'bool'),
                                    d_assessments.create_numeric('temperature_modified', 'bool'))

            check_inconsistent_symptoms_v1(s, d_assessments, d_assessments)


    # sort and generate fields based on tests

    if has_tests:
        src_tests = src_dataset['tests']
        dest_tests = dest_dataset.create_dataframe('tests')

        with utils.Timer("calculate effective test dates for tests"):
            effective_test_date_v1(datetime(2020, 3, 1).timestamp(),
                                   datetime.now().timestamp(),
                                   src_tests['date_taken_specific'],
                                   src_tests['date_taken_between_start'],
                                   src_tests['date_taken_between_end'],
                                   dest_tests.create_timestamp("effective_test_date"),
                                   dest_tests.create_numeric('effective_test_date_valid', 'bool'))

        with utils.Timer("Calculate sorted index for tests"):
            test_indices = s.dataset_sort_index((src_tests['patient_id'],
                                                 dest_tests['effective_test_date']))

        with utils.Timer("Applying indices to test fields"):
            dest_tests['effective_test_date'].apply_index(test_indices, in_place=True)
            dest_tests['effective_test_date_valid'].apply_index(test_indices, in_place=True)
            src_tests.apply_index(test_indices, dest_tests)


    # sort and generate fields based on vaccine doses

    if has_vaccine_doses:
        src_vaccine_doses = src_dataset['vaccine_doses']
        dest_vaccine_doses = dest_dataset.create_dataframe('vaccine_doses')

        print(len(src_vaccine_doses['date_taken_specific'].data),
              np.count_nonzero(src_vaccine_doses['date_taken_specific'].data[:]))

        with utils.Timer("Sorting vaccine symptoms DataFrame"):
            vaccine_indices = s.dataset_sort_index((src_vaccine_doses['patient_id'],
                                                    src_vaccine_doses['date_taken_specific']))

            src_vaccine_doses.apply_index(vaccine_indices, dest_vaccine_doses)


    # sort and generate fields based on vaccine symptoms

    if has_vaccine_symptoms:
        src_vaccine_symptoms = src_dataset['vaccine_symptoms']
        dest_vaccine_symptoms = dest_dataset.create_dataframe('vaccine_symptoms')

        print(len(src_vaccine_symptoms['date_taken_specific'].data),
              np.count_nonzero(src_vaccine_symptoms['date_taken_specific'].data[:]))

        with utils.Timer("Sorting vaccine symptoms DataFrame"):
            vaccine_indices = s.dataset_sort_index((src_vaccine_symptoms['patient_id'],
                                                src_vaccine_symptoms['date_taken_specific']))

            src_vaccine_symptoms.apply_index(vaccine_indices, dest_vaccine_symptoms)


    # sort and generate fields based on vaccine hesitancy

    if has_vaccine_hesitancy:
        src_vaccine_hesitancy = src_dataset['vaccine_hesitancy']
        dest_vaccine_hesitancy = dest_dataset.create_dataframe('vaccine_hesitancy')

        with utils.Timer("Sorting vaccine hesitancy DataFrame"):
            s.sort_on(src_vaccine_hesitancy, dest_vaccine_hesitancy, ('patient_id', 'created_at'))


    # sort and generate fields based on mental health

    if has_mental_health:
        src_mental_health = src_dataset['mental_health']
        dest_mental_health = dest_dataset.create_dataframe('mental_health')

        with utils.Timer("Sorting mental health DataFrame"):
            s.sort_on(src_mental_health, dest_mental_health, ('patient_id', 'created_at'))


    # Phase 2: merging and aggregating fields


    # generate assessment measures

    if has_patients and has_assessments:
        d_patients = dest_dataset['patients']
        d_assessments = dest_dataset['assessments']
        t_agg_assessments = temp_dataset.create_dataframe('aggregated_assessments')

        spans = s.get_spans(d_assessments['patient_id'])

        t_agg_assessments['patient_id'] = \
            d_assessments['patient_id'].apply_spans_first(spans)

        t_agg_assessments.create_numeric('assessment_counts', 'int32')
        s.apply_spans_count(spans, t_agg_assessments['assessment_counts'])

        t_agg_assessments['first_assessment_day'] = \
            d_assessments['created_at_day'].apply_spans_first(spans)

        t_agg_assessments['last_assessment_day'] = \
            d_assessments['created_at_day'].apply_spans_last(spans)

        print(np.unique(d_assessments['tested_covid_positive'].data[:], return_counts=True))
        t_agg_assessments['max_assessment_test_result'] = \
            d_assessments['tested_covid_positive'].apply_spans_max(spans)

        with utils.Timer("merging aggregated assessment data to patient data"):
            right_fields_to_copy = tuple(k for k in t_agg_assessments.keys() if k != 'patient_id')
            dataframe.merge(d_patients, t_agg_assessments, d_patients,
                            left_on=('id',), right_on=('patient_id',),
                            left_fields=[],
                            right_fields=right_fields_to_copy,
                            how='left')
            d_patients.rename('valid_r', 'has_assessment_entries')

        print('max_assessment_test_result',
              np.unique(d_patients['max_assessment_test_result'].data[:], return_counts=True))


    # generate test measures at the patient level

    if has_patients and has_tests:
        d_patients = dest_dataset['patients']
        d_tests = dest_dataset['tests']
        t_agg_tests = temp_dataset.create_dataframe('aggregated_tests')

        spans = s.get_spans(d_tests['patient_id'])

        t_agg_tests['patient_id'] = \
            d_tests['patient_id'].apply_spans_first(spans)

        t_agg_tests.create_numeric('test_counts', 'int32')
        s.apply_spans_count(spans, t_agg_tests['test_counts'])

        t_agg_tests['max_test_result'] = \
            d_tests['result'].apply_spans_max(spans)

        with utils.Timer("merging aggregated test data to patient data"):
            right_fields_to_copy = tuple(k for k in t_agg_tests.keys() if k != 'patient_id')
            dataframe.merge(d_patients, t_agg_tests, d_patients,
                            left_on=('id',), right_on=('patient_id',),
                            left_fields=[], right_fields=None,
                            how='left')
            d_patients.rename('valid_r', 'has_test_entries')


    # generate vaccine dose measures at the patient level

    if has_patients and has_vaccine_doses:
        d_patients = dest_dataset['patients']
        d_vaccine_doses = dest_dataset['vaccine_doses']
        print("vaccine doses:", d_vaccine_doses.keys())
        t_agg_vacc_doses = temp_dataset.create_dataframe('aggregated_vaccine_doses')

        pid_spans = s.get_spans(d_vaccine_doses['patient_id'])

        t_agg_vacc_doses['patient_id'] = \
            d_vaccine_doses['patient_id'].apply_spans_first(pid_spans)
        # t_agg_vacc_doses['dose_count'] = \
        #     d_vaccine_doses['vaccine_id'].apply_spans_count(pid_spans)
        t_agg_vacc_doses.create_numeric('dose_count', 'int32')
        s.apply_spans_count(pid_spans, t_agg_vacc_doses['dose_count'])

        with utils.Timer("merging aggregated vaccine dose data to patient data"):
            dataframe.merge(d_patients, t_agg_vacc_doses, d_patients,
                            left_on=('id',), right_on=('patient_id',),
                            left_fields=[], right_fields=('dose_count',),
                            how='left')
            d_patients.rename('valid_r', 'has_vaccine_doses')


    # generate vaccine symptom measures at the patient level

    if has_patients and has_vaccine_symptoms:
        d_patients = dest_dataset['patients']
        d_vaccine_symptoms = dest_dataset['vaccine_symptoms']
        print("vaccine symptoms:", d_vaccine_symptoms.keys())
        t_agg_vacc_symptoms = temp_dataset.create_dataframe('aggregated_vaccine_symptoms')

        pid_spans = s.get_spans(d_vaccine_symptoms['patient_id'])

        t_agg_vacc_symptoms['patient_id'] = \
            d_vaccine_symptoms['patient_id'].apply_spans_first(pid_spans)
        t_agg_vacc_symptoms.create_numeric('symptom_count', 'int32')
        s.apply_spans_count(pid_spans, t_agg_vacc_symptoms['symptom_count'])

        with utils.Timer("merging aggregated vaccine symptom data to patient data"):
            dataframe.merge(d_patients, t_agg_vacc_symptoms, d_patients,
                            left_on=('id',), right_on=('patient_id',),
                            left_fields=[], right_fields=('symptom_count',),
                            how='left')
            d_patients.rename('valid_r', 'has_vaccine_symptoms')


    # generate mental health measures at the patient level

    if has_patients and has_mental_health:
        d_patients = dest_dataset['patients']
        d_mental_health = dest_dataset['mental_health']
        print("mental_health:", d_mental_health.keys())
        t_agg_mental_health = temp_dataset.create_dataframe('aggregated_mental_health')

        pid_spans = s.get_spans(d_mental_health['patient_id'])

        t_agg_mental_health['patient_id'] = \
            d_mental_health['patient_id'].apply_spans_first(pid_spans)
        t_agg_mental_health.create_numeric('mental_health_count', 'int32')
        s.apply_spans_count(pid_spans, t_agg_mental_health['mental_health_count'])

        with utils.Timer("merging aggregated mental health data to patient data"):
            dataframe.merge(d_patients, t_agg_mental_health, d_patients,
                            left_on=('id',), right_on=('patient_id',),
                            left_fields=[], right_fields=('mental_health_count',),
                            how='left')
            d_patients.rename('valid_r', 'has_mental_health_entries')
