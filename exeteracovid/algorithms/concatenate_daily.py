from exetera.core.utils import Timer
from exetera.core import fields as fld

def merge_daily_assessments_v1(s, src_group, dest_group, overrides=None):
    """
    Organize the assessment dataset to group record of patients in each day.

    :param s: The Exetera session instance.
    :param src_group: The source dataframe that contains the dataset.
    :param dest_group: The destination dataframe to write the result to.
    :param overrides: The group function to apply to different columns, e.g. latest datetime for 'updated_at'
        column, or concat for text columns.
    """
    print("generate daily assessments")
    patient_ids_ = s.get(src_group['patient_id']).data[:]
    created_at_days_ = s.get(src_group['created_at_day']).data[:]

    with Timer("calculating spans", new_line=True):
        patient_id_index_spans = s.get_spans(fields=(patient_ids_,
                                                     created_at_days_))

    with Timer("applying spans", new_line=True):
        if overrides is None:
            overrides = {
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

        for k in src_group.keys():
            with Timer("merging '{}'".format(k), new_line=True):
                reader = s.get(src_group[k])
                if k in overrides:
                    fn = overrides[k]
                else:
                    if isinstance(reader, fld.CategoricalField):
                        fn = s.apply_spans_max
                    elif isinstance(reader, fld.IndexedStringField):
                        fn = s.apply_spans_concat
                    elif isinstance(reader, fld.NumericField):
                        fn = s.apply_spans_max
                    else:
                        fn = None

                if fn is None:
                    print("  Skipping field '{k'}")
                else:
                    with Timer("  Merging field '{k}"):
                        fn(patient_id_index_spans, reader, reader.create_like(dest_group, k))
