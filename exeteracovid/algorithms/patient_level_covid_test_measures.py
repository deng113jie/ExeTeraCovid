from exetera.core.session import Session
import exetera.core.operations as ops


def test_counts_per_patient_v1(session: Session,
                            patient_table,
                            test_table,
                            dest_patient_table,
                            dest_patient_name):
    """
    Counting the number of tests performed for each patient id.

    :param session: The Exetera session instance.
    :param patient_table: The patient dataframe.
    :param test_table: The tests dataframe.
    :param dest_patient_table: The destination dataframe to store the results.
    :param dest_patient_name: The name of the destination field to store the results.
    """

    pid = 'id'
    pids = session.get(patient_table[pid])
    pids_ = pids.data[:]
    if not ops.is_ordered(pids.data[:]):
        raise ValueError("The patient table must be ordered by '{}'".format(pid))

    t_pid = 'patient_id'
    t_pids = session.get(test_table[t_pid])
    t_pids_ = t_pids.data[:]
    if not ops.is_ordered(t_pids_):
        raise ValueError("The test table must be ordered by '{}'".format(t_pid))

    # collapse the test data by patient_id and get the counts
    spans_ = session.get_spans(t_pids_)
    s_t_pids_ = session.apply_spans_first(spans_, t_pids_)
    counts_ = session.apply_spans_count(spans_)

    # merge the counts for the test table into the patient table
    dest = session.create_numeric(dest_patient_table, dest_patient_name, 'int32')
    session.ordered_merge_left(left_on=pids_, right_on=s_t_pids_, right_field_sources=(counts_,),
                               left_field_sinks=(dest,), left_unique=True, right_unique=True)


def first_test_date_per_patient(session: Session,
                                patient_table,
                                test_table,
                                test_date_name,
                                dest_patient_table,
                                dest_patient_name):
    """
    Filter the first date of test performed for each patient id.

    :param session: The Exetera session instance.
    :param patient_table: The patient dataframe.
    :param test_table: The tests dataframe.
    :param test_date_name: The name of the test dataframe, not used.
    :param dest_patient_table: The destination dataframe to store the results.
    :param dest_patient_name: The name of the destination field to store the results.
    """

    pid = 'id'
    pids = session.get(patient_table[pid])
    pids_ = pids.data[:]
    if not ops.is_ordered(pids.data[:]):
        raise ValueError("The patient table must be ordered by '{}'".format(pid))

    t_pid = 'patient_id'
    t_pids = session.get(test_table[t_pid])
    t_pids_ = t_pids.data[:]
    if not ops.is_ordered(t_pids_):
        raise ValueError("The test table must be ordered by '{}'".format(t_pid))

    # collapse the test data by patient_id and get the counts
    cats = session.get(test_table['created_at'])
    spans_ = session.get_spans(t_pids_)
    s_t_pids_ = session.apply_spans_first(spans_, t_pids_)
    counts_ = session.apply_spans_first(spans_, cats)

    # merge the counts for the test table into the patient table
    dest = session.create_numeric(dest_patient_table, dest_patient_name, 'int32')
    session.ordered_merge_left(left_on=pids_, right_on=s_t_pids_, right_field_sources=(counts_,),
                               left_field_sinks=(dest,), left_unique=True, right_unique=True)
