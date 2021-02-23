import numpy as np

from exetera.core.session import Session


def covid_test_date_v1(session: Session,
                       test_table,
                       dest_test_table,
                       dest_field_name='test_date',
                       dest_field_flags_name='test_date_valid'):

    exact = session.get(test_table['date_taken_specific'])
    exact_ = exact.data[:]
    between_start_ = session.get(test_table['date_taken_between_start']).data[:]
    between_end_ = session.get(test_table['date_taken_between_end']).data[:]

    # flag dates where neither exact or between_start are set
    test_date_valid = (exact_ == 0.0) & (between_start_ != 0.0) & (between_end_ != 0.0) &\
                      (between_end_ >= between_start_)
    test_date_valid = test_date_valid | ((exact_ != 0.0) & (between_start_ == 0.0) & (between_end_ == 0.0))

    test_date = np.where(exact_ != 0.0, exact_, between_start_ + (between_end_ - between_start_) / 2)

    exact.create_like(dest_test_table, dest_field_name).data.write(test_date)
    session.create_numeric(dest_test_table, dest_field_flags_name,
                           'bool').data.write(test_date_valid)
