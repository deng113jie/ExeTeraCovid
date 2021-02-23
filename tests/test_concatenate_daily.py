import unittest

from io import BytesIO

import numpy as np

from exetera.core.session import Session
from exeteracovid.algorithms.concatenate_daily import merge_daily_assessments_v1


class TestConcatenateDaily(unittest.TestCase):

    def test_concatenate_daily(self):

        ids = np.asarray(['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c'],
                         dtype='S1')
        days = np.asarray(['2020-05-06', '2020-05-06', '2020-05-07', '2020-06-02', '2020-06-02',
                           '2020-08-01', '2020-08-20', '2020-09-05',
                           '2020-04-10', '2020-04-11', '2020-04-11', '2020-04-11', '2020-04-11',
                           '2020-04-11', '2020-04-11'], dtype='S10')
        idf = ['a', "'b'", 'what', 'some, information', 'x',
               '', 'foo', 'flop',
               "'dun'", "'mun'", "'race, track?'", '', "for, too", 'z', 'now!']
        nums = np.asarray([5, 6, 3, 2, 1, 10, 230, 3, 5, -20, -4, 2, 6, 100, 40], dtype=np.int32)

        bio = BytesIO()
        with Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            src = ds.create_group('src')
            ids_f = s.create_fixed_string(src, 'patient_id', 1)
            ids_f.data.write(ids)
            days_f = s.create_fixed_string(src, 'created_at_day', 10)
            days_f.data.write(days)
            idf_f = s.create_indexed_string(src, 'idf')
            idf_f.data.write(idf)
            nums_f = s.create_numeric(src, 'nums', 'int32')
            nums_f.data.write(nums)

            dest = ds.create_group('dest')

            merge_daily_assessments_v1(s, src, dest)
            print(dest.keys())

            print(s.get(dest['idf']).data[:])
