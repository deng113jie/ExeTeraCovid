import unittest

from datetime import datetime as dt
from io import BytesIO

import numpy as np

from exetera.core.session import Session
from exeteracovid.algorithms.covid_test_date import covid_test_date_v1


class TestCovidTestDate(unittest.TestCase):

    def test_covid_test_date_v1_positive_test(self):
        bio = BytesIO()
        with Session() as s:
            # t_pids_ = np.asarray([b'a', b'a', b'b', b'b', b'b', b'c', b'c', b'e',
            #                       b'e', b'f', b'g', b'h', b'i', b'i'])
            t_dates_exact = np.asarray([0.0,
                                        dt(2020, 10, 12).timestamp(),
                                        dt(2020, 6, 2).timestamp(),
                                        0.0,
                                        dt(2021, 1, 30).timestamp(),
                                        0.0, # 5
                                        0.0,
                                        0.0,
                                        dt(2020, 8, 10).timestamp(), # 8
                                        dt(2020, 12, 1).timestamp(),
                                        dt(2020, 9, 2).timestamp(),
                                        0.0 # 11
                                        ])
            t_dates_from = np.asarray([dt(2020, 5, 12).timestamp(),
                                       0.0,
                                       0.0,
                                       dt(2020, 9, 2).timestamp(),
                                       0.0,
                                       0.0, # 5
                                       dt(2020, 7, 16).timestamp(),
                                       0.0,
                                       dt(2021, 8, 8).timestamp(), # 8
                                       0.0,
                                       dt(2020, 8, 10).timestamp(),
                                       dt(2020, 11, 4).timestamp(), # 11
                                       ])
            t_dates_to = np.asarray([dt(2020, 5, 17).timestamp(),
                                     0.0,
                                     0.0,
                                     dt(2020, 9, 3).timestamp(),
                                     0.0,
                                     0.0, # 5
                                     0.0,
                                     dt(2020, 6, 20).timestamp(),
                                     0.0, # 8
                                     dt(2020, 12, 19).timestamp(),
                                     dt(2020, 10, 5).timestamp(),
                                     dt(2020, 11, 3).timestamp() # 11
                                     ])

            print(dt(2020, 11, 3).timestamp())

            src = s.open_dataset(bio, 'w', 'src')
            tests = src.create_group('tests')
            s.create_timestamp(tests, 'date_taken_specific').data.write(t_dates_exact)
            s.create_timestamp(tests, 'date_taken_between_start').data.write(t_dates_from)
            s.create_timestamp(tests, 'date_taken_between_end').data.write(t_dates_to)

            covid_test_date_v1(s, tests, tests)
            print(s.get(tests['test_date']).data[:])
            print(s.get(tests['test_date_valid']).data[:])
