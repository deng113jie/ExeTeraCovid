import unittest
import numpy as np

from io import BytesIO

from exetera.core.session import Session
from exeteracovid.algorithms import patient_level_covid_test_measures as alg


class TestTestCountsPerPatient(unittest.TestCase):

    def test_test_counts_per_patient_v1_positive_test(self):
        bio = BytesIO()
        with Session() as s:
            pids_ = np.asarray([b'b', b'c', b'd', b'f', b'h', b'i'])
            t_pids_ = np.asarray([b'a', b'a', b'b', b'b', b'b', b'c', b'c', b'e',
                                  b'e', b'f', b'g', b'h', b'i', b'i'])
            src = s.open_dataset(bio, 'w', 'src')
            ptnts = src.create_group('patients')
            tests = src.create_group('tests')
            s.create_fixed_string(ptnts, 'id', 1).data.write(pids_)
            s.create_fixed_string(tests, 'patient_id', 1).data.write(t_pids_)
            alg.test_counts_per_patient_v1(s, ptnts, tests, ptnts, 'counts')
            counts_ = s.get(ptnts['counts']).data[:]
            self.assertListEqual([3, 2, 0, 1, 1, 2], counts_.tolist())
