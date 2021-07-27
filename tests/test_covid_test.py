from io import BytesIO
import unittest
from datetime import datetime

import exetera.core.session as esess
from exeteracovid.algorithms.covid_test import match_assessment

class TestCovidTest(unittest.TestCase):
    def test_match_assessment(self):
        bio = BytesIO()
        with esess.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            # test df
            tests = src.create_dataframe('tests')
            pid = tests.create_numeric('patient_id', 'int32')
            pid.data.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            d = tests.create_timestamp('created_at')
            d.data.write([datetime(2020, 1, i).timestamp() for i in range(5, 15)])
            pid = tests.create_numeric('result', 'int32')
            pid.data.write([3, 4, 3, 4, 3, 4, 3, 4, 3, 4])

            #assessment df
            asmt = src.create_dataframe('assessments')
            pid = asmt.create_numeric('patient_id', 'int32')
            pid.data.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            d = asmt.create_timestamp('created_at')
            d.data.write([datetime(2020, 1, i).timestamp() for i in list(reversed(range(7, 17)))])

            result = src.create_dataframe('result')
            match_assessment(tests, asmt, result, 5)
            self.assertListEqual(result['patient_id_l'].data[:].tolist(), list([7, 8, 9]))
            result = src.create_dataframe('result2')
            match_assessment(tests, asmt, result, 5, True)
            self.assertListEqual(result['patient_id_l'].data[:].tolist(), list([8]))



