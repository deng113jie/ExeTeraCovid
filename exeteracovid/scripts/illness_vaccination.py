import datetime
import time

import numpy as np
from numba import njit

from exetera.core.session import Session

from exeteracovid.operations.nd_array_utils import resize_if_required, truncate_if_required
from exeteracovid.algorithms.calculate_illnesses import calculate_test_based_illnesses_v1


@njit
def is_sorted(arr):
    for i in range(1, len(arr)):
        if arr[i - 1] > arr[i]:
            return False
    return True


def pretty_print(df, row_counts, keys=None):
    if keys is None:
        keys = df.keys()
    for ik, k in enumerate(keys):
        if ik > 0:
            print(' ', end='')
        print(k, end='')
    print()

    if not isinstance(row_counts, list):
        row_counts = list(row_counts)
    for r in row_counts:
        for i in range(r[0], r[1]):
            print(i, end=' ')
            for ik, k in enumerate(keys):
                print(df[k].data[i], end=' ')
            print()


@njit
def calculate_prior_vaccine_deltas_v1(ispans,
                                      ipids,
                                      istarts,
                                      iends,
                                      vspans,
                                      vpids,
                                      vids,
                                      vdates):

    rpids = np.zeros(0, dtype='S32')
    rvids = np.zeros(0, dtype='S32')
    rvcounts = np.zeros(0, dtype='int32')
    rvdates = np.zeros(0, dtype='float64')
    ristarts = np.zeros(0, dtype='float64')
    riends = np.zeros(0, dtype='float64')
    rvdeltas = np.zeros(0, dtype='float64')

    total_illness_count = 0
    delta = 100000

    si = 0
    sv = 0
    while si < len(ispans) - 1 and sv < len(vspans) - 1:
        istart, iend = ispans[si], ispans[si + 1]
        vstart, vend = vspans[sv], vspans[sv + 1]
        ipid = ipids[istart]
        vpid = vpids[vstart]

        if ipid < vpid:
            # add illnesses with no vaccine data
            for i in range(istart, iend):
                rpids = resize_if_required(rpids, total_illness_count, delta)
                rvids = resize_if_required(rvids, total_illness_count, delta)
                rvcounts = resize_if_required(rvcounts, total_illness_count, delta)
                rvdates = resize_if_required(rvdates, total_illness_count, delta)
                ristarts = resize_if_required(ristarts, total_illness_count, delta)
                riends = resize_if_required(riends, total_illness_count, delta)
                rvdeltas = resize_if_required(rvdeltas, total_illness_count, delta)
                rpids[total_illness_count] = ipid
                rvcounts[total_illness_count] = 0
                ristarts[total_illness_count] = istarts[i]
                total_illness_count += 1

            si += 1
        elif ipid > vpid:

            sv += 1
        else:

            i = istart
            v = vstart

            v_latest = -1
            v_count = 0

            while i < iend and v < vend:

                i_date = istarts[i]
                v_date = vdates[v]
                if i_date < v_date:
                    # add the illness and associated prior vaccination data
                    rpids = resize_if_required(rpids, total_illness_count, delta)
                    rvids = resize_if_required(rvids, total_illness_count, delta)
                    rvcounts = resize_if_required(rvcounts, total_illness_count, delta)
                    rvdates = resize_if_required(rvdates, total_illness_count, delta)
                    ristarts = resize_if_required(ristarts, total_illness_count, delta)
                    riends = resize_if_required(riends, total_illness_count, delta)
                    rvdeltas = resize_if_required(rvdeltas, total_illness_count, delta)
                    rpids[total_illness_count] = ipid
                    rvcounts[total_illness_count] = v_count
                    ristarts[total_illness_count] = i_date
                    riends[total_illness_count] = iends[i]
                    if v_count > 0:
                        rvids[total_illness_count] = vids[v_latest]
                        rvdates[total_illness_count] = vdates[v_latest]
                        rvdeltas[total_illness_count] = (i_date - vdates[v_latest]) / 86400
                    total_illness_count += 1
                    i += 1
                else:
                    # update the prior vaccination
                    v_latest = v
                    v_count += 1
                    v += 1

            # handle remaining illnesses
            while i < iend:
                i_date = istarts[i]

                # update the prior vaccination
                rpids = resize_if_required(rpids, total_illness_count, delta)
                rvids = resize_if_required(rvids, total_illness_count, delta)
                rvcounts = resize_if_required(rvcounts, total_illness_count, delta)
                rvdates = resize_if_required(rvdates, total_illness_count, delta)
                ristarts = resize_if_required(ristarts, total_illness_count, delta)
                riends = resize_if_required(riends, total_illness_count, delta)
                rvdeltas = resize_if_required(rvdeltas, total_illness_count, delta)
                ristarts[total_illness_count] = i_date
                rpids[total_illness_count] = ipid
                if v_count > 0:
                    rvids[total_illness_count] = vids[v_latest]
                    rvdates[total_illness_count] = vdates[v_latest]
                    rvdeltas[total_illness_count] = (i_date - vdates[v_latest]) / 86400
                rvcounts[total_illness_count] = v_count
                total_illness_count += 1
                i += 1

            si += 1
            sv += 1

    return (truncate_if_required(rpids, total_illness_count),
            truncate_if_required(rvids, total_illness_count),
            truncate_if_required(rvcounts, total_illness_count),
            truncate_if_required(rvdates, total_illness_count),
            truncate_if_required(ristarts, total_illness_count),
            truncate_if_required(riends, total_illness_count),
            truncate_if_required(rvdeltas, total_illness_count))


def calculate_illness_vaccination_table_v1(src_filename, dst_filename, reprocess):
    """
    This function calculates a table of covid-based illness relative to vaccination
    doses.

    Status: alpha
    :param src_filename: the source dataset from which to get patient, test and vaccine data
    :param dst_filename: the destination dataset to which illnesses relative to vaccations is written
    :param reprocess: If True, regenerate the dataset from scratch
    :return: None
    """
    with Session() as s:

        src_ds = s.open_dataset(src_filename, 'r', 'src')
        if reprocess:
            open_mode = 'w'
        else:
            open_mode = 'r+'
        dst_ds = s.open_dataset(dst_filename, open_mode, 'dst')
        s_tests = src_ds['tests']
        s_vaccds = src_ds['vaccine_doses']

        if 'tests' not in dst_ds:
            print("Sorting and filtering tests")

            # sort the tests by test date if they aren't already sorted
            dst_tests = dst_ds.create_dataframe('tests')
            idx = s.dataset_sort_index((s_tests['patient_id'], s_tests['effective_test_date']))
            s_tests.apply_index(idx, dst_tests)

            # filter out all non-pcr tests
            # TODO: filter out rapid tests using the 'israpid' column
            mechanism = dst_tests['mechanism'].data[:]
            is_pcr = np.where(((mechanism >= 1) & (mechanism < 5)) | (mechanism == 8), True, False)
            print("mechanism pre filter:", len(dst_tests['id']))
            dst_tests.apply_filter(is_pcr)
            print("mechanism post filter:", len(dst_tests['id']))
        else:
            dst_tests = dst_ds['tests']


        # calculate illnesses by covid positive test
        # ---------
        if 'illnesses' not in dst_ds:
            print("Calculate illnesses")
            d_ills = dst_ds.create_dataframe('illnesses')

            tpids_ = dst_tests['patient_id'].data[:]
            tspans = dst_tests['patient_id'].get_spans()
            tdates_ = dst_tests['effective_test_date'].data[:]
            ttypes_ = dst_tests['mechanism'].data[:]
            tresults_ = dst_tests['result'].data[:]

            illness_pids, illness_starts, illness_ends = \
                calculate_test_based_illnesses_v1(tspans, tpids_, tdates_, ttypes_, tresults_,
                                                  max_gap_days=14, min_illness_len=14)
            d_ills.create_fixed_string('patient_id', 32).data.write(illness_pids)
            d_ills.create_numeric('illness_starts', 'int64').data.write(illness_starts)
            d_ills.create_numeric('illness_ends', 'int64').data.write(illness_ends)
            print("Distinct test based illnesses:", len(illness_pids))
        else:
            d_ills = dst_ds['illnesses']

        if 'illness_vacc' not in dst_ds:
            print("Calculating deltas between illnesses and prior vaccinations")
            d_ill_vacc = dst_ds.create_dataframe('illness_vacc')

            # calculate delta between each illness and the vaccine dose prior to it
            # ---------
            ispans = d_ills['patient_id'].get_spans()
            ipids_ = d_ills['patient_id'].data[:]
            istarts_ = d_ills['illness_starts'].data[:]
            iends_ = d_ills['illness_ends'].data[:]
            vspans = s_vaccds['patient_id'].get_spans()
            vpids_ = s_vaccds['patient_id'].data[:]
            vids_ = s_vaccds['id'].data[:]
            vdates_ = s_vaccds['date_taken_specific'].data[:]
            rpids, rvids, rvcounts, rvdates, ristarts, riends, rvdeltas =\
                calculate_prior_vaccine_deltas_v1(ispans, ipids_, istarts_, iends_,
                                               vspans, vpids_, vids_, vdates_)

            d_ill_vacc.create_fixed_string('patient_id', 32).data.write(rpids)
            d_ill_vacc.create_fixed_string('vaccine_id', 32).data.write(rvids)
            d_ill_vacc.create_numeric('dose_number', 'int32').data.write(rvcounts)
            d_ill_vacc.create_numeric('illness_start', 'float64').data.write(ristarts)
            d_ill_vacc.create_numeric('illness_end', 'float64').data.write(riends)
            d_ill_vacc.create_numeric('time_since_vaccine', 'float64').data.write(rvdeltas)

            print("Illnesses and vaccine records", len(d_ill_vacc['patient_id']))
        else:
            d_ill_vacc = dst_ds['illness_vacc']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True, help="the path / name of the source dataset")
    parser.add_argument('-d', required=True, help="the path / name of the destination dataset")
    parser.add_argument('-f', action='store_true', help="force reprocessing")

    args = parser.parse_args()

    calculate_illness_vaccination_table_v1(src_filename=args.s,
                                           dst_filename=args.d,
                                           reprocess=args.f)
