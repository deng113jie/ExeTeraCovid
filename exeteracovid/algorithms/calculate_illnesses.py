import numpy as np
from numba import njit

from exeteracovid.operations.nd_array_utils import resize_if_required, truncate_if_required

@njit
def calculate_test_based_illnesses_v1(tspans, test_pids, test_date, test_type, test_result,
                                      max_gap_days=14, min_illness_len=14):
    """
    Calculate illnesses based on positive covid tests of various types.

    Status: alpha

    :param tspans:
    :param test_pids:
    :param test_date:
    :param test_type:
    :param test_result:
    :param max_gap_days:
    :param min_illness_len:
    :return:
    """

    illness_pids = np.zeros(0, dtype='S32')
    illness_starts = np.zeros(0, dtype='float64')
    illness_ends = np.zeros(0, dtype='float64')

    resize_delta = 100000
    total_illness_count = 0
    max_gap_threshold = max_gap_days * 86400
    min_illness_threshold = min_illness_len * 86400

    for i in range(len(tspans) - 1):
        tstart, tend = tspans[i], tspans[i + 1]
        healthy = True
        start_date = 0
        end_date = 0
        last_unhealthy_index = -1
        first_healthy_after_unhealthy_index = -1

        for t in range(tstart, tend + 1):
            transition_to_healthy = False
            transition_to_unhealthy = False
            if t == tend:
                if not healthy:
                    transition_to_healthy = True
            else:
                if healthy:
                    if test_result[t] == 4:  # +ve result
                        transition_to_unhealthy = True
                else:
                    if test_result[t] == 3:  # -ve result
                        if first_healthy_after_unhealthy_index == -1:
                            first_healthy_after_unhealthy_index = t
                        if test_date[t] - test_date[last_unhealthy_index] > max_gap_threshold:
                            transition_to_healthy = True
                    elif test_result[t] == 4:  # +ve result
                        first_healthy_after_unhealthy_index = -1

                        if test_date[t] - test_date[t - 1] > max_gap_threshold:
                            transition_to_healthy = True
                            transition_to_unhealthy = True
                        else:
                            last_unhealthy_index = t

            if transition_to_healthy:
                if first_healthy_after_unhealthy_index != -1:
                    end_index = first_healthy_after_unhealthy_index
                else:
                    end_index = t

                if end_index == tend:
                    end_date = test_date[last_unhealthy_index] + max_gap_threshold
                else:
                    end_date = min(test_date[end_index],
                                   test_date[last_unhealthy_index] + max_gap_threshold)

                if end_date - start_date >= min_illness_threshold:
                    illness_pids = resize_if_required(illness_pids,
                                                      total_illness_count,
                                                      resize_delta)
                    illness_pids[total_illness_count] = test_pids[tstart]
                    illness_starts = resize_if_required(illness_starts,
                                                        total_illness_count,
                                                        resize_delta)
                    illness_starts[total_illness_count] = start_date
                    illness_ends = resize_if_required(illness_ends,
                                                      total_illness_count,
                                                      resize_delta)
                    illness_ends[total_illness_count] = end_date
                    total_illness_count += 1
                healthy = True

            if transition_to_unhealthy:
                healthy = False
                start_date = test_date[t]
                start_index = t
                last_unhealthy_index = t
                first_healthy_after_unhealthy_index = -1

    return (truncate_if_required(illness_pids, total_illness_count),
            truncate_if_required(illness_starts, total_illness_count),
            truncate_if_required(illness_ends, total_illness_count))