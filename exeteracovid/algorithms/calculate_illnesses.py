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


@njit
def count_illnesses_v1(aspans, adates, ahealthy, asymptoms, alocation,  # atreatment,
                      t_pids, tspans, tdates, tresult,
                      pos_illness_counts, neg_illness_counts,
                      threshold_days, illness_length_threshold_days,
                      t_illness_pids, t_illness_start, t_illness_end,
                      t_illness_symptoms, t_illness_hospitalised, t_illness_covid,
                      verbose=False):
    """
    Calculate illnesses based on tests and assessment-based symptoms.

    Status: alpha

    :param aspans:
    :param adates:
    :param ahealthy:
    :param asymptoms:
    :param alocation:
    :param t_pids:
    :param tspans:
    :param tdates:
    :param tresult:
    :param pos_illness_counts:
    :param neg_illness_counts:
    :param threshold_days:
    :param illness_length_threshold_days:
    :param t_illness_pids:
    :param t_illness_start:
    :param t_illness_end:
    :param t_illness_symptoms:
    :param t_illness_hospitalised:
    :param t_illness_covid:
    :param verbose:
    :return:
    """
    resize_delta = 100000
    threshold = 86400 * threshold_days
    illness_length_threshold = 86400 * illness_length_threshold_days
    total_illness_count = 0

    for i in range(len(aspans) - 1):
        # if i % 100000 == 0:
        #     print(i)
        astart, aend = aspans[i], aspans[i + 1]
        tstart, tend = tspans[i], tspans[i + 1]

        healthy = True
        # illness_count = 0
        start_date = 0
        # cur_end_date = 0
        start_index = -1
        # end_index = -1
        last_unhealthy_index = -1
        first_healthy_after_unhealthy_index = -1
        hospitalised = False
        symptoms = np.uint32(0)
        # iterate to one past the end so that we can handle tying up an open period of
        # unhealthiness

        pos_illness_count = 0
        neg_illness_count = 0

        if verbose:
            first_date = adates[astart]
            for a in range(astart, aend):
                print(a, (adates[a] - first_date) / 86400, ahealthy[a])

        for a in range(astart, aend + 1):
            transition_to_healthy = False
            transition_to_unhealthy = False
            if a == aend:
                # handle wrapping up an open period of unhealthiness
                if not healthy:
                    transition_to_healthy = True
            else:
                if healthy:
                    if ahealthy[a] == 1:
                        # healthy status and unhealthy record so transition to unhealthy status
                        transition_to_unhealthy = True
                else:
                    if ahealthy[a] == 0:
                        # unhealthy status and healthy record:
                        # . in this case, track the duration of healthy assessments
                        #   . if it passes the threshold, end the period of ill health at the
                        #     first healthy assessment
                        #   . if it is too short, ignore the intervening unhealthy records
                        if first_healthy_after_unhealthy_index == -1:
                            first_healthy_after_unhealthy_index = a
                        if adates[a] - adates[last_unhealthy_index] > threshold:
                            transition_to_healthy = True
                    elif ahealthy[a] == 1:
                        # unhealthy status and unhealthy record:

                        # check if the gap is sufficiently large that it counts as a new period of unhealthiness

                        if adates[a] - adates[a - 1] > threshold:
                            transition_to_healthy = True
                            transition_to_unhealthy = True
                        else:
                            # reset any intervening healthy index to void any intervening healthy records
                            first_healthy_after_unhealthy_index = -1
                            last_unhealthy_index = a
                            hospitalised = hospitalised or (alocation[a] == 2 or alocation[a] == 3)
                            symptoms = symptoms | asymptoms[a]

            if transition_to_healthy:
                # first check whether we are transitioning to healthy because we have a long
                # enough healthy period; in this case we go back to the unhealthy -> healthy
                # transition assessment
                if first_healthy_after_unhealthy_index != -1:
                    end_index = first_healthy_after_unhealthy_index
                else:
                    end_index = a

                if end_index == aend:
                    # true end date is unknown; count as last unhealthy entry + threshold days
                    end_date = adates[last_unhealthy_index - 1] + threshold
                else:
                    end_date = min(adates[end_index], adates[last_unhealthy_index] + threshold)

                if verbose:
                    print(start_index, last_unhealthy_index, start_date / 86400, end_date / 86400,
                          (end_date - start_date) / 86400)

                # record the illness
                if end_date - start_date >= illness_length_threshold:
                    # check whether there is a positive pcr test within 7 days of the illness start
                    test_result = 0
                    for t in range(tstart, tend):
                        td = tdates[t]
                        # TODO: pull this parameter out to be varied
                        if td != 0.0 and td >= start_date - (86400 * 5) and td <= end_date - (86400 * 5):
                            if tresult[t] == 3:
                                test_result = 1
                            if tresult[t] == 4:
                                test_result = 2
                                break

                    t_illness_pids = resize_if_required(t_illness_pids, total_illness_count,
                                                        resize_delta)
                    t_illness_pids[total_illness_count] = t_pids[tstart]

                    t_illness_start = resize_if_required(t_illness_start, total_illness_count,
                                                         resize_delta)
                    t_illness_start[total_illness_count] = start_date

                    t_illness_end = resize_if_required(t_illness_end, total_illness_count,
                                                       resize_delta)
                    t_illness_end[total_illness_count] = end_date

                    t_illness_symptoms = resize_if_required(t_illness_symptoms, total_illness_count,
                                                            resize_delta)
                    t_illness_symptoms[total_illness_count] = symptoms

                    t_illness_hospitalised = resize_if_required(t_illness_hospitalised, total_illness_count,
                                                                resize_delta)
                    t_illness_hospitalised[total_illness_count] = hospitalised

                    t_illness_covid = resize_if_required(t_illness_covid, total_illness_count,
                                                         resize_delta)
                    t_illness_covid[total_illness_count] = test_result
                    total_illness_count += 1

                    if test_result == 2:
                        pos_illness_count += 1
                    else:
                        neg_illness_count += 1

                # reset to healthy
                healthy = True
                start_date = 0
                start_index = -1
                last_unhealthy_index = -1
                # cur_end_date = 0
                # end_index = -1
                first_healthy_after_unhealthy_index = -1
                hospitalised = False
                symptoms = np.int32(0)

            if transition_to_unhealthy:
                healthy = False
                start_date = adates[a]
                start_index = a
                last_unhealthy_index = a
                first_healthy_after_unhealthy = -1
                hospitalised = alocation[a] == 2 or alocation[a] == 3
                symptoms = symptoms | asymptoms[a]

        pos_illness_counts[i] = pos_illness_count
        neg_illness_counts[i] = neg_illness_count

    return (truncate_if_required(t_illness_pids, total_illness_count),
            truncate_if_required(t_illness_start, total_illness_count),
            truncate_if_required(t_illness_end, total_illness_count),
            truncate_if_required(t_illness_symptoms, total_illness_count),
            truncate_if_required(t_illness_hospitalised, total_illness_count),
            truncate_if_required(t_illness_covid, total_illness_count))
