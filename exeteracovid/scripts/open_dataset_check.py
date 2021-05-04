from datetime import datetime, timedelta

filename = '/home/ben/covid/ds_20210331_full.hdf5'
tmp_filename = '/home/ben/covid/ds_telemetry_tmp.hdf5'
start_dt = datetime(2020, 3, 1)
end_dt = datetime(2021, 3, 1)

import math

import numpy as np
from numba import njit

from exetera.core.session import Session
from exetera.core.utils import Timer
from exetera.processing.date_time_helpers import \
    get_periods, generate_period_offset_map, get_days, get_period_offsets

@njit
def count_combinations(array1, array2, counts):
    for i in range(len(array1)):
        counts[array1[i], array2[i]] += 1

def human_readible_date(date):
    if isinstance(date, float):
        date = datetime.fromtimestamp(date)
    return date.strftime("%Y/%m/%d")

start_ts = start_dt.timestamp()
end_ts = end_dt.timestamp()
periods = get_periods(end_dt, start_dt, 'week', -1)
periods.reverse()
print("Weekly periods from {} to {}".format(human_readible_date(periods[0]),
                                            human_readible_date(periods[-1])))

with Session() as s:
    src = s.open_dataset(filename, 'r', 'src')
    s_ptnts = src['patients']

    # the timestamps of each user signup
    pcats_ = s.get(s_ptnts['created_at']).data[:]

    # calculate on what day (relative to the start of the first period) each user signed up.
    days, inrange = get_days(pcats_,
                             start_date=periods[0].timestamp(),
                             end_date=periods[-1].timestamp())

    # note that some users will have signed up on a day outside of the periods that we defined
    # so we need to clear those
    days = np.where(inrange, days, 0)

    # generate a period offset map mapping days to periods, and look up the corresponding period
    # for each user's signup date
    cat_period = get_period_offsets(generate_period_offset_map(periods), days)