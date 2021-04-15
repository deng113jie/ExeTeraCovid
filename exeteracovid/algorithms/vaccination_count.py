from typing import Union

from collections import defaultdict

import numpy as np

from exetera.core.fields import Field
from exetera.core.session import Session
from exetera.core.validation import array_from_parameter


def vaccination_count_unsorted(session: Session,
                      patient_ids: Union[Field, np.ndarray],
                      vacc_patient_ids: Union[Field, np.ndarray]):
    d = defaultdict(int)

    pids_ = array_from_parameter(session, "patient_ids", patient_ids)
    vpids_ = array_from_parameter(session, "vacc_patient_ids", vacc_patient_ids)

    for v in vpids_:
        d[v] += 1

    counts = np.ndarray(len(pids_), dtype=np.int32)
    for i_p in range(len(pids_)):
        counts[i_p] = d[pids_[i_p]]

    return counts


def vaccination_count_sorted(session: Session,
                             patient_ids: Union[Field ,np.ndarray],
                             vacc_patient_ids: Union[Field, np.ndarray]):

    vpids_ = array_from_parameter(vacc_patient_ids)
    spans = session.get_spans(vpids_)
    svpids_ = session.apply_spans_first(spans, vpids_)
    svpid_counts_ = session.apply_spans_count(spans)

    pid_counts_ = session.merge_left(left_on=patient_ids,
                                     right_on=svpids_,
                                     right_fields=(svpid_counts_,))
    return pid_counts_
