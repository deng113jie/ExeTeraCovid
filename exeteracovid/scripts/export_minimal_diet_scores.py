import pandas as pd

from exetera.core.session import Session
from exeteracovid.algorithms import healthy_diet_index as hdi



src_dataset = '/home/ben/covid/ds_20201101_full.hdf5'


with Session() as s:
    src = s.open_dataset(src_dataset, 'r', 'src')
    diet = src['diet']
    field_names = [fk for fk in diet.keys() if 'ffq' in fk]
    field_dict = {fk: s.get(diet[fk]).data[:] for fk in field_names}
    hd_index = hdi.healthy_diet_index(field_dict)
    gut_index, _, _ = hdi.gut_friendly_score(field_dict)

    ids = s.get(diet['id']).data[:]
    pids = s.get(diet['patient_id']).data[:]

    df = pd.DataFrame()
    df['ids'] = ids
    df['patient_ids'] = pids
    df['health_diet_index'] = hd_index
    df['gut_friendly_index'] = gut_index
    df.to_csv('/home/ben/covid/minimal_diet_scores.csv')
