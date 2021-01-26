import numpy as np


def create_imd_data_map(s, src, lsoa, dest):
    s_ptnts = src['patients']
    p_ids = s.get(s_ptnts['id']).data[:]
    p_lsoas = s.get(s_ptnts['lsoa11cd']).data[:]
    l_lsoa_table = lsoa['lsoa11cd']
    l_lsoas = s.get(l_lsoa_table['lsoa11cd']).data[:]

    d_lsoa_indices = dict()
    for i_v, v in enumerate(l_lsoas):
        d_lsoa_indices[v] = i_v

    p_lsoa_filter = np.zeros(len(p_lsoas), dtype=np.bool)
    p_lsoa_indices = np.full(len(p_lsoas), np.iinfo(np.int32).min, dtype=np.int32)
    for i_p, p in enumerate(p_lsoas):
        if p in d_lsoa_indices:
            p_lsoa_filter[i_p] = True
            p_lsoa_indices[i_p] = d_lsoa_indices[p]

    result_dict = dict()
    for k in ('imd_rank', 'imd_decile', 'ruc11cd', 'townsend_score', 'townsend_quintile'):
        source = s.get(l_lsoa_table[k]).data[:]
        results = np.zeros_like(p_lsoa_indices, dtype=source.dtype)

        for i in range(len(p_lsoa_filter)):
            if p_lsoa_filter[i]:
                results[i] = source[p_lsoa_indices[i]]
        result_dict[k] = results

    d_ptnts = dest.create_group("patients")
    s.get(s_ptnts['id']).create_like(d_ptnts, 'id').data.write(p_ids)
    s.get(s_ptnts['lsoa11cd']).create_like(d_ptnts, 'lsoa11cd').data.write(p_lsoas)
    s.create_numeric(d_ptnts, 'has_imd_data', 'bool').data.write(p_lsoa_filter)
    for k, v in result_dict.items():
        s.get(l_lsoa_table[k]).create_like(d_ptnts, k).data.write(v)
