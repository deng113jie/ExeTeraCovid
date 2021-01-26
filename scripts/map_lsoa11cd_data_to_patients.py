#!/usr/bin/env python

import argparse
from exetera.core.utils import Timer
from exetera.core.session import Session
from exeteracovid.scripts.create_imd_data_map import create_imd_data_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='The dataset containing the patient data')
    parser.add_argument('-g', '--geodata', required=True,  help='The dataset containing lsoa11cd data')
    parser.add_argument('-o', '--output', required=True,  help='The output dataset that maps patient lsoa11cd codes to geo data')
    args = parser.parse_args()

    with Session() as s:
        src = s.open_dataset(args.source, 'r', 'src')
        lsoa = s.open_dataset(args.geodata, 'r', 'lsoa')
        dest = s.open_dataset(args.output, 'w', 'dest')
        with Timer("Running create_imd_data_map", new_line=True):
            create_imd_data_map(s, src, lsoa, dest)
