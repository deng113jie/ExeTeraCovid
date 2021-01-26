#!/usr/bin/env python

import argparse
from exetera.core.utils import Timer
from exetera.core.session import Session
from exeteracovid.scripts.export_diet_data_to_csv import export_diet_data_to_csv



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='The dataset containing the patient and diet data')
    parser.add_argument('-g', '--geodata', required=True, help='The dataset containing patient-level geocode data')
    parser.add_argument('-o', '--output', required=True, help='The output dataset that contains the output data')
    parser.add_argument('-c', '--csvoutput', required=True, help='The csv file that contains the output data')
    args = parser.parse_args()
    with Session() as s:
        src = s.open_dataset(args.source, 'r', 'src')
        geo = s.open_dataset(args.geodata, 'r', 'geo')
        output = s.open_dataset(args.output, 'w', 'output')
        with Timer("Running export_diet_data_to_csv", new_line=True):
            export_diet_data_to_csv(s, src, geo, output, args.csvoutput)
