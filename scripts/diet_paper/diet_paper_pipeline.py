import argparse

from exetera.core.utils import Timer
from exetera.core.session import Session
from exeteracovid.scripts.create_imd_data_map import create_imd_data_map
from exeteracovid.scripts.export_diet_data_to_csv import export_diet_data_to_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True,
                        help='The dataset containing the patient data')
    parser.add_argument('-g', '--geodata', required=True,
                        help='The dataset containing lsoa11cd data')
    parser.add_argument('-p', '--patient_geodata', required=True,
                        help='The dataset that the patient-mapped lsoa11cd data is written to (overwritten)')
    parser.add_argument('-o', '--output', required=True,
                        help='The output dataset that maps patient lsoa11cd codes to geo data (overwritten)')
    parser.add_argument('-c', '--output_csv', required=True,
                        help='The csv file that contains the output data')
    args = parser.parse_args()

    with Session() as s:
        source = s.open_dataset(args.source, 'r', 'source')
        geodata = s.open_dataset(args.geodata, 'r', 'geodata')
        patient_geodata = s.open_dataset(args.patient_geodata, 'w', 'patient_geodata')
        with Timer("Mapping lsoa11cd data to patients", new_line=True):
            create_imd_data_map(s, source, geodata, patient_geodata)

        s.close_dataset('geo')
        output = s.open_dataset(args.output, 'w', 'output')
        with Timer("Exporting diet data", new_line=True):
            export_diet_data_to_csv(s, source, patient_geodata, output, args.output_csv)
        print("Output written to {}".format(args.output_csv))
