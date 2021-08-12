#!/usr/bin/env python

# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from datetime import datetime, timezone
import os
import sys
import h5py

try:
    import exetera
except ModuleNotFoundError:
    fixed_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(fixed_path)
    import exetera

from exeteracovid.processing import postprocess
from exetera.core.session import Session

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version=exetera.__version__)

parser.add_argument('-i', '--input', required=True, help='The dataset to load')
parser.add_argument('-o', '--output', required=True,
                    help='The dataset to write results to. If this is an existing file, it will be'
                                     ' overwritten')
parser.add_argument('-t', '--temp', required=True,
                    help="A temporary file that holds intermediate data to be deleted.")
parser.add_argument('-d', '--daily', action='store_true',
                    help='If set, generate daily assessments from assessments')

parser.add_argument('-a', '--algorithm_version', required=False,
                    default='2',
                    help="The version number for the processing pipeline. Use version '1' if you "
                         "need to generate daily assessments. We recommend version '2' otherwise, "
                         "which is the default")

args = parser.parse_args()

if 'dev' in exetera.__version__:
    msg = ("Warning: this is a development version of exetera ({}). " 
           "Please use one of the release versions for actual work")
    print(msg.format(exetera.__version__))


timestamp = str(datetime.now(timezone.utc))

flags = set()
if args.daily is True:
    flags.add('daily')
print(exetera.__version__)

if args.algorithm_version == '1':
    with Session() as s:
        ds = s.open_dataset(args.input, 'r', 'ds')
        ts = s.open_dataset(args.output, 'w', 'ts')
        postprocess.postprocess_v1(s, ds, ts, timestamp, flags)
elif args.algorithm_version == '2':
    if args.daily is True:
        print("-d/--daily is not supported in version 2 of the standard processing pipeline")
    with Session() as s:
        postprocess.postprocess_v2(s, args.input, args.temp, args.output, flags)
