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

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version=exetera.__version__)

parser.add_argument('-i', '--input', required=True, help='The dataset to load')
parser.add_argument('-o', '--output', required=True,
                    help='The dataset to write results to. If this is an existing file, it will be'
                                     ' overwritten')
parser.add_argument('-d', '--daily', action='store_true',
                    help='If set, generate daily assessments from assessments')

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
with h5py.File(args.input, 'r') as ds:
    with h5py.File(args.output, 'w') as ts:
        postprocess.postprocess(ds, ts, timestamp, flags)
