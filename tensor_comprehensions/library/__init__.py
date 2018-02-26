# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

__all__ = ['database']

import os
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

layers_file = os.path.join(os.path.dirname(__file__), 'layers.yaml')


def process_db(database):
    db = {}
    for entry in database:
        name = entry['name']
        db[name] = entry
    return db


def get_tc_database():
    database = None
    with open(layers_file, 'r') as fopen:
        database = load(fopen, Loader=Loader)
    database = process_db(database)
    return database

database = get_tc_database()
