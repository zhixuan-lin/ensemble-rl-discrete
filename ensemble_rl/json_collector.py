# coding=utf-8
# Copyright 2022 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collector class for saving iteration statistics to a pickle file."""

import collections
import functools
import os.path as osp
import json
from typing import Sequence

from dopamine.metrics import collector
from dopamine.metrics import statistics_instance
import tensorflow as tf
from collections import OrderedDict


class JSONCollector(collector.Collector):
    """Collector class for reporting statistics to the console."""
    def __init__(self, base_dir: str):
        if base_dir is None:
            raise ValueError(
                'Must specify a base directory for PickleCollector.')
        super().__init__(base_dir)
        self._statistics = collections.defaultdict(dict)

    def get_name(self) -> str:
        return 'json'

    def write(
            self, statistics: Sequence[statistics_instance.StatisticsInstance]
    ) -> None:
        # This Collector is trying to write metrics as close as possible to what
        # is currently written by the Dopamine Logger, so as to be as compatible
        # with user's plotting setups.
        for s in statistics:
            if not self.check_type(s.type):
                continue
            self._statistics[s.step][s.name] = s.value

    def get_ordered_data(self):
        stats = []
        for iteration in sorted(self._statistics):
            iter_stat = OrderedDict([
                ('iteration', iteration),
                *list(self._statistics[iteration].items())
            ])
            stats.append(iter_stat)
        return stats

    def flush(self):
        stats = self.get_ordered_data()
        json_file = osp.join(self._base_dir, 'stats.json')
        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def dump(self):
        return self.get_ordered_data()

    def load(self, data):
        for iter_stat in data:
            iteration = iter_stat.pop('iteration')
            self._statistics[iteration] = iter_stat
