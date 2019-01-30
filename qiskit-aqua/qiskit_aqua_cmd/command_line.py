# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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
# =============================================================================

import argparse
import json
import logging
import tkinter as tk

_ROOT = None


def main():
    global _ROOT
    _ROOT = tk.Tk()
    _ROOT.withdraw()
    _ROOT.update_idletasks()
    _ROOT.after(0, main_algorithm)
    _ROOT.mainloop()


def main_algorithm():
    try:
        from qiskit.aqua._logging import get_logging_level, build_logging_config, set_logging_config
        from qiskit_aqua_cmd import Preferences
        from qiskit.aqua import run_algorithm
        from qiskit.aqua.utils import convert_json_to_dict
        parser = argparse.ArgumentParser(description='Qiskit Aqua Command Line Tool')
        parser.add_argument('input',
                            metavar='input',
                            help='Algorithm JSON input file')
        parser.add_argument('-jo',
                            metavar='output',
                            help='Algorithm JSON output file name',
                            required=False)

        args = parser.parse_args()

        # update logging setting with latest external packages
        preferences = Preferences()
        logging_level = logging.INFO
        if preferences.get_logging_config() is not None:
            set_logging_config(preferences.get_logging_config())
            logging_level = get_logging_level()

        preferences.set_logging_config(build_logging_config(logging_level))
        preferences.save()

        set_logging_config(preferences.get_logging_config())

        params = None
        with open(args.input) as json_file:
            params = json.load(json_file)

        ret = run_algorithm(params, None, True)

        if args.jo is not None:
            with open(args.jo, 'w') as f:
                print('{}'.format(ret), file=f)
        else:
            convert_json_to_dict(ret)
            print('\n\n--------------------------------- R E S U L T ------------------------------------\n')
            if isinstance(ret, dict):
                for k, v in ret.items():
                    print("'{}': {}".format(k, v))
            else:
                print(ret)
    finally:
        global _ROOT
        if _ROOT is not None:
            _ROOT.destroy()
            _ROOT = None
