#!/bin/sh
config_file='conf/ctc_config.yaml'

echo "Decoding..."
python steps/test_ctc_nosil.py --conf $config_file || exit 1;