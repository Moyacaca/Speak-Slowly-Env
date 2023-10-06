#!/bin/bash

# need  1. 跟讀文本音素序列 ref
#       2. CTC識別序列 hyp


. ./path.sh

data_dir=$1
# note : sequence only have xx phoneme, no sil

align-text ark:$data_dir/ref  ark:$data_dir/hyp_beam ark,t:- | utils/scoring/wer_per_utt_details.pl > $data_dir/ref_our_detail
# python utils/scoring/ins_del_sub_cor_analysis.py
#rm -rf ref_human_detail human_our_detail ref_our_detail



