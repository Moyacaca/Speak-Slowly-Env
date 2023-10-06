# encoding=utf-8

import sys
sys.path.append('./')
from utils.TestDecoder import GreedyCTCDecoder, CustomLM
from steps.train_ctc import Config
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from model.cnn_rnn import *
from torchaudio.models.decoder import ctc_decoder
import os
import time
import torch
import yaml
import argparse
import torch.nn as nn
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')

def test():
    args = parser.parse_args()
    try:
        conf = yaml.safe_load(open(args.conf, 'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)

    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))


    model_path = os.path.join(
        opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pkl')
    #TODO
    #顯卡壞了 都先改成cpu，之後記得改回來
    use_cuda = opts.use_gpu
    # device = torch.device('cuda') if use_cuda else torch.device('cpu')
    device = torch.device('cpu')
    # package = torch.load(model_path)
    package = torch.load(model_path,map_location=torch.device('cpu'))

    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']
    mel = opts.mel

    beam_width = opts.beam_width
    lm_alpha = opts.lm_alpha
    decoder_type = opts.decode_type
    vocab_file = opts.vocab_file

    vocab = Vocab(vocab_file)
    test_dataset = SpeechDataset(
        vocab, opts.test_scp_path, opts.test_trans_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=1,shuffle=False, 
                                   num_workers=opts.num_workers, pin_memory=False)

    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn,
                      cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()

    decoder = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    decoder_beam = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0,
                              space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha)
    
    #Test
    # decoder_greedy = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    # decoder_greedy_pytorch = GreedyCTCDecoder(list(vocab.index2word.values()))
    # token = list(vocab.index2word.values())
    # token[1] = '<unk>'
    # beam_search_decoder = ctc_decoder(
    #     lexicon=None,
    #     tokens=token,
    #     lm=opts.lm_path,
    #     nbest=3,
    #     beam_size=10,
    #     blank_token='blank',
    #     sil_token='sil',
    #     unk_word='<unk>',)
    #Test

    w1 = open(opts.decode_dir+"/ref", 'w+')  
    w2 = open(opts.decode_dir+"/hyp_greedy", 'w+')
    w3 = open(opts.decode_dir+"/hyp_beam",'w+')

    start = time.time()
    with torch.no_grad():
        for data in test_loader:
            inputs, input_sizes, trans, trans_sizes, utt_list = data
            inputs = inputs.to(device)
            trans = trans.to(device)
            #rnput_sizes = input_sizes.to(device)
            #target = target.to(device)
            #target_sizes = target_sizes.to(device)

            probs = model(inputs, trans)
            max_length, batch_size, _ = probs.size()  # probs每次的長度會不一樣，根據input長度而定
            # input_sizes = (input_sizes * max_length).long()
            input_sizes = torch.full(
                size=(batch_size,), fill_value=max_length, dtype=torch.long)
            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
            decoded_beam = decoder_beam.decode(probs, input_sizes.numpy().tolist())
            #test
            # input_sizes = torch.full(size=(probs.size(0),), fill_value=max_length, dtype=torch.long)
            # decoder_greedy_result = decoder_greedy.decode(probs, input_sizes.numpy().tolist())
            # probs = probs.transpose(0, 1)
            # results = beam_search_decoder(probs)
            
            # # decoder_greedy_pytorch_result = []
            # results_beam_search_decoder = []
            # for i in range(len(probs)):
            #     tmp = [vocab.index2word[num] for num in list(results[i][0].tokens.numpy())]
            #     results_beam_search_decoder.append(' '.join(tmp))
            #     decoder_greedy_pytorch_result.append(decoder_greedy_pytorch(probs[i]))
            #test

            trans, trans_sizes = trans.numpy(), trans_sizes.numpy()
            trans_words = []
            for i in range(len(trans)):
                trans_word = [vocab.index2word[num]
                         for num in trans[i][:trans_sizes[i]]]
                trans_words.append(' '.join(trans_word))
            assert len(trans_words) == len(decoded)
            # compute with out sil
            decoded_nosil = []
            trans_nosil = []
            decoded_beam_nosil = []
            for i in range(len(trans_words)):
                ref = trans_words[i].split(" ")
                hyp = decoded[i].strip().split(" ")
                hyp_beam = decoded_beam[i].strip().split(" ")
                ref_precess = [i for i in ref if(i != "sil")]
                hyp_precess = [i for i in hyp if(i != "sil")]
                hyp_precess_beam = [i for i in hyp_beam if(i != "sil")]
                trans_nosil.append(' '.join(ref_precess))
                decoded_nosil.append(' '.join(hyp_precess))
                decoded_beam_nosil.append(' '.join(hyp_precess_beam))
            for x in range(len(trans)):
                w1.write(utt_list[x] + " " + trans_nosil[x] + "\n")
                w2.write(utt_list[x] + " " + decoded_nosil[x] + "\n")
                w3.write(utt_list[x] + " " + decoded_beam_nosil[x] + "\n")


    print("total_phoneme:", decoder.num_word)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." %
          (len(test_dataset), time_used))
    w1.close()
    w2.close()
    w3.close()


if __name__ == "__main__":
    test()
