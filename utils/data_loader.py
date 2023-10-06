#encoding=utf-8

import torch
import kaldiio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
from utils.tools import load_wave, F_Mel, make_context, skip_feat, spec_augment, data_enhance
import json
audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}

class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.word2index = {"blank": 0, "<unk>": 1}
        self.index2word = {0: "blank", 1: "<unk>"}
        self.word2count = {}
        self.n_words = 2
        self.read_lang()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def read_lang(self):
        print("Reading vocabulary from {}".format(self.vocab_file))
        with open(self.vocab_file, 'r') as rf:
            line = rf.readline()
            while line:
                line = line.strip().split(' ')
                if len(line) > 1:
                    sen = ' '.join(line[1:])
                else:
                    sen = line[0]
                self.add_sentence(sen)
                line = rf.readline()
        print("Vocabulary size is {}".format(self.n_words))


class SpeechDataset(Dataset):
    def __init__(self, vocab, scp_path, test_trans_path, opts):
        self.vocab = vocab
        self.scp_path = scp_path
        self.test_trans_path = test_trans_path
        self.left_ctx = opts.left_ctx
        self.right_ctx = opts.right_ctx
        self.n_skip_frame = opts.n_skip_frame
        self.n_downsample = opts.n_downsample
        self.feature_type = opts.feature_type
        self.process_feature_label()
    
    def process_feature_label(self):
        with open(self.test_trans_path) as f:
            transcript_dict = json.loads(f.read())
        self.item = []
        #read the ark path
        with open(self.scp_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, path = line.strip().split(' ')
                wordcard = '_'.join(utt.split("_")[1:])
                if(wordcard not in transcript_dict):
                    print("wordcard not in transcript_dict!!!")
                    sys.exit(1)
                tmp = transcript_dict[wordcard]
                transcript= [self.vocab.word2index[c] if c in self.vocab.word2index else self.vocab.word2index['<unk>'] for c in tmp.split()]
                self.item.append((path,transcript, utt))
                line = rf.readline()

       	# #read the label
        # label_dict = dict()
        # with open(self.lab_path, 'r') as rf:
        #     line = rf.readline()
        #     while line:
        #         utt, label = line.strip().split(' ', 1)
        #         label_dict[utt] = [self.vocab.word2index[c] if c in self.vocab.word2index else self.vocab.word2index['<unk>'] for c in label.split()]
        #         line = rf.readline() 
                
        #read the transcript
        # trans_dict = dict()
        # with open(self.trans_path, 'r') as rf:
        #     line = rf.readline()
        #     while line:
        #         utt, trans = line.strip().split(' ', 1)
        #         trans_dict[utt] = [self.vocab.word2index[c] if c in self.vocab.word2index else self.vocab.word2index['<unk>'] for c in trans.split()]
        #         line = rf.readline() 



    def __getitem__(self, idx):
        path, trans, utt = self.item[idx]
        feat = kaldiio.load_mat(path)

        # make_context為concat前後時間的特徵、skip_feat為對frame做取樣
        feat = skip_feat(make_context(feat, self.left_ctx,self.right_ctx), self.n_skip_frame)
        seq_len, dim = feat.shape #(跟時間有關的序列、dim為fbank加上後面兩個時間的fank再除2)
        if seq_len % self.n_downsample != 0:
            pad_len = self.n_downsample - seq_len % self.n_downsample
            feat = np.vstack([feat, np.zeros((pad_len, dim))])
        #feat:特徵(時間相關、一個frame(或更多)的特徵)  #trans:transcript_phn_text的phoneme序列 #utt:每個表單索引
        return (torch.from_numpy(feat), torch.LongTensor(trans), utt)

    def __len__(self):
        return len(self.item) 

def create_input(batch):
    inputs_max_length = max(x[0].size(0) for x in batch) #一個batch裡最長的時間序列
    feat_size = batch[0][0].size(1)  # feature的維度

    trans_max_length = max(x[1].size(0)
                           for x in batch)  # 一個batch裡最長的transcript_phn_text phoneme序列
    
    batch_size = len(batch)
    batch_data = torch.zeros(batch_size, inputs_max_length, feat_size) 
    batch_trans = torch.zeros(batch_size, trans_max_length)
    input_sizes = torch.zeros(batch_size)
    trans_sizes = torch.zeros(batch_size)
    utt_list = []

    for x in range(batch_size):
        feature, trans, utt = batch[x]
        feature_length = feature.size(0) #此feature的時間維度
        trans_length = trans.size(0)
        batch_data[x].narrow(0, 0, feature_length).copy_(feature)
        batch_trans[x].narrow(0, 0, trans_length).copy_(trans)
        input_sizes[x] = feature_length / inputs_max_length
        trans_sizes[x] = trans_length
        utt_list.append(utt)
    # batch_data: 整個batch的feature data,time序列為整個batch最長的time序列, input_sizes: 時間序列佔整個batch最長的時間序列比例
    # batch_trans: transcript_phn_text的phoneme序列(擴增成整個batch最長的phoneme序列) 
    # trans_sizes: transcript_phn_text的phoneme序列長度 utt_list: 存有整個batch utt的list
    return batch_data.float(), input_sizes.float(), batch_trans.long(), trans_sizes.long(),utt_list 
    
'''
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, 
                                                    collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
subclass of DataLoader and rewrite the collate_fn to form batch
'''

class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_input

if __name__ == '__main__':
    dev_dataset = SpeechDataset()
    dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=2, shuffle=True)
    
    import visdom
    viz = visdom.Visdom(env='fan')
    for i in range(1):
        show = dev_dataset[i][0].transpose(0, 1)
        text = dev_dataset[i][1]
        for num in range(len(text)):
            text[num] = dev_dataset.int2class[text[num]]
        text = ' '.join(text)
        opts = dict(title=text, xlabel='frame', ylabel='spectrum')
        viz.heatmap(show, opts = opts)
