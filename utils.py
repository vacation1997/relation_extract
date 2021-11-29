import codecs
from keras_bert import Tokenizer
import math
import numpy as np
from random import choice


def read_token(bert_dict_path):
    """
    读取bert的token字典
    """
    token_dict = {}

    with codecs.open(bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


class OurTokenizer(Tokenizer):
    '''
    自定义的Tokenizer类，把空格作为[unused1]类
    '''

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


class data_generator:
    def __init__(self, data, tokenizer, predicate2id, batch_size=16, maxlen=160):
        self.data = data
        self.batch_size = batch_size
        self.steps = math.ceil(len(self.data) / self.batch_size)
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.num_classes = len(predicate2id)
        self.maxlen = maxlen

    def __len__(self):
        return self.steps

    def __iter__(self):
        '''
        理由传入的数据(data)按对应的需求生成批次数据并返回
        '''
        def seq_padding(X, padding=0):
            '''
            X:一个批次不同长度使用id表示的句子
            返回填充为本批次最大句子长度的矩阵，形状如[batch_size,seqence_length]
            '''
            Len = [len(x) for x in X]
            MaxLen = max(Len)
            return np.array([np.concatenate([x, [padding] * (MaxLen - len(x))]) if len(x) < MaxLen else x for x in X])

        def list_find(list1, list2):
            '''
            传入分割好的list，每个元素为一个字符
            如果list2中含有list1，返回起始位置，否则返回值-1
            '''
            n_list2 = len(list2)
            for i in range(len(list1)):
                if list1[i:i + n_list2] == list2:
                    return i
            return -1

        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            token_batch, segment_batch, sbj_heads_batch, sbj_tails_batch, sbj_head_batch, sbj_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []
            for idx in idxs:
                line = self.data[idx]
                text = line['text'][:self.maxlen]
                token_list = self.tokenizer.tokenize(text)
                items = {}
                for spo in line['spo_list']:
                    spo_list = (self.tokenizer.tokenize(spo[0])[
                                1:-1], spo[1], self.tokenizer.tokenize(spo[2])[1:-1])
                    sbj_head_temp = list_find(token_list, spo_list[0])
                    obj_head_temp = list_find(token_list, spo_list[2])
                    if sbj_head_temp != -1 and obj_head_temp != -1:
                        key = (sbj_head_temp, sbj_head_temp +
                               len(spo_list[0]) - 1)
                        if key not in items:
                            items[key] = []
                        items[key].append(
                            (obj_head_temp, obj_head_temp + len(spo_list[2]) - 1, self.predicate2id[spo_list[1]]))
                if items:
                    token, segment = self.tokenizer.encode(first=text)
                    sbj_heads, sbj_tails = np.zeros(
                        len(token_list)), np.zeros(len(token_list))
                    for key in items.keys():
                        sbj_heads[key[0]] = 1
                        sbj_tails[key[1]] = 1
                    sbj_head, sbj_tail = choice(list(items.keys()))
                    obj_heads, obj_tails = np.zeros((len(token_list), self.num_classes)), np.zeros(
                        (len(token_list), self.num_classes))
                    for value in items.get((sbj_head, sbj_tail), []):
                        obj_heads[value[0]][value[2]] = 1
                        obj_tails[value[1]][value[2]] = 1
                    token_batch.append(token)
                    segment_batch.append(segment)
                    sbj_heads_batch.append(sbj_heads)
                    sbj_tails_batch.append(sbj_tails)
                    sbj_head_batch.append(sbj_head)
                    sbj_tail_batch.append(sbj_tail)
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)

                    if len(token_batch) == self.batch_size or idx == idxs[-1]:
                        token_batch = seq_padding(token_batch)
                        segment_batch = seq_padding(segment_batch)
                        sbj_heads_batch = seq_padding(sbj_heads_batch)
                        sbj_tails_batch = seq_padding(sbj_tails_batch)
                        sbj_head_batch, sbj_tail_batch = np.array(
                            sbj_head_batch), np.array(sbj_tail_batch)
                        obj_heads_batch = seq_padding(
                            obj_heads_batch, np.zeros(self.num_classes))
                        obj_tails_batch = seq_padding(
                            obj_tails_batch, np.zeros(self.num_classes))
                        yield [token_batch, segment_batch, sbj_heads_batch, sbj_tails_batch, sbj_head_batch, sbj_tail_batch, obj_heads_batch, obj_tails_batch]
                        token_batch, segment_batch, sbj_heads_batch, sbj_tails_batch, sbj_head_batch, sbj_tail_batch, obj_heads_batch, obj_tails_batch, = [], [], [], [], [], [], [], []
