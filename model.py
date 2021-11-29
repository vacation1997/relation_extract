import json
from keras import models
import numpy as np
from tqdm import tqdm
from keras_bert import load_trained_model_from_checkpoint, get_custom_objects


import tensorflow as tf
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


from utils import data_generator, OurTokenizer, read_token


import datetime


def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.expand_dims(K.arange(0, K.shape(seq)[0]), 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs)


def get_loss(mask, sbj_heads, sbj_tails, sbj_heads_out, sbj_tails_out, obj_heads, obj_tails, obj_heads_out, obj_tails_out):
    sbj_heads = K.expand_dims(sbj_heads, 2)
    sbj_tails = K.expand_dims(sbj_tails, 2)

    sbj_heads_loss = K.sum(K.binary_crossentropy(sbj_heads, sbj_heads_out) * mask) / K.sum(mask)
    sbj_tails_loss = K.sum(K.binary_crossentropy(sbj_tails, sbj_tails_out) * mask) / K.sum(mask)

    obj_heads_loss = K.sum(K.sum(K.binary_crossentropy(obj_heads, obj_heads_out), 2, keepdims=True) * mask) / K.sum(mask)
    obj_tails_loss = K.sum(K.sum(K.binary_crossentropy(obj_tails, obj_tails_out), 2, keepdims=True) * mask) / K.sum(mask)

    loss = (sbj_heads_loss + sbj_tails_loss) + (obj_heads_loss + obj_tails_loss)
    return loss


def create_model(config_path, checkpoint_path, num_classes, learning_rate=1e-5):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for layer in bert_model.layers:
        layer.trainable = True

    token_in, segment_in = Input(shape=(None,)), Input(shape=(None,))
    sbj_heads_in, sbj_tails_in = Input(shape=(None,)), Input(shape=(None,))
    sbj_head_in, sbj_tail_in = Input(shape=(1,)), Input(shape=(1,))
    obj_heads_in, obj_tails_in = Input(shape=(None, num_classes)), Input(shape=(None, num_classes))
    token, segment, sbj_heads, sbj_tails, sbj_head, sbj_tail, obj_heads, obj_tails = token_in, segment_in, sbj_heads_in, sbj_tails_in, sbj_head_in, sbj_tail_in, obj_heads_in, obj_tails_in

    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(token)

    bert_out = bert_model([token, segment])
    sbj_heads_out = Dense(1, activation='sigmoid')(bert_out)
    sbj_tails_out = Dense(1, activation='sigmoid')(bert_out)
    subject_model = Model([token_in, segment_in], [sbj_heads_out, sbj_tails_out])

    sbj_heads_info = Lambda(seq_gather)([bert_out, sbj_head])
    sbj_tails_info = Lambda(seq_gather)([bert_out, sbj_tail])
    sbj_ave = Average()([sbj_heads_info, sbj_tails_info])
    sbj_info = Add()([bert_out, sbj_ave])
    obj_heads_out = Dense(num_classes, activation='sigmoid')(sbj_info)
    obj_tails_out = Dense(num_classes, activation='sigmoid')(sbj_info)
    object_model = Model([token_in, segment_in, sbj_head, sbj_tail], [obj_heads_out, obj_tails_out])

    train_model = Model([token_in, segment_in, sbj_heads_in, sbj_tails_in, sbj_head_in, sbj_tail_in, obj_heads_in, obj_tails_in], [sbj_heads_out, sbj_tails_out, obj_heads_out, obj_tails_out])

    loss = get_loss(mask, sbj_heads, sbj_tails, sbj_heads_out, sbj_tails_out, obj_heads, obj_tails, obj_heads_out, obj_tails_out)

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate))
    train_model.summary()
    return subject_model, object_model, train_model


def extract_items(text, token, segment, subject_model, object_model, id2predicate):
    sbj_heads, sbj_tails = subject_model.predict([token, segment])
    sbj_heads, sbj_tails = np.where(sbj_heads[0] > 0.5)[0], np.where(sbj_tails[0] > 0.4)[0]
    sbj_list = []
    for start in sbj_heads:
        ends = sbj_tails[sbj_tails > start]
        if len(ends) > 0:
            end = ends[0]
            sbj = text[start-1:end]
            sbj_list.append((sbj, start, end))
    if sbj_list:
        all_list = []
        token = np.repeat(token, len(sbj_list), 0)
        segment = np.repeat(segment, len(sbj_list), 0)
        sbj_head, sbj_tail = np.array([x[1:] for x in sbj_list]).T.reshape((2, -1, 1))
        obj_heads, obj_tails = object_model.predict([token, segment, sbj_head, sbj_tail])
        for i, item in enumerate(sbj_list):
            obj_head_idxs, obj_tail_idxs = np.where(obj_heads[i] > 0.5), np.where(obj_tails[i] > 0.4)
            for token_idx_s, predicate_idx_s in zip(*obj_head_idxs):
                for token_idx_e, predicate_idx_e in zip(*obj_tail_idxs):
                    if token_idx_s <= token_idx_e and predicate_idx_s == predicate_idx_e:
                        obj = text[token_idx_s-1:token_idx_e]
                        predicate = id2predicate[predicate_idx_s]
                        all_list.append((item[0], predicate, obj))
                        break
        spo_set = set()
        for s, p, o in all_list:
            spo_set.add((s, p, o))
        return spo_set
    else:
        return set()


class Evaluate(Callback):
    def __init__(self, tokenizer, subject_model, object_model, id2predicate):
        self.F1 = []
        self.best = 0.
        self.tokenizer = tokenizer
        self.subject_model = subject_model
        self.object_model = object_model
        self.id2predicate = id2predicate

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(self.tokenizer, self.subject_model, self.object_model, self.id2predicate)
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            save_time = now.strftime('%Y-%m-%d-%H-%M')
            self.subject_model.save(f'./models/sbj_model_{save_time}.h5')
            self.object_model.save(f'./models/obj_model_{save_time}.h5')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
              (f1, precision, recall, self.best))


def evaluate(tokenizer, subject_model, object_model, id2predicate):
    """模型评估"""
    orders = ['subject', 'predicate', 'object']
    A, B, C = 1e-10, 1e-10, 1e-10
    F = open('./models/dev_pred.json', 'w', encoding='utf-8')

    for item in tqdm(valid_data, desc='evaluate:'):
        token, segment = tokenizer.encode(first=item['text'])
        token, segment = np.array([token]), np.array([segment])
        R = extract_items(item['text'], token, segment, subject_model, object_model, id2predicate)
        T = set([tuple(spo) for spo in item['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        s = json.dumps({
            'text': item['text'],
            'spo_list': [
                dict(zip(orders, spo)) for spo in T
            ],
            'spo_list_pred': [
                dict(zip(orders, spo)) for spo in R
            ],
            'new': [
                dict(zip(orders, spo)) for spo in R - T
            ],
            'lack': [
                dict(zip(orders, spo)) for spo in T - R
            ]
        }, ensure_ascii=False, indent=4)
        F.write(s + '\n')
    F.close()
    return 2 * A / (B + C), A / B, A / C


def predict_(test_data, tokenizer, subject_model, object_model, id2predicate):
    """
    输出测试结果
    """
    orders = ['subject', 'predicate', 'object']
    F = open('./models/test_pred.json', 'w', encoding='utf-8')
    for item in tqdm(iter(test_data)):
        token, segment = tokenizer.encode(first=item['text'])
        token, segment = np.array([token]), np.array([segment])
        R = extract_items(item['text'], token, segment, subject_model, object_model, id2predicate)
        s = json.dumps({
            'text': item['text'],
            'spo_list': [
                dict(zip(orders, spo + ('', ''))) for spo in R
            ]
        }, ensure_ascii=False)
        F.write(s + '\n')
    F.close()


if __name__ == '__main__':
    now = datetime.datetime.now()

    config_path = './bert/bert_config.json'
    checkpoint_path = './bert/bert_model.ckpt'
    dict_path = './bert/vocab.txt'

    train_file = './corpus/train_data.json'
    valid_file = './corpus/valid_data.json'
    test_file = './corpus/test_data.json'
    schema_file = './corpus/all_type.json'

    train_data = json.load(open(train_file), strict=False)
    valid_data = json.load(open(valid_file), strict=False)
    test_data = json.load(open(test_file), strict=False)
    id2predicate, predicate2id = json.load(open(schema_file), strict=False)
    id2predicate = {int(i): j for i, j in id2predicate.items()}
    num_classes = len(id2predicate)

    token_dict = read_token(dict_path)
    tokenizer = OurTokenizer(token_dict)

    subject_model, object_model, train_model = create_model(config_path, checkpoint_path, num_classes)
    # from keras.utils import plot_model
    # plot_model(subject_model, to_file="sbj_model.png", show_shapes=True)
    # plot_model(object_model, to_file="obj_model.png", show_shapes=True)
    # plot_model(train_model, to_file="train_model.png", show_shapes=True)
    # subject_model, object_model = models.load_model('./models/2021-11-24-06-51_sbj.h5', custom_objects=get_custom_objects()), models.load_model('./models/2021-11-24-06-51_obj.h5', custom_objects=get_custom_objects())

    generator = data_generator(train_data, tokenizer, predicate2id, batch_size=4)
    evaluator = Evaluate(tokenizer, subject_model, object_model, id2predicate)

    train_model.fit(generator.__iter__(), steps_per_epoch=len(generator), epochs=10, callbacks=[evaluator])

    # predict_(test_data, tokenizer, subject_model, object_model, id2predicate)
