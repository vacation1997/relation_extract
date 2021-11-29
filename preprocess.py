import json
import numpy as np


def get_data(file_, data, num=None, save_path=None, save_flag=1):
    '''
    将百度2019数据集保存成模型所需的格式,如下例：
        {"text": "1《红色》就是钟汉良歌曲中较少类型的抒情慢歌", "spo_list": [{"subject": "红色", "predicate": "歌手", "object": "钟汉良"}]}
    '''
    with open(file_, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if num == None:
            num = len(lines) - 1
        idxs = [it for it in range(len(lines))]
        np.random.shuffle(idxs)
        for index in idxs:
            line = json.loads(lines[index])
            spo_list = []
            for spo in line['spo_list']:
                spo_list.append(
                    [spo['subject'], spo['predicate'], spo['object']])
            if len(data) <= num - 1:
                data.append({"text": line['text'], "spo_list": spo_list})
                continue
            if save_flag:
                save_json(data, save_path)


def save_json(data, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f)


def save_predicate():
    '''
    从百度2019数据集中将49种关系保存成id2word与word2id的形式
    '''
    id2pred = {}
    pred2id = {}
    with open('./corpus/all_50_schemas', 'r', encoding='utf8') as f:
        lines = f.readlines()
        flag = 0
        for index, line in enumerate(lines):
            value = json.loads(line)['predicate']
            if value in pred2id:
                flag += 1
                continue
            id2pred[f"{index - flag}"] = value
            pred2id[value] = index - flag
    save_json([id2pred, pred2id], './corpus/all_type.json')


if __name__ == '__main__':
    save_predicate()

    train_data = []
    validation_data = []
    test_data = []
    get_data('./base_corpus/train_data.json', train_data,
             save_path='./corpus/train_data.json')
    get_data('./base_corpus/dev_data.json', validation_data, num=500,
             save_path='./corpus/valid_data.json')
    get_data('./base_corpus/dev_data.json', test_data, num=500,
             save_path='./corpus/test_data.json')
