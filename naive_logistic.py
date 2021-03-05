import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import *
from utils import *
import random


def get_feature_dict(feature_path, id_path):
    features = pd.read_csv(feature_path, header=None).values.tolist()
    print('feature dimension:', len(features[0]))
    id_list = pd.read_csv(id_path, header=None).values.tolist()
    id_list = [item[0] for item in id_list]
    feature_dict = {k:v for k,v in zip(id_list, features)}
    return feature_dict

def load_feature(human_feature_path, human_id_path, virus_feature_path, virus_id_path):
    if human_id_path == '' or str(human_id_path) == 'None':
        human_features = pd.read_csv(human_feature_path, header=None).values.tolist()
        human_feature_dict = {i:item for i, item in enumerate(human_features)}
    else:
        human_feature_dict = get_feature_dict(human_feature_path, human_id_path)

    if virus_id_path == '' or str(virus_id_path) == 'None':
        virus_features = pd.read_csv(virus_feature_path, header=None).values.tolist()
        virus_feature_dict = {i:item for i, item in enumerate(virus_features)}
    else:
        virus_feature_dict = get_feature_dict(virus_feature_path, virus_id_path)

    return human_feature_dict, virus_feature_dict

def get_data(pair_list, human_feature_dict, virus_feature_dict):
    data = list()
    for pair in pair_list:
        cur_data = list()
        if pair[1] not in human_feature_dict:
            print(pair[0])
            continue
        cur_data.extend(human_feature_dict[pair[1]])
        cur_data.extend(virus_feature_dict[pair[0]])
        data.append(cur_data)
    return data

def get_data2(pair_list, human_feature_dict, virus_feature_dict):
    data = list()
    for pair in pair_list:

        if pair[0] not in human_feature_dict:
            print(pair[0])
            continue
        cur_data = [item1 * item2 for item1, item2 in zip(human_feature_dict[pair[0]],virus_feature_dict[pair[1]])]
        data.append(cur_data)
    return data

def load_data(pos_int_path, neg_int_path, human_feature_dict, virus_feature_dict):
    pos_int = pd.read_csv(pos_int_path).values.tolist()
    neg_int = pd.read_csv(neg_int_path).values.tolist()
    lbl_list = [1.0] * len(pos_int) + [0.0] * len(neg_int)
    pos_data = get_data(pos_int, human_feature_dict, virus_feature_dict)
    neg_data = get_data(neg_int, human_feature_dict, virus_feature_dict)
    train_data = pos_data + neg_data
    train_data = np.array(train_data)
    return train_data, lbl_list



def main():
    # pos_int_path, id_path, atlas_path, save_path, remove_1freq
    parser = argparse.ArgumentParser(description='Linear regression model for virus-host PPI')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--data_dir', default='dataset/Denovo_slim/', help='dataset directory')
    parser.add_argument('--pos_train_path', default='pos_train.csv', help='pos train path')
    parser.add_argument('--neg_train_path', default='neg_train.csv', help='neg train path')
    parser.add_argument('--pos_test_path', default='pos_test.csv', help='test path')
    parser.add_argument('--neg_test_path', default='neg_test.csv', help='test label path')

    parser.add_argument('--human_feature_path', default='hfeatures.csv', help='human feature path')
    parser.add_argument('--human_id_path', default=None, help='human id path')
    parser.add_argument('--virus_feature_path', default='virus_seq_1900emb.csv', help='virus feature path')
    parser.add_argument('--virus_id_path', default=None, help='virus id path')

    parser.add_argument('--save_path', default='logistic_emb.csv', help='path to the node feature')
    args = parser.parse_args()

    args.data_dir = standardize_dir(args.data_dir)
    args.pos_train_path = args.data_dir + args.pos_train_path
    args.neg_train_path = args.data_dir + args.neg_train_path
    args.pos_test_path = args.data_dir + args.pos_test_path
    args.neg_test_path = args.data_dir + args.neg_test_path

    args.human_feature_path = args.data_dir + args.human_feature_path
    args.human_id_path = args.data_dir + args.human_id_path if args.human_id_path != '' and str(args.human_id_path) != 'None' else args.human_id_path
    args.virus_feature_path = args.data_dir + args.virus_feature_path
    args.virus_id_path = args.data_dir + args.virus_id_path if args.virus_id_path != '' and str(args.virus_id_path) != 'None' else args.virus_id_path

    args.save_path = args.data_dir + args.save_path

    human_feature_dict, virus_feature_dict = load_feature(args.human_feature_path, args.human_id_path, args.virus_feature_path, args.virus_id_path)
    all_scores = list()

    for i in range(10):
        print('Loading train data')
        train_data, train_lbl = load_data(args.pos_train_path, args.neg_train_path, human_feature_dict, virus_feature_dict)
        print('finish loading train data', args.pos_train_path)
        print('Loading test data')
        test_data, test_lbl = load_data(args.pos_test_path, args.neg_test_path, human_feature_dict, virus_feature_dict)
        save_path = args.save_path.replace('.csv', '_' + str(i) + '.csv')
        print('finish loading test data')

        print('Train pairs: ', len(train_data), 'Test pairs: ', len(test_data))

        clf = LogisticRegression(max_iter=10000)
        
        print('Start training ...')
        clf.fit(train_data, train_lbl)
        print('Finish training ...')
        preds = clf.predict_proba(test_data)[:,1]
        
        #print(preds[:2])
        print('Finish testing ...')

        auc_score, aupr_score,sn, sp, acc, topk = get_score(test_lbl, preds)
        all_scores.append([auc_score, aupr_score,sn, sp, acc, topk])

        print('AUC: ', round(auc_score, 4))
        print('AP: ', round(aupr_score, 4))

        join_vals = [[lbl, pred] for lbl, pred in zip(test_lbl, preds)]
        join_df = pd.DataFrame(np.array(join_vals), columns=['target', 'pred'])
        if args.data_dir.find('H1N1') > 0:
            name = 'h1n1_'
        elif args.data_dir.find('Ebola') > 0:
            name = 'ebola_'
        elif args.data_dir.find('Barman') > 0:
            name = 'barman_'
        elif args.data_dir.find('Denovo') > 0:
            name = 'denovo_slim_'
        else:
            name = 'bacteria_'
        join_df.to_csv('log/' + name + str(i) + '_naive.scores', index=False)
    arr = np.array(all_scores)
    avg = np.mean(arr, axis=0).tolist()
    std = np.std(arr, axis=0).tolist()
    print('Average: auc_score, aupr_score,sn, sp, acc, topk:')
    print(avg)
    print('Stadard deviation: auc_score, aupr_score,sn, sp, acc, topk:')
    print(std)



if __name__ == "__main__":
    main()

