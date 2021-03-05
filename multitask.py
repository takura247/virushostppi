from model import *
import argparse
import pandas as pd
import numpy as np
from utils import *


def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return pos_df, t.LongTensor(int_edges), t.FloatTensor(int_lbl)


def main():

    parser = argparse.ArgumentParser(description='Multitask transfer model for novel virus-human PPI')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n_runs', type=int, default=10, metavar='N',
                        help='number of experiment runs')
    parser.add_argument('--ppi_weight', type=float, default=0.001, metavar='N',
                        help='weight of PPI prediction')
    parser.add_argument('--data_dir', default='dataset/human-H1N1/', help='dataset directory')
    parser.add_argument('--virus_feature_path', default='virus_seq_1900emb.csv', help='virus_feature_path')
    parser.add_argument('--human_feature_path', default='hfeatures.csv', help = 'human_feature path')
    parser.add_argument('--hppi_edge_list', default='hppi_edge_list.csv', help='human ppi edge list path')
    parser.add_argument('--hppi_edge_weight', default='hppi_edge_weight.csv', help='human ppi edge weight path')
    parser.add_argument('--pos_train_path', default='pos_train.csv', help='pos train path')
    parser.add_argument('--pos_test_path', default='pos_test.csv', help='pos test path')
    parser.add_argument('--neg_train_path', default='neg_train.csv', help='neg train path')
    parser.add_argument('--neg_test_path', default='neg_test.csv', help='neg test path')


    args = parser.parse_args()

    args.data_dir = standardize_dir(args.data_dir)
    args.virus_feature_path = args.data_dir + args.virus_feature_path
    args.human_feature_path = args.data_dir + args.human_feature_path
    args.hppi_edge_list = args.data_dir + args.hppi_edge_list
    args.hppi_edge_weight = args.data_dir + args.hppi_edge_weight
    args.pos_train_path = args.data_dir + args.pos_train_path
    args.pos_test_path = args.data_dir + args.pos_test_path
    args.neg_train_path = args.data_dir + args.neg_train_path
    args.neg_test_path = args.data_dir + args.neg_test_path
    args.ppi_weight = float(args.ppi_weight)
    args.n_runs = int(args.n_runs)

    human_features = pd.read_csv(args.human_feature_path, header=None).values
    human_features = t.FloatTensor(human_features)

    virus_features = t.FloatTensor(pd.read_csv(args.virus_feature_path, header=None).values)
    n_virus = virus_features.size(0)
    n_human = human_features.size(0)
    print('Finish loading features')
    hppi_edgeweight = pd.read_csv(args.hppi_edge_weight).values
    hppi_edgelist = pd.read_csv(args.hppi_edge_list).values


    hppi_edgeweight = t.FloatTensor(hppi_edgeweight)
    hppi_edgelist = t.LongTensor(hppi_edgelist)
    print('Finish loading human PPI')

    vindex_tensor = t.LongTensor(list(range(n_virus)))
    hindex_tensor = t.LongTensor(list(range(n_human)))

    pos_train_pairs, train_tensor, train_lbl_tensor = read_int_data(args.pos_train_path, args.neg_train_path)
    _, test_tensor, test_lbl_tensor = read_int_data(args.pos_test_path, args.neg_test_path)
    test_lbl = test_lbl_tensor.detach().numpy()

    print('Finish loading int pairs')

    hppi_edgeweight = hppi_edgeweight.view(-1)


    criterion = t.nn.BCELoss()

    if t.cuda.is_available():
        hppi_edgelist = hppi_edgelist.cuda()
        hppi_edgeweight = hppi_edgeweight.cuda()

        vindex_tensor = vindex_tensor.cuda()
        hindex_tensor = hindex_tensor.cuda()
        virus_features = virus_features.cuda()
        human_features = human_features.cuda()
        train_tensor = train_tensor.cuda()
        test_tensor = test_tensor.cuda()
        train_lbl_tensor = train_lbl_tensor.cuda()
        criterion = criterion.cuda()


    auc_scores = list()
    epoch_rec = list()

    args.epochs = int(args.epochs)
    max_auc = [0,0]
    lrs = [0.001]
    for lr in lrs:
        all_scores = list()
        for i in range(args.n_runs):
            model = Model(n_virus, n_human, 32)
            # Initialize embedding layers with pre-trained sequence embedding
            model.vemb.weight.data = virus_features
            model.hemb.weight.data = human_features
            optimizer = t.optim.Adam(model.parameters(), lr=lr)
            if t.cuda.is_available():
                model = model.cuda()

            for epoch in range(0, args.epochs):
                model.train()
                optimizer.zero_grad()
                score, hppi_out = model(vindex_tensor, hindex_tensor,
                                        train_tensor, hppi_edgelist)
                loss = criterion(score, train_lbl_tensor) + args.ppi_weight * criterion(hppi_out, hppi_edgeweight)
                loss.backward()
                optimizer.step()
                loss_val = loss.item() if not t.cuda.is_available() else loss.cpu().item()
                print('Epoch: ', epoch, ' loss: ', loss_val / train_lbl_tensor.size(0))

                if epoch % 2 == 0:
                    
                    model.eval()
                    pred_score, _ = model(vindex_tensor, hindex_tensor,
                                          test_tensor, hppi_edgelist)

                    pred_score = pred_score.detach().numpy() if not t.cuda.is_available() else pred_score.cpu().detach().numpy()

                    test_pred_lbl = pred_score.tolist()
                    test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
                    auc_score, aupr_score, sn, sp, acc, topk = get_score(
                        test_lbl, test_pred_lbl)
                    print('lr:%.4f, auc:%.4f, aupr:%.4f' %(lr, auc_score, aupr_score))
            
            model.eval()
            pred_score, _= model(vindex_tensor, hindex_tensor,
                                 test_tensor, hppi_edgelist)

            pred_score = pred_score.detach().numpy() if not t.cuda.is_available() else pred_score.cpu().detach().numpy()

            test_pred_lbl = pred_score.tolist()
            test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
            auc_score, aupr_score, sn, sp, acc, topk = get_score(test_lbl, test_pred_lbl)
            join_vals = [[lbl, pred] for lbl, pred in zip(test_lbl, test_pred_lbl)]
            join_df = pd.DataFrame(np.array(join_vals), columns=['target', 'pred'])
            join_df.to_csv(args.data_dir + 'h1n1_seqemb_' + str(i) +'.scores', index=False)
            print('lr:%.4f, auc:%.4f, aupr:%.4f' %(lr, auc_score, aupr_score))

            all_scores.append([auc_score, aupr_score, sn, sp, acc, topk])
            if max_auc[0] < auc_score:
                max_auc = [auc_score, aupr_score]
        arr = np.array(all_scores)
        print('all_scores: ', all_scores)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        print('Mean auc_score, aupr_score, sn, sp, acc, topk:')
        print(mean)
        print('Std auc_score, aupr_score, sn, sp, acc, topk:')
        print(std)
        print('max auc, aupr:', max_auc)

if __name__ == "__main__":
    main()






