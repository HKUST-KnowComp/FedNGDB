import argparse
from gqe import GQE
from q2b import Q2B
from q2p import Q2P
from tree_lstm import TreeLSTM
from fedCQA import Client, Server, log_aggregation

import torch
from dataloader import TrainDataset, ValidDataset, TestDataset, SingledirectionalOneShotIterator
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import gc
import pickle
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import json



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The training and evaluation script for the models')

    parser.add_argument("--query_dir", required=True)
    parser.add_argument('--kg_data_dir', default="KG_data_fed/", help="The path the original kg data")

    parser.add_argument('--num_layers', default=3, type=int, help="num of layers for sequential models")

    parser.add_argument('--log_steps', default=50000, type=int, help='train log every xx steps')
    parser.add_argument('-dn', '--data_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', default=64, type=int)

    parser.add_argument('-d', '--entity_space_dim', default=400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.002, type=float)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)

    parser.add_argument("--warm_up_steps", default=1000, type=int)

    parser.add_argument("-m", "--model", required=True)

    parser.add_argument("--checkpoint_path", type=str, default="../logs_fed")
    parser.add_argument("-old", "--old_loss_fnt", action="store_true")
    # parser.add_argument("-fol", "--use_full_fol", action="store_true")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--n_clients", type=int, default=3, choices=[3, 5, 10])
    parser.add_argument("--global_steps", type=int, default=300000)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--evaluate_cross", action="store_true")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--server_model", type=str, choices=["NA", "fedR", "fedE", "fedC"])

    # debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--partial_update", action="store_true")
    parser.add_argument("--optimizer_update", action="store_true")

    # parser.add_argument("--few_shot", type=int, default=32)

    args = parser.parse_args()

    KG_data_path = "../" + args.kg_data_dir
    data_name = args.data_name
    args.checkpoint_path = args.checkpoint_path + "/fed-" + str(args.n_clients)
    args.query_dir = args.query_dir + "/fed-" + str(args.n_clients)

    train_query_file_names = []
    valid_query_file_names = []
    test_query_file_names = []
    # The train, valid and test files in the clients.

    data_path = KG_data_path + args.data_name

    loss = "old-loss" if args.old_loss_fnt else "new-loss"
    # new_loss: label smoothing for iterative models, default for sequential models
    # old_loss: negative sampling for iterative models
    info = loss + "_" + str(args.n_clients) + "_" + args.server_model

    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])



    batch_size = args.batch_size

    # create model


    clients = []
    models_clients = []
    train_iterators_clients = []
    valid_iterators_clients = []
    test_iterators_clients = []

    entity_mask = []
    relation_mask = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # client data preparation and client server generation
    train_stats = {}
    valid_stats = {}
    test_stats = {}

    for i in range(args.n_clients):

        train_query_file_names.append(data_name + "_train_queries_" + str(i) + ".pkl")
        valid_query_file_names.append(data_name + "_valid_queries_" + str(i) + ".pkl")
        test_query_file_names.append(data_name + "_test_queries_" + str(i) + ".pkl")

        with open(args.query_dir + "/" + train_query_file_names[i], "rb") as fin:
            train_data_dict = pickle.load(fin)

        for query_type, query_answer_dict in train_data_dict.items():
            if query_type not in train_stats:
                train_stats[query_type] = len(query_answer_dict)
            else:
                train_stats[query_type] += len(query_answer_dict)


        with open(args.query_dir + "/" + valid_query_file_names[i], "rb") as fin:
            valid_data_dict = pickle.load(fin)

        for query_type, query_answer_dict in valid_data_dict.items():
            if query_type not in valid_stats:
                valid_stats[query_type] = len(query_answer_dict)
            else:
                valid_stats[query_type] += len(query_answer_dict)


        with open(args.query_dir + "/" + test_query_file_names[i], "rb") as fin:
            test_data_dict = pickle.load(fin)

        for query_type, query_answer_dict in test_data_dict.items():
            if query_type not in test_stats:
                test_stats[query_type] = len(query_answer_dict)
            else:
                test_stats[query_type] += len(query_answer_dict)
    length = 0
    for key in train_stats.keys():
        length += train_stats[key]
    print("train stats: ", length/args.n_clients)

    length = 0
    for key in valid_stats.keys():
        length += valid_stats[key]
    print("valid stats: ", length/args.n_clients)

    length = 0
    for key in test_stats.keys():
        length += test_stats[key]
    print("test stats: ", length/args.n_clients)

    print()
    #
    #     test_iterators = {}
    #     for query_type, query_answer_dict in test_data_dict.items():
    #         new_iterator = DataLoader(
    #             TestDataset(nentity, nrelation, query_answer_dict),
    #             batch_size=args.batch_size,
    #             shuffle=True,
    #             collate_fn=TestDataset.collate_fn
    #         )
    #         test_iterators[query_type] = new_iterator
    #
    #     train_iterators_clients.append(train_iterators)
    #     valid_iterators_clients.append(valid_iterators)
    #     test_iterators_clients.append(test_iterators)
    #
    # cross_train_iterators = {}
    # cross_valid_iterators = {}
    # cross_test_iterators = {}
    #
    # # cross data preparation
    # if args.evaluate_cross:
    #     with open(args.query_dir + "/" + data_name + "_cross_clients_valid_queries.pkl", "rb") as fin:
    #         valid_data_dict = pickle.load(fin)
    #     for query_type, query_answer_dict in valid_data_dict.items():
    #         new_iterator = DataLoader(
    #             ValidDataset(nentity, nrelation, query_answer_dict),
    #             batch_size=args.batch_size,
    #             shuffle=True,
    #             collate_fn=ValidDataset.collate_fn
    #         )
    #         cross_valid_iterators[query_type] = new_iterator
    #
    #
    cross_stats = {}
    with open(args.query_dir + "/" + data_name + "_cross_clients_test_queries.pkl", "rb") as fin:
        test_data_dict = pickle.load(fin)
    for query_type, query_answer_dict in test_data_dict.items():
        if query_type not in cross_stats:
            cross_stats[query_type] = len(query_answer_dict)
        else:
            cross_stats[query_type] += len(query_answer_dict)

    length = 0
    for key in cross_stats.keys():
        length += test_stats[key]
    print("cross stats: ", length)
















