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

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args.checkpoint_path + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name[
                                                                                                                    :-6] + '/train'
    test_log_dir = args.checkpoint_path + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name[
                                                                                                                   :-6] + '/test'

    if args.debug:
        train_log_dir = args.checkpoint_path + '/gradient_tape/'+ "_train_" + current_time
        test_log_dir = args.checkpoint_path + '/gradient_tape/' + "_test_" + current_time

    # train_summary_writer = SummaryWriter(train_log_dir)
    train_summary_writer = None
    test_summary_writer = SummaryWriter(test_log_dir)

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
    for i in range(args.n_clients):
        # clients.append(Client())
        client_entity_mask = [0] * nentity
        client_relation_mask = [0] * nrelation
        train_path = "../KG_data_fed/" + data_name + "/fed-"+ str(args.n_clients) + "/train_" + str(i) + ".txt"
        with open(train_path, "r") as file_in:
            for line in file_in:
                line_list = line.strip().split("\t")
                client_entity_mask[int(line_list[0])] = 1
                client_entity_mask[int(line_list[2])] = 1
                client_relation_mask[int(line_list[1])] = 1

        entity_mask.append(client_entity_mask)
        relation_mask.append(client_relation_mask)

        print("====== Initialize Model ======", args.model)
        if args.model == 'gqe':
            model = GQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim,
                        use_old_loss=args.old_loss_fnt)
        elif args.model == 'q2b':
            model = Q2B(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim,
                        use_old_loss=args.old_loss_fnt)
        elif args.model == "q2p":
            model = Q2P(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
        elif args.model == "tree_lstm":
            model = TreeLSTM(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
        else:
            raise NotImplementedError

        models_clients.append(model)

        train_query_file_names.append(data_name + "_train_queries_" + str(i) + ".pkl")
        valid_query_file_names.append(data_name + "_valid_queries_" + str(i) + ".pkl")
        test_query_file_names.append(data_name + "_test_queries_" + str(i) + ".pkl")

        with open(args.query_dir + "/" + train_query_file_names[i], "rb") as fin:
            train_data_dict = pickle.load(fin)

        train_iterators = {}
        for query_type, query_answer_dict in train_data_dict.items():

            new_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(nentity, nrelation, query_answer_dict),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            ))
            train_iterators[query_type] = new_iterator

        with open(args.query_dir + "/" + valid_query_file_names[i], "rb") as fin:
            valid_data_dict = pickle.load(fin)

        valid_iterators = {}
        for query_type, query_answer_dict in valid_data_dict.items():
            new_iterator = DataLoader(
                ValidDataset(nentity, nrelation, query_answer_dict),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=ValidDataset.collate_fn
            )
            valid_iterators[query_type] = new_iterator

        with open(args.query_dir + "/" + test_query_file_names[i], "rb") as fin:
            test_data_dict = pickle.load(fin)

        test_iterators = {}
        for query_type, query_answer_dict in test_data_dict.items():
            new_iterator = DataLoader(
                TestDataset(nentity, nrelation, query_answer_dict),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=TestDataset.collate_fn
            )
            test_iterators[query_type] = new_iterator

        train_iterators_clients.append(train_iterators)
        valid_iterators_clients.append(valid_iterators)
        test_iterators_clients.append(test_iterators)
    for i in range(args.n_clients):
        clients.append(Client(models_clients[i],
                              train_iterators_clients[i], valid_iterators_clients[i], test_iterators_clients[i],
                              train_summary_writer, test_summary_writer,
                              args, device, i, entity_mask[i], relation_mask[i]))


    cross_train_iterators = {}
    cross_valid_iterators = {}
    cross_test_iterators = {}

    # cross data preparation
    if args.evaluate_cross:
        # with open(args.query_dir + "/" + data_name + "_cross_clients_train_queries.pkl", "rb") as fin:
        #     train_data_dict = pickle.load(fin)
        # for query_type, query_answer_dict in train_data_dict.items():
        #     new_iterator = SingledirectionalOneShotIterator(DataLoader(
        #         TrainDataset(nentity, nrelation, query_answer_dict),
        #         batch_size=batch_size,
        #         shuffle=True,
        #         collate_fn=TrainDataset.collate_fn
        #     ))
        #     cross_train_iterators[query_type] = new_iterator

        with open(args.query_dir + "/" + data_name + "_cross_clients_valid_queries.pkl", "rb") as fin:
            valid_data_dict = pickle.load(fin)
        for query_type, query_answer_dict in valid_data_dict.items():
            new_iterator = DataLoader(
                ValidDataset(nentity, nrelation, query_answer_dict),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=ValidDataset.collate_fn
            )
            cross_valid_iterators[query_type] = new_iterator


        with open(args.query_dir + "/" + data_name + "_cross_clients_test_queries.pkl", "rb") as fin:
            test_data_dict = pickle.load(fin)
        for query_type, query_answer_dict in test_data_dict.items():
            new_iterator = DataLoader(
                TestDataset(nentity, nrelation, query_answer_dict),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=TestDataset.collate_fn
            )
            cross_test_iterators[query_type] = new_iterator


    server = Server(clients, args, entity_mask, relation_mask,
                    cross_train_iterators, cross_valid_iterators, cross_test_iterators, device)



    # Train the model
    print("====== Training ======", args.model)
    train_iteration_names = list(train_iterators_clients[0].keys())
    for step in tqdm(range(args.global_steps)):
        n_sample = max(round(args.fraction * len(clients)), 1)
        sample_id = np.random.choice(range(len(clients)), n_sample, replace=False)

        
        task_name = np.random.choice(train_iteration_names)
        client_data = []
        ent_update = []
        rel_update = []
        optimizer_update = []
        for client_id in sample_id:
            client_data.append(clients[client_id].local_train(task_name))
            ent_update.append(clients[client_id].update_ent)
            rel_update.append(clients[client_id].update_rel)
            optimizer_update.append(clients[client_id].optimizer)

        server.aggregate_and_update(sample_id, client_data, ent_update, rel_update, optimizer_update)
        server.update_clients(sample_id, ent_update)

        if args.evaluate_cross and step % args.log_steps == 0:
            print("====== Validation Server ======")

            generalization_logs = []
            generalization_dict = {}

            for task_name, loader in cross_valid_iterators.items():
                all_generalization_logs = []

                for batched_query, unified_ids, train_answers, valid_answers in loader:
                    query_embedding = server.cross_query(batched_query)
                    generalization_logs_batch = server.cross_evaluate(query_embedding, train_answers, valid_answers)

                    all_generalization_logs.extend(generalization_logs_batch)
                    generalization_logs.extend(generalization_logs_batch)

                if task_name not in generalization_dict:
                    generalization_dict[task_name] = []
                generalization_dict[task_name].extend(all_generalization_logs)

            for task_name, logs in generalization_dict.items():
                aggregated_generalization_logs = log_aggregation(logs)
                for key, value in aggregated_generalization_logs.items():
                    test_summary_writer.add_scalar("z-valid-cross-" + task_name + "-" + key, value, step)

            generalization_logs = log_aggregation(generalization_logs)

            for key, value in generalization_logs.items():
                test_summary_writer.add_scalar("x-valid-cross-" + key, value, step)

            print("====== Testing Server ======")

            generalization_logs = []
            generalization_dict = {}

            for task_name, loader in cross_test_iterators.items():
                all_generalization_logs = []

                for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
                    query_embedding = server.cross_query(batched_query)
                    generalization_logs_batch = server.cross_evaluate(query_embedding, valid_answers, test_answers)

                    all_generalization_logs.extend(generalization_logs_batch)
                    generalization_logs.extend(generalization_logs_batch)

                if task_name not in generalization_dict:
                    generalization_dict[task_name] = []
                generalization_dict[task_name].extend(all_generalization_logs)

            for task_name, logs in generalization_dict.items():
                aggregated_generalization_logs = log_aggregation(logs)
                for key, value in aggregated_generalization_logs.items():
                    test_summary_writer.add_scalar("z-test-cross-" + task_name + "-" + key, value, step)

            generalization_logs = log_aggregation(generalization_logs)

            for key, value in generalization_logs.items():
                test_summary_writer.add_scalar("x-test-cross-" + key, value, step)

            gc.collect()


















