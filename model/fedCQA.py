import copy
import gc
from copy import deepcopy
import torch
import numpy as np
from dataloader import std_offset
import itertools
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

def log_aggregation(list_of_logs):
    all_log = {}

    for __log in list_of_logs:
        # Sometimes the number of answers are 0, so we need to remove all the keys with 0 values
        # The average is taken over all queries, instead of over all answers, as is done following previous work.
        ignore_exd = False
        ignore_ent = False
        ignore_inf = False

        if "exd_num_answers" in __log and __log["exd_num_answers"] == 0:
            ignore_exd = True
        if "ent_num_answers" in __log and __log["ent_num_answers"] == 0:
            ignore_ent = True
        if "inf_num_answers" in __log and __log["inf_num_answers"] == 0:
            ignore_inf = True

        for __key, __value in __log.items():
            if "num_answers" in __key:
                continue

            else:
                if ignore_ent and "ent_" in __key:
                    continue
                if ignore_exd and "exd_" in __key:
                    continue
                if ignore_inf and "inf_" in __key:
                    continue

                if __key in all_log:
                    all_log[__key].append(__value)
                else:
                    all_log[__key] = [__value]

    average_log = {_key: np.mean(_value) for _key, _value in all_log.items()}

    return average_log


class Server:
    def __init__(self, clients, args, entity_mask, relation_mask,
                 train_iterators, valid_iterators, test_iterators, device):
        self.clients = clients
        self.args = args

        self.train_iterators = train_iterators
        self.valid_iterators = valid_iterators
        self.test_iterators = test_iterators

        self.entity_mask = entity_mask
        self.relation_mask = relation_mask

        self.entity_mask_bool = torch.tensor(self.entity_mask, dtype=torch.bool)
        self.relation_mask_bool = torch.tensor(self.relation_mask, dtype=torch.bool)


        if args.server_model == "NA":
            self.model = None
        elif args.server_model == "fedE" or args.server_model == "fedR":
            self.model = copy.deepcopy(clients[0].model).to(device)
        elif args.server_model == "fedC":
            self.model = copy.deepcopy(clients[0].model).to(device)
            self.perturb_embedding_exchange()
        else:
            raise NotImplementedError

        self.device = device

    def perturb_embedding_exchange(self):
        perturb_embedding = 0
        for client_i in self.clients:
            for client_j in self.clients:
                client_i.perturb_embedding_clients.append(client_j.perturb_embedding)

    def update_clients(self, clients_id, entity_update):
        if self.args.server_model == "NA":
            return
        for i in clients_id:
            self.clients[i].update(self.model.state_dict(), clients_id, entity_update)
        
            

    def aggregate_and_update(self, clients_id, data_set, ent_update=None, rel_update=None, optimizer=None):
        if self.args.partial_update:
            if self.args.server_model == "NA":
                return
            elif self.args.server_model == "fedE" or self.args.server_model == "fedC":
                entity_mask = torch.zeros_like(self.entity_mask_bool)
                relation_mask = torch.zeros_like(self.relation_mask_bool)

                if ent_update is not None:
                    for i in range(len(clients_id)):
                        entity_mask[clients_id[i]][ent_update[i]] = True

                if rel_update is not None:
                    for i in range(len(clients_id)):
                        relation_mask[clients_id[i]][rel_update[i]] = True

                aggregated_parameter = deepcopy(data_set[0])
                for key in aggregated_parameter.keys():
                    if key.startswith("entity"):
                        aggregated_parameter[key][~entity_mask[clients_id[0]]] = 0
                    elif key.startswith("relation"):
                        aggregated_parameter[key][~relation_mask[clients_id[0]]] = 0

                for i in range(1, len(data_set)):
                    for key in aggregated_parameter.keys():
                        if key.startswith("entity"):
                            data_set[i][key][~entity_mask[clients_id[i]]] = 0
                            aggregated_parameter[key] += data_set[i][key]
                        elif key.startswith("relation"):
                            data_set[i][key][~relation_mask[clients_id[i]]] = 0
                            aggregated_parameter[key] += data_set[i][key]
                        elif key.startswith("decoder"):
                            continue
                        else:
                            aggregated_parameter[key] += data_set[i][key]

                update_entity = torch.any(entity_mask[clients_id], dim=0)
                update_relation = torch.any(relation_mask[clients_id], dim=0)

                # divide by the number of clients
                divide_entity = [sum(col) for col in zip(*entity_mask[clients_id])]
                divide_entity_tensor = torch.tensor(divide_entity, dtype=torch.float32).to(self.device)
                divide_entity_tensor += ~update_entity.to(self.device)
                divide_relation = [sum(col) for col in zip(*relation_mask[clients_id])]
                divide_relation_tensor = torch.tensor(divide_relation, dtype=torch.float32).to(self.device)
                divide_relation_tensor += ~update_relation.to(self.device)

                original_model_parameter = self.model.state_dict()
                for key in aggregated_parameter.keys():
                    if key.startswith("entity"):
                        original_model_parameter[key][update_entity] = 0
                        aggregated_parameter[key] += original_model_parameter[key]
                        aggregated_parameter[key] /= divide_entity_tensor.view(-1, 1)
                    elif key.startswith("relation"):
                        original_model_parameter[key][update_relation] = 0
                        aggregated_parameter[key] += original_model_parameter[key]
                        aggregated_parameter[key] /= divide_relation_tensor.view(-1, 1)
                    elif key.startswith("decoder"):
                        continue
                    else:
                        aggregated_parameter[key] /= len(clients_id)

                self.model.load_state_dict(aggregated_parameter)

            elif self.args.server_model == "fedR":
                relation_mask = torch.zeros_like(self.relation_mask_bool)

                if rel_update is not None:
                    for i in range(len(clients_id)):
                        relation_mask[clients_id[i]][rel_update[i]] = True

                aggregated_parameter = deepcopy(data_set[0])
                for key in aggregated_parameter.keys():
                    if key.startswith("relation"):
                        aggregated_parameter[key][~relation_mask[clients_id[0]]] = 0

                for i in range(1, len(data_set)):
                    for key in aggregated_parameter.keys():
                        if key.startswith("relation"):
                            data_set[i][key][~relation_mask[clients_id[i]]] = 0
                            aggregated_parameter[key] += data_set[i][key]
                        elif key.startswith("decoder") or key.startswith("entity"):
                            continue
                        else:
                            aggregated_parameter[key] += data_set[i][key]

                update_relation = torch.any(relation_mask[clients_id], dim=0)

                # divide by the number of clients
                divide_relation = [sum(col) for col in zip(*relation_mask[clients_id])]
                divide_relation_tensor = torch.tensor(divide_relation, dtype=torch.float32).to(self.device)
                divide_relation_tensor += ~update_relation.to(self.device)

                original_model_parameter = self.model.state_dict()
                for key in aggregated_parameter.keys():
                    if key.startswith("relation"):
                        original_model_parameter[key][update_relation] = 0
                        aggregated_parameter[key] += original_model_parameter[key]
                        aggregated_parameter[key] /= divide_relation_tensor.view(-1, 1)
                    elif key.startswith("decoder") or key.startswith("entity"):
                        continue
                    else:
                        aggregated_parameter[key] /= len(clients_id)

                self.model.load_state_dict(aggregated_parameter)
            
        else:

            if self.args.server_model == "NA":
                return
            elif self.args.server_model == "fedE" or self.args.server_model == "fedC":
                # aggregate the parameters
                aggregated_parameter = deepcopy(data_set[0])
                for key in aggregated_parameter.keys():
                    if key.startswith("entity"):
                        aggregated_parameter[key][~self.entity_mask_bool[clients_id[0]]] = 0
                    elif key.startswith("relation"):
                        aggregated_parameter[key][~self.relation_mask_bool[clients_id[0]]] = 0

                for i in range(1, len(data_set)):
                    for key in aggregated_parameter.keys():
                        if key.startswith("entity"):
                            data_set[i][key][~self.entity_mask_bool[clients_id[i]]] = 0
                            aggregated_parameter[key] += data_set[i][key]
                        elif key.startswith("relation"):
                            data_set[i][key][~self.relation_mask_bool[clients_id[i]]] = 0
                            aggregated_parameter[key] += data_set[i][key]
                        elif key.startswith("decoder"):
                            continue
                        else:
                            aggregated_parameter[key] += data_set[i][key]
                # divide by the number of clients
                divide_entity = [sum(col) for col in zip(*torch.tensor(self.entity_mask)[clients_id])]
                divide_entity_tensor = torch.tensor(divide_entity, dtype=torch.float32).to(self.device)
                divide_relation = [sum(col) for col in zip(*torch.tensor(self.relation_mask)[clients_id])]
                divide_relation_tensor = torch.tensor(divide_relation, dtype=torch.float32).to(self.device)
                for key in aggregated_parameter.keys():
                    if key.startswith("entity"):
                        aggregated_parameter[key] /= divide_entity_tensor.view(-1, 1)
                    elif key.startswith("relation"):
                        aggregated_parameter[key] /= divide_relation_tensor.view(-1, 1)
                    elif key.startswith("decoder"):
                        continue
                    else:
                        aggregated_parameter[key] /= len(clients_id)

                self.model.load_state_dict(aggregated_parameter)
            elif self.args.server_model == "fedR":
                aggregated_parameter = deepcopy(data_set[0])
                for key in aggregated_parameter.keys():
                    if key.startswith("relation"):
                        aggregated_parameter[key][~self.relation_mask_bool[clients_id[0]]] = 0
                for i in range(1, len(data_set)):
                    for key in aggregated_parameter.keys():
                        if key.startswith("relation"):
                            data_set[i][key][~self.relation_mask_bool[clients_id[i]]] = 0
                            aggregated_parameter[key] += data_set[i][key]
                        elif not key.startswith("decoder"):
                            aggregated_parameter[key] += data_set[i][key]
                # divide by the number of clients
                for key in aggregated_parameter.keys():
                    if not key.startswith("decoder") or not key.startswith("relation"):
                        aggregated_parameter[key] /= len(clients_id)
                self.model.load_state_dict(aggregated_parameter)
            else:
                pass

        if self.args.optimizer_update:
            if optimizer is None:
                raise ValueError("optimizer is None")
            optimizer.load_state_dict(data_set[0])
            for i in range(1, len(data_set)):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        param_state = optimizer.state[p]
                        if 'step' not in param_state or 'exp_avg' not in param_state or 'exp_avg_sq' not in param_state:
                            raise ValueError("optimizer state does not match")
                        param_state['step'] += data_set[i]['step']
                        param_state['exp_avg'] += data_set[i]['exp_avg']
                        param_state['exp_avg_sq'] += data_set[i]['exp_avg_sq']
        
        return 

    def cross_evaluate(self, query_embedding, train_answers, valid_answers):
        if self.args.server_model == "NA" or self.args.server_model == "fedR":
            return None
        elif self.args.server_model == "fedE":
            return self.model.evaluate_generalization(query_embedding, train_answers, valid_answers)
        elif self.args.server_model == "fedC":
            all_scoring_list = []
            for i in range(len(self.clients)):
                all_scoring_list.append(self.clients[i].model.scoring(query_embedding).T)
            all_scoring = torch.zeros_like(all_scoring_list[0])
            for i in range(len(self.clients)):
                all_scoring[self.entity_mask_bool[i]] = all_scoring_list[i][self.entity_mask_bool[i]]
            all_scoring = all_scoring.T
            return self.model.evaluate_generalization(query_embedding, train_answers, valid_answers, all_scoring=all_scoring)
            


    def cross_query(self, batched_query):
        if self.args.server_model == "NA":
            return None
        elif self.args.server_model == "fedE":
            return self.model(batched_query)
        elif self.args.server_model == 'fedC':
            return self.evaluate_from_clients(batched_query)
        else:
            pass

    def evaluate_from_clients(self, batched_structured_query):
        assert batched_structured_query[0] in ["p", "e", "i", "u", "n"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.evaluate_from_clients(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.model.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.model.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.evaluate_from_clients(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.model.intersection(sub_query_result_list)

        elif batched_structured_query[0] == "u":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.evaluate_from_clients(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.model.union(sub_query_result_list)

        elif batched_structured_query[0] == "n":
            sub_query_result = self.evaluate_from_clients(batched_structured_query[1])
            this_query_result = self.model.negation(sub_query_result)

        elif batched_structured_query[0] == "e":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids_device = entity_ids.to(self.model.entity_embedding.weight.device)
            this_query_result = self.model.entity_embedding(entity_ids_device)
            this_query_result = torch.zeros_like(this_query_result).to(entity_ids_device.device)
            this_query_result_list = []
            this_query_result_mask = []
            for i in range(len(self.clients)):
                this_query_result_list.append(self.clients[i].model.entity_embedding(entity_ids_device))
                this_query_result_mask.append(self.entity_mask_bool[i][entity_ids].to(self.model.entity_embedding.weight.device))
                this_query_result[this_query_result_mask[i]] = this_query_result_list[i][this_query_result_mask[i]]

            divide_entity = [sum(col) for col in zip(*this_query_result_mask)]
            divide_entity_tensor = torch.tensor(divide_entity, dtype=torch.float32).to(self.device)
            this_query_result /= divide_entity_tensor.view(-1, 1)

        else:
            this_query_result = None

        return this_query_result




class Client:
    def __init__(self, model, train_iterators, valid_iterators, test_iterators,
                 train_summary_writer, test_summary_writer,
                 args, device, client_id, entity_mask, relation_mask):
        self.model = copy.deepcopy(model).to(device)

        self.train_iterators = train_iterators
        self.valid_iterators = valid_iterators
        self.test_iterators = test_iterators

        self.train_summary_writer = train_summary_writer
        self.test_summary_writer = test_summary_writer

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate
        )

        self.device = device
        self.local_epochs = args.local_epochs
        self.train_iteration_names = list(self.train_iterators.keys())
        self.args = args

        self.local_epoch_total = 0
        self.client_id = client_id

        self.update_ent= None
        self.update_rel= None

        self.entity_mask = entity_mask
        self.relation_mask = relation_mask

        self.entity_mask_bool = torch.tensor(self.entity_mask, dtype=torch.bool)
        self.relation_mask_bool = torch.tensor(self.relation_mask, dtype=torch.bool)

        self.num_entity = len(entity_mask)
        self.embedding_dim = args.entity_space_dim

        if args.server_model == 'fedC':
            self.perturb_embedding = nn.Embedding(self.num_entity, self.embedding_dim).to(device)
            self.perturb_embedding_clients = []


    def upload(self):
        model_state = copy.deepcopy(self.model.state_dict())
        if self.args.server_model == 'fedC':
            model_state['entity_embedding.weight'] += self.perturb_embedding.weight.detach()
        return model_state

    def update(self, parameter, clients_update, entity_update=None):
        parameter = copy.deepcopy(parameter)
        if self.args.server_model == "fedE":
            self.model.load_state_dict(parameter)

        elif self.args.server_model == "fedR":
            model_dict = self.model.state_dict()
            partial_dict = {k: v for k, v in parameter.items()
                            if k in model_dict and not k.startswith("entity") and not k.startswith("decoder")}
            model_dict.update(partial_dict)
            self.model.load_state_dict(model_dict)

        elif self.args.server_model == "fedC":
            model_dict = self.model.state_dict()
            for key in model_dict.keys():
                if key.startswith("entity"):
                    # remove perturb parameters
                
                    perturb_embedding = torch.zeros_like(parameter[key])
                    entity_mask = torch.zeros((len(clients_update),self.num_entity),dtype=torch.bool)
                    for i in range(len(clients_update)):
                        if entity_update is not None:
                            entity_mask[i][entity_update[i]] = True

                        perturb_embedding[entity_mask[i]] += self.perturb_embedding_clients[clients_update[i]].weight.detach()[entity_mask[i]]
                    
                    
                    
                    divide_entity = [sum(col) for col in zip(*entity_mask)]
                    divide_entity_tensor = torch.tensor(divide_entity, dtype=torch.float32).to(self.device)
                    reciprocal_tensor = torch.where(divide_entity_tensor == 0, torch.tensor(0), 1. / divide_entity_tensor)
                    result = reciprocal_tensor.unsqueeze(1) * perturb_embedding
                    parameter[key] -= result
                    parameter[key][~torch.tensor(self.entity_mask, dtype=torch.bool)] = model_dict[key][~torch.tensor(self.entity_mask, dtype=torch.bool)]


                    entity_update_all = list(set().union(*entity_update))
                    entity_mask_all = torch.zeros(self.num_entity, dtype=torch.bool)
                    entity_mask_all[entity_update_all] = True
                    parameter[key][~entity_mask_all] = model_dict[key][~entity_mask_all]



            self.model.load_state_dict(parameter)


    def local_train(self, task_name):
        # if not self.args.server_model == "NA":
        #     self.optimizer = torch.optim.AdamW(
        #         self.model.parameters(),
        #         lr=self.args.learning_rate
        #     )

        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.args.learning_rate)
        self.update_ent = []
        self.update_rel = []
        for i in range(self.local_epochs):
            self.model.train()
            self.local_epoch_total += 1
            # task_name = np.random.choice(self.train_iteration_names)
            iterator = self.train_iterators[task_name]
            batched_query, unified_ids, positive_sample = next(iterator)
            if self.args.partial_update:
                flatten_unified_ids = list(itertools.chain.from_iterable(unified_ids))
                flatten_unified_ids = list(set(flatten_unified_ids))
                flatten_emb_ids = [x - std_offset for x in flatten_unified_ids
                                   if std_offset <= x < std_offset + self.num_entity]
                flatten_rel_ids = [x - std_offset - self.num_entity for x in flatten_unified_ids
                                   if x >= std_offset + self.num_entity]
                self.update_ent = list(set(self.update_ent) | set(flatten_emb_ids) | set(positive_sample))
                self.update_rel = list(set(self.update_rel) | set(flatten_rel_ids))


            loss = self.model(batched_query, positive_sample)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.local_epoch_total % (self.args.log_steps // 2) == 0:
                # self.train_summary_writer.add_scalar("y-train-" + str(self.client_id) + '-' + task_name,
                #                                      loss.item(), self.local_epoch_total)

                model_name = self.args.model



                # Save the model
                if self.args.save_model:
                    self.model.eval()
                    general_checkpoint_path = self.args.checkpoint_path + "/" + model_name + "_" + str(self.client_id) + "_" + str(
                        self.local_epoch_total) + "_" + self.args.data_name + ".bin"

                    torch.save({
                        'steps': self.local_epoch_total,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, general_checkpoint_path)

                print("====== Validation ======: client ID:", self.client_id)

                generalization_logs = []
                generalization_dict = {}

                for task_name, loader in self.valid_iterators.items():
                    all_generalization_logs = []

                    for batched_query, unified_ids, train_answers, valid_answers in loader:


                        query_embedding = self.model(batched_query)
                        generalization_logs_batch = self.model.evaluate_generalization(query_embedding,
                                                                                       train_answers,
                                                                                       valid_answers,
                                                                                       client_evaluate=True,
                                                                                       entity_mask=self.entity_mask)

                        all_generalization_logs.extend(generalization_logs_batch)
                        generalization_logs.extend(generalization_logs_batch)


                    if task_name not in generalization_dict:
                        generalization_dict[task_name] = []
                    generalization_dict[task_name].extend(all_generalization_logs)

                for task_name, logs in generalization_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        self.test_summary_writer.add_scalar("z-valid-" + str(self.client_id) + "-" + task_name + "-" + key,
                                                            value, self.local_epoch_total)

                generalization_logs = log_aggregation(generalization_logs)

                for key, value in generalization_logs.items():
                    self.test_summary_writer.add_scalar("x-valid-" + str(self.client_id) +  "-" + key,
                                                        value, self.local_epoch_total)


                print("====== Testing ======: client ID:", self.client_id)

                generalization_logs = []
                generalization_dict = {}

                for task_name, loader in self.test_iterators.items():
                    all_generalization_logs = []

                    for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:

                        query_embedding = self.model(batched_query)
                        generalization_logs_batch = self.model.evaluate_generalization(query_embedding,
                                                                                       valid_answers,
                                                                                       test_answers,
                                                                                       client_evaluate=True,
                                                                                       entity_mask=self.entity_mask)

                        all_generalization_logs.extend(generalization_logs_batch)
                        generalization_logs.extend(generalization_logs_batch)


                    if task_name not in generalization_dict:
                        generalization_dict[task_name] = []
                    generalization_dict[task_name].extend(all_generalization_logs)

                for task_name, logs in generalization_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        self.test_summary_writer.add_scalar("z-test-" + str(self.client_id) + "-" + task_name + "-" + key,
                                                            value, self.local_epoch_total)

                generalization_logs = log_aggregation(generalization_logs)

                for key, value in generalization_logs.items():
                    self.test_summary_writer.add_scalar("x-test-" + str(self.client_id) + "-" + key,
                                                        value, self.local_epoch_total)

                gc.collect()

        return self.upload()






