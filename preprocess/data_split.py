

from random import sample, choice, random, randint,shuffle


dir_list = ["../KG_data/"]
output_dir = "../KG_data_fed/"
data_names = ["NELL-betae", "FB15k-237-betae", "FB15k-betae"]
num_clients = 10

def dataset_split_with_relation(file_name, num_relation=0):
    triple_list = []
    # (h, t, r)
    if isinstance(file_name, list):
        for _file_name in file_name:
            with open(_file_name, "r") as file_in:
                for line in file_in:
                    line_list = line.strip().split("\t")

                    line_numerical_list = [line_list[0], line_list[2], line_list[1]]
                    triple_list.append(line_numerical_list)

    else:
        with open(file_name, "r") as file_in:
            for line in file_in:
                line_list = line.strip().split("\t")

                line_numerical_list = [line_list[0], line_list[2], line_list[1]]
                triple_list.append(line_numerical_list)

    def classify_list(lst):
        classified_list = []

        for item in lst:
            relation = item[2]
            found = False

            for sublist in classified_list:
                if sublist[0][2] == relation:
                    sublist.append(item)
                    found = True
                    break
            if not found:
                classified_list.append([item])
        return classified_list
    if num_relation == 0:
        triple_list_split = classify_list(triple_list)
    else:
        triple_list_split = []
        for i in range(num_relation):
            triple_list_split.append([])

        for triple in triple_list:
            triple_list_split[int(triple[2])].append(triple)

    return triple_list_split



if __name__ == '__main__':
    for data_dir in data_names:
        print(data_dir)
        print('Split Train Graph:' + data_dir)
        train_path = "../KG_data/" + data_dir + "/train.txt"
        train_triple_split = dataset_split_with_relation(train_path)
        # print("nodes:", train_graph.number_of_nodes())
        # print("edges:", train_graph.number_of_edges())

        # generate random groups
        num_relations = len(train_triple_split)
        nums = list(range(num_relations))
        shuffle(nums)
        print("number of relations:", len(train_triple_split))
        group_size = len(train_triple_split) // num_clients
        remaining = len(train_triple_split) % num_clients
        groups = []
        start = 0
        for i in range(num_clients):
            end = start + group_size

            if remaining > 0:
                end += 1
                remaining -= 1

            group = nums[start:end]
            groups.append(group)

            start = end

        # split train data
        train_triple_clients = []
        for i in range(num_clients):
            client_triples = []
            for j in groups[i]:
                client_triples += train_triple_split[j]
            train_triple_clients.append(client_triples)

        # split valid data
        valid_path = "../KG_data/" + data_dir + "/valid.txt"
        valid_triple_split = dataset_split_with_relation(valid_path, num_relations)
        valid_triple_clients = []
        for i in range(num_clients):
            client_triples = []
            for j in groups[i]:
                client_triples += valid_triple_split[j]
            valid_triple_clients.append(client_triples)

        # split test data
        test_path = "../KG_data/" + data_dir + "/test.txt"
        test_triple_split = dataset_split_with_relation(test_path, num_relations)
        test_triple_clients = []
        for i in range(num_clients):
            client_triples = []
            for j in groups[i]:
                client_triples += test_triple_split[j]
            test_triple_clients.append(client_triples)

        # save data
        for i in range(num_clients):
            train_file = open(output_dir + data_dir + "/fed-" + str(num_clients) + "/" + "train_" + str(i) + ".txt", "w")
            for triple in train_triple_clients[i]:
                train_file.write(triple[0] + "\t" + triple[2] + "\t" + triple[1] + "\n")
            train_file.close()

            valid_file = open(output_dir + data_dir + "/fed-" + str(num_clients) + "/" + "valid_" + str(i) + ".txt", "w")
            for triple in valid_triple_clients[i]:
                valid_file.write(triple[0] + "\t" + triple[2] + "\t" + triple[1] + "\n")
            valid_file.close()

            test_file = open(output_dir + data_dir + "/fed-" + str(num_clients) + "/" + "test_" + str(i) + ".txt", "w")
            for triple in test_triple_clients[i]:
                test_file.write(triple[0] + "\t" + triple[2] + "\t" + triple[1] + "\n")
            test_file.close()


