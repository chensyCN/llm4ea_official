import os
import pdb
import pickle
import argparse


from objects.KG import KG
from objects.KGs import KGs
from ea.model import EntityAlignmentModel
from config import Config


argparser = argparse.ArgumentParser()
argparser.add_argument('--bottomk', action='store_true')
argparser.add_argument('--topk_match', type=int, default=10)
argparser.add_argument('--query_scheme', type=str, default="aggregated")
argparser.add_argument('--dataset', type=str, default="D_W_15K")
argparser.add_argument('--iter', type=int, default=3)
argparser.add_argument('--tpr', type=float, default=0.95)
argparser.add_argument('--simulate', action='store_true', help="simulate the label annotation process, used only in case studies or have no access to llm api, by default False")
argparser.add_argument('--budget', type=float, default=0.1, help="ratio of the number of inserted pairs to the number of entities in KG1")
args = argparser.parse_args()

Config.init_with_attr = False
print(f"init_with_attr: {Config.init_with_attr}")
Config.query_scheme = args.query_scheme
Config.simulate = args.simulate

def construct_kg(path_r, path_a=None, sep='\t', name=None):
    kg = KG(name=name)
    if path_a is not None:
        with open(path_r, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_relation_tuple(h, r, t)

        with open(path_a, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                # assert len(params) == 3
                e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_attribute_tuple(e, a, v)
    else:
        with open(path_r, "r", encoding="utf-8") as f:
            prev_line = ""
            for line in f.readlines():
                params = line.strip().split(sep)
                if len(params) != 3 or len(prev_line) == 0:
                    prev_line += "\n" if len(line.strip()) == 0 else line.strip()
                    continue
                prev_params = prev_line.strip().split(sep)
                e, a, v = prev_params[0].strip(), prev_params[1].strip(), prev_params[2].strip()
                prev_line = "".join(line)
                if len(e) == 0 or len(a) == 0 or len(v) == 0:
                    print("Exception: " + e)
                    continue
                if v.__contains__("http"):
                    kg.insert_relation_tuple(e, a, v)
                else:
                    kg.insert_attribute_tuple(e, a, v)
    kg.init()
    kg.print_kg_info()
    return kg


def get_top_match(f_path):
    top_match = pickle.load(open(f_path, "rb"))
    top_match = {k: {x[0] for x in v} for k, v in top_match.items()}
    print(f"loaded top match from {f_path}, one example of top match k-v: {list(top_match.items())[0]}")
    return top_match


def construct_kgs(dataset_dir, name="KGs", load_chk=None):
    path_r_1 = os.path.join(dataset_dir, "rel_triples_1")
    path_a_1 = os.path.join(dataset_dir, "attr_triples_1")

    path_r_2 = os.path.join(dataset_dir, "rel_triples_2")
    path_a_2 = os.path.join(dataset_dir, "attr_triples_2")

    kg1 = construct_kg(path_r_1, path_a_1, name=str(name + "-KG1"))
    kg2 = construct_kg(path_r_2, path_a_2, name=str(name + "-KG2"))
    kgs = KGs(kg1=kg1, kg2=kg2, ground_truth_path=os.path.join(dataset_path, "ent_links"))
    # load the previously saved PRASE model
    if load_chk is not None:
        kgs.util.load_params(load_chk)
    
    topk_match_path = os.path.join(dataset_path, f"top{args.topk_match}_match.dict")
    topk_match = get_top_match(topk_match_path)
    kgs.topk_match = topk_match

    return kgs


def align(kgs):

    iter = args.iter
    tpr = args.tpr
    pairBudget = int(len(kgs.kg_l.entity_set) * args.budget)
    pairPerIter = pairBudget // iter

    # active selection process
    for i in range(iter):
        print(f"Inserting {pairPerIter} pairs in iteration {i}...")
        kgs.generate_labels(budget=pairPerIter, tpr=tpr)
        # label refine
        kgs.set_iteration(10)
        # kgs.run()
        # use refined labels to train EA model
        data, bias = kgs.util.generate_input_for_emb_model_active_only()
        print(f"bias: {bias}")
        if i == 0:
            ea_model = EntityAlignmentModel(data)
        else:
            train_pair= data[-2]
            ea_model.reset_data(train_pair)
        new_pairs = ea_model.train(epoch=20)

        # feed the inferred pairs into label refiner, for the next iteration
        kgs.inject_ea_inferred_pairs(new_pairs, bias[0], filter=True, reinject=True)
        print(f"propagate alignment confidence by inference...")
        # kgs.set_iteration(10)
        kgs.run()

    # train the EA model
    data, bias = kgs.util.generate_input_for_emb_model_active_only()
    ea_model = EntityAlignmentModel(data)
    ea_model.fine_tune()



if __name__ == '__main__':

    print(f"\nExp config:\n {Config()}\n")

    base, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = args.dataset
    dataset_path = os.path.join(os.path.join(base, "data"), dataset_name)
    topk_match_path = os.path.join(dataset_path, f"top{args.topk_match}_match.dict")

    print("Construct KGs...")
    kgs = construct_kgs(dataset_dir=dataset_path, name=dataset_name, load_chk=None)


    num_workers = os.cpu_count()
    kgs.set_worker_num(num_workers)
    
    kgs.set_iteration(20)

    align(kgs=kgs)