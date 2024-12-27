import pdb
import random
from rapidfuzz import fuzz, process
from tqdm import tqdm
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser(description='String matching')
parser.add_argument('--dataset', type=str, default='EN_DE_15K', help='dataset name')
args = parser.parse_args()

def get_entity_set(file_path):
    with open(file_path, 'r') as f:
        return {line.strip().split('\t')[i].split('/')[-1] for line in f for i in [0, 2]}

def load_links(file_path):
    with open(file_path, 'r') as f:
        return [(line.strip().split('\t')[0].split('/')[-1], line.strip().split('\t')[1].split('/')[-1]) for line in f]

def find_matches_for_batch(batch, target_entities):
    results = {}
    for entity in batch:
        # Removed the 'limit=top_n' parameter to get all matches
        top_matches = process.extract(entity, target_entities, scorer=fuzz.ratio, limit=100)
        results[entity] = top_matches  # List of tuples (matched_entity, score)
    return results

def find_matching_pairs(source_entities, target_entities, top_n=10, batch_size=10):
    source_list = list(source_entities)
    target_list = list(target_entities)
    all_matches = {}

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, len(source_list), batch_size):
            batch = source_list[i:i + batch_size]
            futures.append(executor.submit(find_matches_for_batch, batch, target_list))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            all_matches.update(future.result())

    return all_matches

def evaluate_extended(links, predictions, top_n=10):
    hits_at_1 = 0
    hits_at_10 = 0
    reciprocal_ranks = []

    for src, tgt in links:
        if src in predictions:
            matches = predictions[src]
            # Update the enumeration to match the structure of the tuples (entity, score, index)
            ranks = [i for i, (entity, score, _) in enumerate(matches, start=1) if entity == tgt]
            if ranks:
                rank = ranks[0]
                reciprocal_ranks.append(1 / rank)
                if rank == 1:
                    hits_at_1 += 1
                if rank <= top_n:
                    hits_at_10 += 1
            else:
                reciprocal_ranks.append(1/100)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    hit_at_1_rate = hits_at_1 / len(links)
    hit_at_10_rate = hits_at_10 / len(links)

    return hit_at_1_rate, hit_at_10_rate, mrr


data_path = 'data/' + args.dataset
target_file = os.path.join(data_path, 'rel_triples_2')
link_file = os.path.join(data_path, 'ent_links')

links = load_links(link_file)
# shuffle the link order
random.shuffle(links)
target_entities = get_entity_set(target_file)
# sample 10000 source entities
num_evaluated = 1000
source_entities = set(src for src, _ in links[:num_evaluated])

predictions = find_matching_pairs(source_entities, target_entities, top_n=20)

hit_at_1_rate, hit_at_10_rate, mrr = evaluate_extended(links[:num_evaluated], predictions, top_n=20)

print(f'Hit@1: {hit_at_1_rate}, Hit@10: {hit_at_10_rate}, MRR: {mrr}')

top_20_predictions = {src: matches[:20] for src, matches in predictions.items()}
import pickle
with open(os.path.join(data_path, 'top10_match.dict'), 'wb') as f:
    pickle.dump(top_20_predictions, f)