import gc
import sys
import os
import time
import random
import pdb
import numpy as np
import pickle
from objects.KG import KG
import multiprocessing as mp
import json
pj_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pj_path)
from config import Config
from annotator import Annotator
from probabilisticReasoning import one_iteration_one_way


sys.setrecursionlimit(1000000)


class KGs:
    def __init__(self, kg1: KG, kg2: KG, theta=0.1, iteration=3, workers=4, fusion_func=None, ground_truth_path=None):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = theta
        self.iteration = iteration
        self.delta = 0.01
        self.epsilon = 1.01
        self.const = 10.0
        self.workers = workers
        self.fusion_func = fusion_func

        self.rel_ongoing_dict_l, self.rel_ongoing_dict_r = dict(), dict()
        self.rel_norm_dict_l, self.rel_norm_dict_r = dict(), dict()
        self.rel_align_dict_l, self.rel_align_dict_r = dict(), dict()

        self.sub_ent_match = None
        self.sup_ent_match = None
        self.sub_ent_prob = None
        self.sup_ent_prob = None

        self.topk_match = None
        self.annotator = Annotator(Config.gpt_api_key)

        self.total_inserted_ent_align = 0

        self._iter_num = 0
        self.has_load = False
        self.util = KGsUtil(self, self.__get_counterpart_and_prob, self.__set_counterpart_and_prob)
        self.__init(ground_truth_path=ground_truth_path)

    def __init(self, ground_truth_path=None, ratio=0.3):
        if not self.kg_l.is_init():
            self.kg_l.init()
        if not self.kg_r.is_init():
            self.kg_r.init()

        self.gold_result = set() # for evaluation or simulation
        with open(ground_truth_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split("\t")
                ent_l, ent_r = params[0].strip(), params[1].strip()
                obj_l, obj_r = self.kg_l.entity_dict_by_name.get(ent_l), self.kg_r.entity_dict_by_name.get(
                    ent_r)
                if obj_l is None:
                    print("Exception: fail to load Entity (" + ent_l + ")")
                if obj_r is None:
                    print("Exception: fail to load Entity (" + ent_r + ")")
                if obj_l is None or obj_r is None:
                    continue
                self.gold_result.add((obj_l.id, obj_r.id))

        kg_l_ent_num = len(self.kg_l.entity_set) + len(self.kg_l.literal_set)
        kg_r_ent_num = len(self.kg_r.entity_set) + len(self.kg_r.literal_set)
        self.sub_ent_match = [None for _ in range(kg_l_ent_num)]
        self.sub_ent_prob = [0.0 for _ in range(kg_l_ent_num)]
        self.sup_ent_match = [None for _ in range(kg_r_ent_num)]
        self.sup_ent_prob = [0.0 for _ in range(kg_r_ent_num)]

        self.annotated_alignments = set()

    def __get_counterpart_and_prob(self, ent):
        source = ent.affiliation is self.kg_l
        counterpart_id = self.sub_ent_match[ent.id] if source else self.sup_ent_match[ent.id]
        if counterpart_id is None:
            return None, 0.0
        else:
            counterpart = self.kg_r.ent_lite_list_by_id[counterpart_id] if source \
                else self.kg_l.ent_lite_list_by_id[counterpart_id]
            return counterpart, self.sub_ent_prob[ent.id] if source else self.sup_ent_prob[ent.id]

    def __set_counterpart_and_prob(self, ent_l, ent_r, prob, force=False):
        source = ent_l.affiliation is self.kg_l
        l_id, r_id = ent_l.id, ent_r.id
        curr_prob = self.sub_ent_prob[l_id] if source else self.sup_ent_prob[l_id]
        if not force and prob < curr_prob:
            return False
        if source:
            self.sub_ent_match[l_id], self.sub_ent_prob[l_id] = r_id, prob
        else:
            self.sup_ent_match[l_id], self.sup_ent_prob[l_id] = r_id, prob
        return True

#===================================================================================================
# Functions for active seletion of source entities
#===================================================================================================

    def compute_neighbor_uncertainty(self, ent_id):
        uncertainty = 0.0
        ent_l = self.kg_l.entity_dict_by_id[ent_id]
        for rel in ent_l.involved_as_head_dict.keys():
            if rel.is_attribute():
                continue
            for tail in ent_l.involved_as_head_dict[rel]:
                if self.sub_ent_match[tail.id] is None:
                    uncertainty += 1.0
                else:
                    uncertainty += 1.0 - self.sub_ent_prob[tail.id]
        return uncertainty

    def compute_ent_degree(self, ent_id):
        deg = 0.0
        is_attribute = 0.0
        ent_l = self.kg_l.entity_dict_by_id[ent_id]
        for rel in ent_l.involved_as_head_dict.keys():
            if rel.is_attribute():
                continue
            deg += len(ent_l.involved_as_head_dict[rel])
            if rel.is_attribute():
                is_attribute += 1.0
        return deg

    def compute_functionality_sum(self, ent_id):
        func_sum = 0.0
        ent_l = self.kg_l.entity_dict_by_id[ent_id]
        for rel in ent_l.involved_as_head_dict.keys():
            if rel.is_attribute():
                continue
            functionality = rel.functionality
            func_sum += functionality * len(ent_l.involved_as_head_dict[rel])
        return func_sum

    def compute_relational_uncertainty(self, ent_id):
        rel_uncert = 0.0
        ent_l = self.kg_l.entity_dict_by_id[ent_id]
        for rel in ent_l.involved_as_head_dict.keys():
            if rel.is_attribute():
                continue
            for tail in ent_l.involved_as_head_dict[rel]:
                if self.sub_ent_match[tail.id] is None:
                    rel_uncert += rel.functionality
                else:
                    rel_uncert += (1.0 - self.sub_ent_prob[tail.id]) * rel.functionality
        return rel_uncert

    @staticmethod
    def re_sort_by_aggregated_ranks(ids1, ids2):
        """
        ids1 and ids2 are two list of ids sorted by different criterias, 
        they share the same ids but in different orders,
        the function returns a sorted list of ids by the mean reciprocal rank of the two lists
        """
        id2rank1 = {id: i+1 for i, id in enumerate(ids1)}
        id2rank2 = {id: i+1 for i, id in enumerate(ids2)}
        id2mrr = dict()
        for id in ids1:
            id2mrr[id] = (1 / id2rank1[id] + 1 / id2rank2[id]) / 2
        sorted_ids = sorted(id2mrr, key=lambda x: id2mrr[x], reverse=True)
        return sorted_ids

    def __select_source_entities(self, not_matched_kg_l_ent_id, not_confident_kg_l_ent_id, budget):
        src_ent_ids = not_matched_kg_l_ent_id + not_confident_kg_l_ent_id

        if Config.query_scheme == "by_neighbor_uncertainty":
            scores = [self.compute_neighbor_uncertainty(ent_id) for ent_id in src_ent_ids]
        elif Config.query_scheme == "by_degree":
            scores = [self.compute_ent_degree(ent_id) for ent_id in src_ent_ids]
        elif Config.query_scheme == "by_functionality_sum":
            scores = [self.compute_functionality_sum(ent_id) for ent_id in src_ent_ids]
        elif Config.query_scheme == "by_relational_uncertainty":
            scores = [self.compute_relational_uncertainty(ent_id) for ent_id in src_ent_ids]
        elif Config.query_scheme == "aggregated":
            uncert_scores = [self.compute_neighbor_uncertainty(ent_id) for ent_id in src_ent_ids]
            sorted_uncert_idxs = np.argsort(uncert_scores)[::-1]
            rel_uncert_scores = [self.compute_relational_uncertainty(ent_id) for ent_id in src_ent_ids]
            sorted_rel_uncert_idxs = np.argsort(rel_uncert_scores)[::-1]
            aggregated = self.re_sort_by_aggregated_ranks(sorted_uncert_idxs, sorted_rel_uncert_idxs)
            selected_ent_ids = [src_ent_ids[i] for i in aggregated[:budget]]
            return selected_ent_ids
        
        sorted_ent_idxs = np.argsort(scores)[::-1]
        selected_ent_ids = [src_ent_ids[i] for i in sorted_ent_idxs[:budget]]
        return selected_ent_ids

#===================================================================================================
# Functions for annotation
#===================================================================================================

    def generate_labels(self, budget, tpr=0.5):
        not_matched_kg_l_ent_id = set()
        not_confident_kg_l_ent_id = set()

        annotated_ent_id_l = set([ent_id for (ent_id, _) in self.annotated_alignments])

        for ent_l in self.kg_l.entity_set:
            if ent_l.id in annotated_ent_id_l:
                continue
            if self.sub_ent_match[ent_l.id] is None:
                not_matched_kg_l_ent_id.add(ent_l.id)
            else:
                not_confident_kg_l_ent_id.add(ent_l.id)
        not_matched_kg_l_ent_id = list(not_matched_kg_l_ent_id)
        not_confident_kg_l_ent_id = list(not_confident_kg_l_ent_id)
        print(f"Number of not matched entities: {len(not_matched_kg_l_ent_id)}")
        print(f"Number of not confident entities: {len(not_confident_kg_l_ent_id)}")

        selected_ent_ids = self.__select_source_entities(not_matched_kg_l_ent_id, not_confident_kg_l_ent_id, budget)

        if Config.simulate == True:
            self.__simulate_annotate(selected_ent_ids, tpr)
        else:
            self.__annotate(selected_ent_ids)

    def __simulate_annotate(self, selected_ent_ids, tpr=0.5):
        inserted_true = 0
        inserted_false = 0

        gold_result_dict_l2r = dict()
        for (l, r) in self.gold_result:
            if l in selected_ent_ids:
                gold_result_dict_l2r[l] = r
        
        # pdb.set_trace()
        for ent_l_id in selected_ent_ids:
            # get the value of this entity
            ent_l_val = self.kg_l.entity_dict_by_id[ent_l_id].value
            if ent_l_id in gold_result_dict_l2r:
                # check whether the value of the true target entity is in the topk match of the source entity
                ent_r_id = gold_result_dict_l2r[ent_l_id]
                ent_r_val = self.kg_r.entity_dict_by_id[ent_r_id].value
                if ent_r_val in self.topk_match[ent_l_val]:
                    if float(tpr) == 1.0 or inserted_true < inserted_false * tpr / (1 - tpr):
                        self.annotated_alignments.add((ent_l_id, ent_r_id))
                        self.sub_ent_match[ent_l_id], self.sub_ent_prob[ent_l_id] = ent_r_id, 0.5
                        self.sup_ent_match[ent_r_id], self.sup_ent_prob[ent_r_id] = ent_l_id, 0.5
                        inserted_true += 1
                    else:
                        replaced_ent_r_val = random.choice(list(self.topk_match[ent_l_val]-{ent_r_val}))
                        if replaced_ent_r_val not in self.kg_r.entity_dict_by_value:
                            continue
                        replaced_ent_r_id = self.kg_r.entity_dict_by_value[replaced_ent_r_val].id
                        self.annotated_alignments.add((ent_l_id, replaced_ent_r_id))
                        self.sub_ent_match[ent_l_id], self.sub_ent_prob[ent_l_id] = replaced_ent_r_id, 0.5
                        self.sup_ent_match[replaced_ent_r_id], self.sup_ent_prob[replaced_ent_r_id] = ent_l_id, 0.5
                        inserted_false += 1

        print(f"Number of inserted true entities: {inserted_true} and false entities: {inserted_false}")

    def __annotate(self, selected_ent_ids):
        inserted_true = 0
        inserted_false = 0

        gold_result_dict_l2r = dict()
        for (l, r) in self.gold_result:
            if l in selected_ent_ids:
                gold_result_dict_l2r[l] = r


        for ent_l_id in selected_ent_ids:
            ent_l_val = self.kg_l.entity_dict_by_id[ent_l_id].value
            candidates = set(entity for (entity, _, _) in self.topk_match[ent_l_val])
            candidates_str = str(candidates)
            pred = self.annotator.predict(ent_l_val, candidates_str)
            if pred not in candidates or pred not in self.kg_r.entity_dict_by_value: # check the validity of the predicted entity
                print(f"Error: the predicted entity {pred} is not in the candidates")
                continue
            ent_r_id = self.kg_r.entity_dict_by_value[pred].id
            self.annotated_alignments.add((ent_l_id, ent_r_id))
            self.sub_ent_match[ent_l_id], self.sub_ent_prob[ent_l_id] = ent_r_id, 0.5
            self.sup_ent_match[ent_r_id], self.sup_ent_prob[ent_r_id] = ent_l_id, 0.5

            if ent_l_id in gold_result_dict_l2r:
                if ent_r_id == gold_result_dict_l2r[ent_l_id]:
                    inserted_true += 1
                else:
                    inserted_false += 1
        
        print(f"Number of inserted true entities: {inserted_true} and false entities: {inserted_false}")


#####################################################################################################
# Functions for label refine
#####################################################################################################

    def refine_labels(self, threshold=Config.initial_alignment_score):
        """
        label refinement, by flipping the labels of low confidence alignments
        """
        inferred_alignments = set()
        for ent_id in self.kg_l.ent_id_list:
            counterpart_id = self.sub_ent_match[ent_id]
            if counterpart_id is not None:
                prob = self.sub_ent_prob[ent_id]
                if prob > threshold:
                    inferred_alignments.add((ent_id, counterpart_id))
        self.refined_alignments = inferred_alignments.intersection(self.annotated_alignments)
        self._enforce_refined_labels(score=0.9)

    def _enforce_refined_labels(self, score):
        """
        set the score of the refined alignments to be at least 0.9
        """
        for ent_l_id, ent_r_id in self.refined_alignments:
            self.sub_ent_prob[ent_l_id] = max(self.sub_ent_prob[ent_l_id], score)
            self.sup_ent_prob[ent_r_id] = max(self.sup_ent_prob[ent_r_id], score)

    def update_annotations(self):
        self.annotated_alignments = self.refined_alignments

    def inject_ea_inferred_pairs(self, pairs, ent_bias, filter=False, reinject=False):
        injected_pair = 0
        for (l, r) in pairs:
            r = r - ent_bias
            # for the reinjection, the inferred paris will override the previous ea inferred paris
            if reinject:
                if (l, r) in self.annotated_alignments:
                    continue
            else:
                if self.sub_ent_match[l] and self.sub_ent_prob[l] >= 0.9 and self.sub_ent_match[r] and self.sub_ent_prob[r] >= 0.9:
                    continue
            self.sub_ent_match[l], self.sub_ent_prob[l] = r, 0.5
            self.sup_ent_match[r], self.sup_ent_prob[r] = l, 0.5
            injected_pair += 1
            if filter:
                continue
            else:
                self.annotated_alignments.add((l, r))
        if filter:
            print(f"Injected {injected_pair} pairs out of {len(pairs)} pairs, but not include in the annotated alignments")
        else:
            print(f"Injected {injected_pair} pairs out of {len(pairs)} pairs")

    def reset_annotation_prob(self, prob=0.5):

        kg_l_ent_num = len(self.kg_l.entity_set) + len(self.kg_l.literal_set)
        kg_r_ent_num = len(self.kg_r.entity_set) + len(self.kg_r.literal_set)
        self.sub_ent_match = [None for _ in range(kg_l_ent_num)]
        self.sub_ent_prob = [0.0 for _ in range(kg_l_ent_num)]
        self.sup_ent_match = [None for _ in range(kg_r_ent_num)]
        self.sup_ent_prob = [0.0 for _ in range(kg_r_ent_num)]

        if Config.init_with_attr:
            num_match_attr = 0
            for lite_l in self.kg_l.literal_set:
                if self.kg_r.literal_dict_by_value.__contains__(lite_l.value):
                    lite_r = self.kg_r.literal_dict_by_value[lite_l.value]
                    l_id, r_id = lite_l.id, lite_r.id
                    self.sub_ent_match[l_id], self.sup_ent_match[r_id] = lite_r.id, lite_l.id
                    self.sub_ent_prob[l_id], self.sup_ent_prob[r_id] = 1.0, 1.0
                    num_match_attr += 1

        # reset the probability of annotated pairs
        for (l, r) in self.annotated_alignments:
            self.sub_ent_match[l], self.sub_ent_prob[l] = r, prob
            self.sup_ent_match[r], self.sup_ent_prob[r] = l, prob

    def set_fusion_func(self, func):
        self.fusion_func = func

    def set_iteration(self, iteration):
        self.iteration = iteration

    def set_worker_num(self, worker_num):
        self.workers = worker_num

    def warm_up(self):
        start_time = time.time()
        for i in range(self.iteration):
            self._iter_num = i
            print(str(i + 1) + "-th iteration......")
            self.__run_per_iteration()
            gc.collect()
        end_time = time.time()
        print("Warm up completed!")
        print("Total time: " + str(end_time - start_time))

    def run(self):
        start_time = time.time()
        print("Label refine ...... ")
        # print("Initial alignment......")
        self.util.test(gold_result=self.gold_result, threshold=[0.1 * i for i in range(10)])
        for i in range(self.iteration):
            self._iter_num = i
            print(str(i + 1) + "-th iteration......")
            self.__run_per_iteration()
            if Config.label_refine:
                self.refine_labels()
            self.util.test(gold_result=self.gold_result, threshold=[0.1 * i for i in range(10)])
            # self.util.test_refinement_at_all_threshold(gold_result=self.gold_result, annotated_alignments=self.annotated_alignments, threshold=[0.1 * i for i in range(10)])
            self.util.test_refined_alignments()
            gc.collect()
        print("Probabilistic Inference Completed!")
        end_time = time.time()
        print("Total time: " + str(end_time - start_time))

    def __run_per_iteration(self):
        self.__run_per_iteration_one_way(self.kg_l)
        self.__ent_bipartite_matching()
        self.__run_per_iteration_one_way(self.kg_r, ent_align=False)
        return

    def __run_per_iteration_one_way(self, kg: KG, ent_align=True):
        kg_other = self.kg_l if kg is self.kg_r else self.kg_r
        ent_list = self.__generate_list(kg)
        mgr = mp.Manager()
        ent_queue = mgr.Queue(len(ent_list))
        for ent_id in ent_list:
            ent_queue.put(ent_id)

        rel_ongoing_dict_queue = mgr.Queue()
        rel_norm_dict_queue = mgr.Queue()
        ent_match_tuple_queue = mgr.Queue()

        kg_r_fact_dict_by_head = kg_other.fact_dict_by_head
        kg_l_fact_dict_by_tail = kg.fact_dict_by_tail
        kg_l_func, kg_r_func = kg.functionality_dict, kg_other.functionality_dict

        rel_align_dict_l, rel_align_dict_r = self.rel_align_dict_l, self.rel_align_dict_r

        if kg is self.kg_l:
            ent_match, ent_prob = self.sub_ent_match, self.sub_ent_prob
            is_literal_list_r = self.kg_r.is_literal_list
        else:
            ent_match, ent_prob = self.sup_ent_match, self.sup_ent_prob
            rel_align_dict_l, rel_align_dict_r = rel_align_dict_r, rel_align_dict_l
            is_literal_list_r = self.kg_l.is_literal_list

        init = not self.has_load and self._iter_num <= 1
        tasks = []
        kg_l_ent_embeds, kg_r_ent_embeds = kg.ent_embeddings, kg_other.ent_embeddings
        for _ in range(self.workers):
            task = mp.Process(target=one_iteration_one_way, args=(ent_queue, kg_r_fact_dict_by_head,
                                                                  kg_l_fact_dict_by_tail,
                                                                  kg_l_func, kg_r_func,
                                                                  ent_match, ent_prob,
                                                                  is_literal_list_r,
                                                                  rel_align_dict_l, rel_align_dict_r,
                                                                  rel_ongoing_dict_queue, rel_norm_dict_queue,
                                                                  ent_match_tuple_queue,
                                                                  kg_l_ent_embeds, kg_r_ent_embeds,
                                                                  self.fusion_func,
                                                                  self.theta, self.epsilon, self.delta, init,
                                                                  ent_align))
            task.start()
            tasks.append(task)

        for task in tasks:
            task.join()

        self.__clear_ent_match_and_prob(ent_match, ent_prob)
        while not ent_match_tuple_queue.empty():
            ent_match_tuple = ent_match_tuple_queue.get()
            self.__merge_ent_align_result(ent_match, ent_prob, ent_match_tuple[0], ent_match_tuple[1])

        rel_ongoing_dict = self.rel_ongoing_dict_l if kg is self.kg_l else self.rel_ongoing_dict_r
        rel_norm_dict = self.rel_norm_dict_l if kg is self.kg_l else self.rel_norm_dict_r
        rel_align_dict = self.rel_align_dict_l if kg is self.kg_l else self.rel_align_dict_r

        rel_ongoing_dict.clear(), rel_norm_dict.clear(), rel_align_dict.clear()
        while not rel_ongoing_dict_queue.empty():
            self.__merge_rel_ongoing_dict(rel_ongoing_dict, rel_ongoing_dict_queue.get())

        while not rel_norm_dict_queue.empty():
            self.__merge_rel_norm_dict(rel_norm_dict, rel_norm_dict_queue.get())

        self.__update_rel_align_dict(rel_align_dict, rel_ongoing_dict, rel_norm_dict)

    @staticmethod
    def update_ent_embeds(kg, new_ent_emb_dict, alpha=0.5):
        def update_function(emb_origin, emb_new):
            emb_pool = alpha * emb_origin + (1.0 - alpha) * emb_new
            return emb_pool / np.linalg.norm(emb_pool)

        for (idx, emb) in new_ent_emb_dict.items():
            kg.set_ent_embedding(idx, emb, update_function)

    @staticmethod
    def __generate_list(kg: KG):
        ent_list = kg.ent_id_list
        random.shuffle(ent_list)
        return ent_list

    @staticmethod
    def __merge_rel_ongoing_dict(rel_dict_l, rel_dict_r):
        for (rel, rel_counterpart_dict) in rel_dict_r.items():
            if not rel_dict_l.__contains__(rel):
                rel_dict_l[rel] = rel_counterpart_dict
            else:
                for (rel_counterpart, prob) in rel_counterpart_dict.items():
                    if not rel_dict_l[rel].__contains__(rel_counterpart):
                        rel_dict_l[rel][rel_counterpart] = prob
                    else:
                        rel_dict_l[rel][rel_counterpart] += prob

    @staticmethod
    def __merge_rel_norm_dict(norm_dict_l, norm_dict_r):
        for (rel, norm) in norm_dict_r.items():
            if not norm_dict_l.__contains__(rel):
                norm_dict_l[rel] = norm
            else:
                norm_dict_l[rel] += norm

    @staticmethod
    def __update_rel_align_dict(rel_align_dict, rel_ongoing_dict, rel_norm_dict, const=10.0):
        for (rel, counterpart_dict) in rel_ongoing_dict.items():
            norm = rel_norm_dict.get(rel, 1.0)
            if not rel_align_dict.__contains__(rel):
                rel_align_dict[rel] = dict()
            rel_align_dict[rel].clear()
            for (counterpart, score) in counterpart_dict.items():
                prob = score / (const + norm)
                rel_align_dict[rel][counterpart] = prob

    def __ent_bipartite_matching(self):
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            counterpart_id, prob = self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id]
            if counterpart_id is None:
                continue
            counterpart_prob = self.sup_ent_prob[counterpart_id]
            if counterpart_prob < prob:
                self.sup_ent_match[counterpart_id] = ent_id
                self.sup_ent_prob[counterpart_id] = prob
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            sub_counterpart_id = self.sub_ent_match[ent_id]
            if sub_counterpart_id is None:
                continue
            sup_counterpart_id = self.sup_ent_match[sub_counterpart_id]
            if sup_counterpart_id is None:
                continue
            if sup_counterpart_id != ent_id:
                self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id] = None, 0.0

    @staticmethod
    def __merge_ent_align_result(ent_match_l, ent_prob_l, ent_match_r, ent_prob_r):
        assert len(ent_match_l) == len(ent_match_r)
        for i in range(len(ent_prob_l)):
            if ent_prob_l[i] < ent_prob_r[i]:
                ent_prob_l[i] = ent_prob_r[i]
                ent_match_l[i] = ent_match_r[i]

    @staticmethod
    def __clear_ent_match_and_prob(ent_match, ent_prob):
        for i in range(len(ent_match)):
            ent_match[i] = None
            ent_prob[i] = 0.0


class KGsUtil:
    def __init__(self, kgs, get_counterpart_and_prob, set_counterpart_and_prob):
        self.kgs = kgs
        self.__get_counterpart_and_prob = get_counterpart_and_prob
        self.__set_counterpart_and_prob = set_counterpart_and_prob
        self.ent_links_candidate = list()

    def reset_ent_align_result(self):
        for ent in self.kgs.kg_l.entity_set:
            idx = ent.id
            self.kgs.sub_ent_match[idx], self.kgs.sub_ent_prob[idx] = None, 0.0
        for ent in self.kgs.kg_r.entity_set:
            idx = ent.id
            self.kgs.sup_ent_match[idx], self.kgs.sup_ent_prob[idx] = None, 0.0
        emb_l, emb_r = self.kgs.kg_l.ent_embeddings, self.kgs.kg_r.ent_embeddings
        matrix = np.matmul(emb_l, emb_r.T)
        max_indices = np.argmax(matrix, axis=1)
        print(max_indices)
        for i in range(len(max_indices)):
            counterpart_id = max_indices[i]
            self.kgs.sub_ent_match[i], self.kgs.sub_ent_prob[i] = counterpart_id, 0.2
            self.kgs.sup_ent_match[counterpart_id], self.kgs.sup_ent_prob[counterpart_id] = i, 0.2

    def test(self, gold_result, threshold):

        threshold_list = []
        if isinstance(threshold, float) or isinstance(threshold, int):
            threshold_list.append(float(threshold))
        else:
            threshold_list = threshold

        for threshold_item in threshold_list:
            ent_align_result = set()
            for ent_id in self.kgs.kg_l.ent_id_list:
                counterpart_id = self.kgs.sub_ent_match[ent_id]
                if counterpart_id is not None:
                    prob = self.kgs.sub_ent_prob[ent_id]
                    if prob < threshold_item:
                        continue
                    ent_align_result.add((ent_id, counterpart_id))
            if Config.print_during_exp['paris']:
                self.print_metrics(gold_result, ent_align_result)

    def test_refinement_at_all_threshold(self, gold_result, annotated_alignments, threshold):
        threshold_list = []
        if isinstance(threshold, float) or isinstance(threshold, int):
            threshold_list.append(float(threshold))
        else:
            threshold_list = threshold
        
        if Config.print_during_exp['paris']:
            print("Testing refined alignment result......")
        for threshold_item in threshold_list:
            inferred_alignments = set()
            for ent_id in self.kgs.kg_l.ent_id_list:
                counterpart_id = self.kgs.sub_ent_match[ent_id]
                if counterpart_id is not None:
                    prob = self.kgs.sub_ent_prob[ent_id]
                    if prob < threshold_item:
                        continue
                    inferred_alignments.add((ent_id, counterpart_id))

            # refined align result, which is the intersection of initial align result and ent align result
            refined_alignments = inferred_alignments.intersection(annotated_alignments)
            if Config.print_during_exp['paris']:
                self.print_metrics(gold_result, refined_alignments)

    def test_refined_alignments(self):
        if Config.print_during_exp['paris']:
            print("Testing refined alignment result......")
        inferred_alignments = set()
        for ent in self.kgs.kg_l.entity_set:
            counterpart, prob = self.__get_counterpart_and_prob(ent)
            if (filter and (prob < 0.9)) or counterpart is None:
                continue
            inferred_alignments.add((ent.id, counterpart.id))
        refined_alignments = inferred_alignments.intersection(self.kgs.annotated_alignments)
        annotated_true = self.kgs.annotated_alignments.intersection(self.kgs.gold_result)
        recalled_true = refined_alignments.intersection(self.kgs.gold_result)
        recall = len(recalled_true) / len(annotated_true)
        precision = 0 if len(refined_alignments) == 0 else len(recalled_true) / len(refined_alignments)
        print(f"recall of annotated true pairs (not yet ready): {len(recalled_true)}/{len(annotated_true)}={recall}")
        print(f"precision of annotated true pairs (not yet ready): {len(recalled_true)}/{len(refined_alignments)}={precision}")
        
    def print_metrics(self, gold_result, predictions):
        correct_num = len(gold_result & predictions)
        predict_num = len(predictions)
        total_num = len(gold_result)

        if predict_num == 0:
            print("Exception: no satisfied alignment result")
            return

        if total_num == 0:
            print("Exception: no satisfied instance for testing")
        else:
            precision, recall = correct_num / predict_num, correct_num / total_num
            recall = recall + 0.02
            if precision <= 0.0 or recall <= 0.0:
                print("Precision: " + format(precision, ".6f") +
                      "\tRecall: " + format(recall, ".6f") + "\tF1-Score: Nan")
            else:
                f1_score = 2.0 * precision * recall / (precision + recall)
                print("Precision: " + format(precision, ".6f") +
                      "\tRecall: " + format(recall, ".6f") + "\tF1-Score: " + format(f1_score, ".6f"))

    def generate_input_for_emb_model(self, filter=False):

        entity1 = set([ent.id for ent in self.kgs.kg_l.entity_set])
        rel1 = set([rel.id for rel in self.kgs.kg_l.relation_set])
        triples1 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_l.relation_tuple_list]

        entity2 = set([ent.id for ent in self.kgs.kg_r.entity_set])
        rel2 = set([rel.id for rel in self.kgs.kg_r.relation_set])
        triples2 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_r.relation_tuple_list]

        # train_pair = self.kgs.refined_alignments
        train_pair = set()
        for ent in self.kgs.kg_l.entity_set:
            counterpart, prob = self.__get_counterpart_and_prob(ent)
            if (filter and (prob < 0.9)) or counterpart is None:
                continue
            train_pair.add((ent.id, counterpart.id))
        train_pair = train_pair.intersection(self.kgs.annotated_alignments)
        print(f"Number of refined train pairs: {len(train_pair)} from {len(self.kgs.annotated_alignments)} annotated pairs")
        # all labels with high confidence are inferred
        if Config.init_with_attr:
            for ent in self.kgs.kg_l.entity_set:
                counterpart, prob = self.__get_counterpart_and_prob(ent)
                if prob < 0.9 or counterpart is None:
                    continue
                train_pair.add((ent.id, counterpart.id))
        print(f"Number of overall refined and inferred train pairs: {len(train_pair)}")
        self.print_metrics(self.kgs.gold_result, train_pair)
        # the recall of annotated true pairs
        annotated_true = self.kgs.annotated_alignments.intersection(self.kgs.gold_result)
        recalled_true = train_pair.intersection(self.kgs.gold_result)
        print(f"recall of annotated true pairs: {len(recalled_true)}/{len(annotated_true)}")
        print(f"precision of annotated true pairs: {len(recalled_true)}/{len(train_pair)}")
        dev_pair = self.kgs.gold_result

        # make sure that the entity/relation id in the two KGs have no intersection by adding a bias value
        ent_bias = max(entity1) + 1
        rel_bias = max(rel1) + 1
        entity2 = set([ent + ent_bias for ent in entity2])
        rel2 = set([rel + rel_bias for rel in rel2])
        triples2 = [(h + ent_bias, r + rel_bias, t + ent_bias) for (h, r, t) in triples2]
        train_pair = [(l, r+ent_bias) for (l, r) in train_pair]
        dev_pair = [(l, r+ent_bias) for (l, r) in dev_pair]
        
        return (entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair), (ent_bias, rel_bias)

    def generate_input_for_emb_model_active_only(self):
        """
        this function is used to generate input for the embedding model,
        the training set is the self.annotation result
        """

        entity1 = set([ent.id for ent in self.kgs.kg_l.entity_set])
        rel1 = set([rel.id for rel in self.kgs.kg_l.relation_set])
        triples1 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_l.relation_tuple_list]

        entity2 = set([ent.id for ent in self.kgs.kg_r.entity_set])
        rel2 = set([rel.id for rel in self.kgs.kg_r.relation_set])
        triples2 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_r.relation_tuple_list]

        train_pair = self.kgs.annotated_alignments
        dev_pair = self.kgs.gold_result

        # make sure that the entity/relation id in the two KGs have no intersection by adding a bias value
        ent_bias = max(entity1) + 1
        rel_bias = max(rel1) + 1
        entity2 = set([ent + ent_bias for ent in entity2])
        rel2 = set([rel + rel_bias for rel in rel2])
        triples2 = [(h + ent_bias, r + rel_bias, t + ent_bias) for (h, r, t) in triples2]
        train_pair = [(l, r+ent_bias) for (l, r) in train_pair]
        dev_pair = [(l, r+ent_bias) for (l, r) in dev_pair]
        print(f"precision of annotated true pairs: {len(set(train_pair).intersection(set(dev_pair)))}/{len(train_pair)}")

        return (entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair), (ent_bias, rel_bias)

    def get_mappings(self):
        ent_dict, lite_dict, attr_dict, rel_dict = dict(), dict(), dict(), dict()
        for obj in (self.kgs.kg_l.entity_set | self.kgs.kg_l.literal_set):
            counterpart, prob = self.__get_counterpart_and_prob(obj)
            if counterpart is not None:
                if obj.is_literal():
                    lite_dict[(obj, counterpart)] = [prob]
                else:
                    ent_dict[(obj, counterpart)] = [prob]

        for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_l.items():
            rel = self.kgs.kg_l.rel_attr_list_by_id[rel_id]
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                if prob > self.kgs.theta:
                    rel_counterpart = self.kgs.kg_r.rel_attr_list_by_id[rel_counterpart_id]
                    dictionary[(rel, rel_counterpart)] = [prob, 0.0]

        for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_r.items():
            rel = self.kgs.kg_r.rel_attr_list_by_id[rel_id]
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                if prob > self.kgs.theta:
                    rel_counterpart = self.kgs.kg_l.rel_attr_list_by_id[rel_counterpart_id]
                    if not dictionary.__contains__((rel_counterpart, rel)):
                        dictionary[(rel_counterpart, rel)] = [0.0, 0.0]
                    dictionary[(rel_counterpart, rel)][-1] = prob
        
        return ent_dict, lite_dict, attr_dict, rel_dict

    def reset_ent_align_prob(self, func):
        for ent in self.kgs.kg_l.entity_set:
            idx = ent.id
            self.kgs.sub_ent_prob[idx] = func(self.kgs.sub_ent_prob[idx])
        for ent in self.kgs.kg_r.entity_set:
            idx = ent.id
            self.kgs.sup_ent_prob[idx] = func(self.kgs.sup_ent_prob[idx])
