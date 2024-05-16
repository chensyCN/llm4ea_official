def get_counterpart_id_and_prob(ent_match, ent_prob, ent_id):
    counterpart = ent_match[ent_id]
    if counterpart is None:
        return None, 0.0
    else:
        return counterpart, ent_prob[ent_id]


def set_counterpart_id_and_prob(ent_match, ent_prob, ent_l_id, ent_r_id, prob):
    curr_prob = ent_prob[ent_l_id]
    if prob < curr_prob:
        return
    ent_match[ent_l_id], ent_prob[ent_l_id] = ent_r_id, prob


def register_rel_align_prob_norm(dictionary, rel, prob):
    if not dictionary.__contains__(rel):
        dictionary[rel] = 0.0
    dictionary[rel] += prob


def register_ongoing_prob_product(dictionary, key1, key2, prob):
    if not dictionary.__contains__(key1):
        dictionary[key1] = dict()
    if not dictionary[key1].__contains__(key2):
        dictionary[key1][key2] = 0.0
    dictionary[key1][key2] += prob


def get_rel_align_prob(dictionary, rel_l, rel_r):
    if not dictionary.__contains__(rel_l):
        return 0.0
    if not dictionary[rel_l].__contains__(rel_r):
        return 0.0
    prob = dictionary[rel_l][rel_r]
    prob = 1.0 if prob > 1.0 else prob
    prob = 0.0 if prob < 0.0 else prob
    return prob


def update_ent_align_prob(ent_align_ongoing_dict, ent_match, ent_prob, kg_l_ent_embeds, kg_r_ent_embeds, ent, fusion_func, init):
    counterpart, value = None, 0.0
    for (candidate, prob) in ent_align_ongoing_dict.items():
        val = 1.0 - prob
        if not init and kg_l_ent_embeds is not None and kg_r_ent_embeds is not None and fusion_func is not None:
            ent_emb = kg_l_ent_embeds[ent, :]
            candidate_emb = kg_r_ent_embeds[candidate, :]
            val = fusion_func(val, ent_emb, candidate_emb)
        if val >= value:
            value, counterpart = val, candidate
    value = 1.0 if value > 1.0 else value
    value = 0.0 if value < 0.0 else value
    set_counterpart_id_and_prob(ent_match, ent_prob, ent, counterpart, value)


def register_ent_equality(ent_align_ongoing_dict, rel_align_dict_l, rel_align_dict_r,
                          kg_l_func, kg_r_func,
                          rel, rel_counterpart, tail_counterpart,
                          head_eqv_prob, theta, epsilon, delta, init):
    prob_sub = get_rel_align_prob(rel_align_dict_l, rel, rel_counterpart) / epsilon
    prob_sup = get_rel_align_prob(rel_align_dict_r, rel_counterpart, rel) / epsilon
    if prob_sub < theta and prob_sup < theta:
        if init:
            prob_sub, prob_sup = theta, theta
        else:
            return
    func_l, func_r = kg_l_func.get(rel, 0.0) / epsilon, kg_r_func.get(rel_counterpart, 0.0) / epsilon
    factor = 1.0
    factor_l = 1.0 - head_eqv_prob * prob_sup * func_r
    factor_r = 1.0 - head_eqv_prob * prob_sub * func_l
    if prob_sub >= 0.0 and func_l >= 0.0:
        factor *= factor_l
    if prob_sup >= 0.0 and func_r >= 0.0:
        factor *= factor_r
    if 1.0 - factor > delta:
        if not ent_align_ongoing_dict.__contains__(tail_counterpart):
            ent_align_ongoing_dict[tail_counterpart] = 1.0
        ent_align_ongoing_dict[tail_counterpart] *= factor


def one_iteration_one_way(queue, kg_r_fact_dict_by_head,
                          kg_l_fact_dict_by_tail,
                          kg_l_func, kg_r_func,
                          sub_ent_match, sub_ent_prob,
                          is_literal_list_r,
                          rel_align_dict_l, rel_align_dict_r,
                          rel_ongoing_dict_queue, rel_norm_dict_queue,
                          ent_match_tuple_queue,
                          kg_l_ent_embeds, kg_r_ent_embeds,
                          fusion_func,
                          theta, epsilon, delta, init=False, ent_align=True):
    rel_ongoing_dict, rel_norm_dict = dict(), dict()
    while not queue.empty():
        # noinspection PyBroadException
        try:
            ent_id = queue.get_nowait()
        except Exception:
            break
        ent_align_ongoing_dict = dict()
        ent_fact_list = kg_l_fact_dict_by_tail.get(ent_id, list())
        for (rel_id, head_id) in ent_fact_list:
            head_counterpart, head_eqv_prob = get_counterpart_id_and_prob(sub_ent_match, sub_ent_prob, head_id)
            if head_counterpart is None or head_eqv_prob < theta:
                continue
            ent_counterpart, tail_eqv_prob = get_counterpart_id_and_prob(sub_ent_match, sub_ent_prob, ent_id)
            if ent_counterpart is not None:
                register_rel_align_prob_norm(rel_norm_dict, rel_id, head_eqv_prob * tail_eqv_prob)
            head_counterpart_fact_list = kg_r_fact_dict_by_head.get(head_counterpart, list())
            for (rel_counterpart_id, tail_counterpart_id) in head_counterpart_fact_list:
                if is_literal_list_r[tail_counterpart_id]:
                    continue
                eqv_prob = tail_eqv_prob if tail_counterpart_id == ent_counterpart else 0.0
                if eqv_prob > 0.0:
                    register_ongoing_prob_product(rel_ongoing_dict, rel_id, rel_counterpart_id,
                                                  head_eqv_prob * eqv_prob)
                if ent_align:
                    register_ent_equality(ent_align_ongoing_dict, rel_align_dict_l, rel_align_dict_r,
                                          kg_l_func, kg_r_func,
                                          rel_id, rel_counterpart_id, tail_counterpart_id,
                                          head_eqv_prob, theta, epsilon, delta, init)
        if ent_align:
            update_ent_align_prob(ent_align_ongoing_dict, sub_ent_match, sub_ent_prob, kg_l_ent_embeds, kg_r_ent_embeds, ent_id, fusion_func, init)
    rel_ongoing_dict_queue.put(rel_ongoing_dict), rel_norm_dict_queue.put(rel_norm_dict)
    ent_match_tuple_queue.put((sub_ent_match, sub_ent_prob))
    exit(1)
