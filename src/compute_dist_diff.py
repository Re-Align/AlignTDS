import pickle
import math
from tqdm import tqdm 




def norm_prob(token_list):
    max_logit = max(token["logit"] for token in token_list)
    sum_exp_logit = sum(math.exp(token["logit"] - max_logit) for token in token_list)

    for token in token_list:
        logit_diff = token["logit"] - max_logit
        token["norm_prob"] = math.exp(logit_diff) / sum_exp_logit

    return token_list


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity


def compute_similarity(token_list_1, token_list_2, metric="jaccard", top_k=100):
    token_list_1 = norm_prob(token_list_1)
    token_list_2 = norm_prob(token_list_2) 
    list1 = [i["token"] for i in token_list_1[:top_k]]
    list2 = [i["token"] for i in token_list_2[:top_k]]
    set1 = set(list1)
    set2 = set(list2)
    similarity = 0.0
    if metric == "jaccard":
        similarity = jaccard_similarity(set1, set2)
    elif metric == "weighted_sum":
        all_tokens = set1.union(set2)
        for t in all_tokens:
            w1 = token_list_1[list1.index(t)]["norm_prob"] if t in list1 else 0
            w2 = token_list_2[list2.index(t)]["norm_prob"] if t in list2 else 0
            # if w1 > 0 and w2 > 0:
            similarity += w1 * w2
    elif metric == "KL":
        all_tokens = set1.union(set2)
        for t in all_tokens:
            w1 = token_list_1[list1.index(t)]["norm_prob"] if t in list1 else 0
            w2 = token_list_2[list2.index(t)]["norm_prob"] if t in list2 else 0
            if w1 > 0 and w2 > 0:
                similarity += w1 * math.log(w1 / w2)
    elif metric == "nDCG":
        relevance_scores = {}
        for i, token in enumerate(token_list_2):
            relevance_scores[token["token_id"]] = 1 / math.log2(
                i + 2
            )  # Assign relevance scores based on position

        dcg = 0.0
        for i, token in enumerate(token_list_1):
            relevance_score = relevance_scores.get(token["token_id"], 0.0)
            dcg += relevance_score / math.log2(i + 2)  # Compute DCG

        # Compute IDCG by sorting token_list_2 in descending order of relevance scores
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        idcg = sum(ideal_scores[i] / math.log2(i + 2) for i in range(len(token_list_1)))

        if idcg == 0:
            return 0.0  # To avoid division by zero, return 0 if IDCG is 0

        ndcg = dcg / idcg
        similarity = ndcg
    elif metric == "top_rank":
        top_token = list2[0]
        rank = top_k
        if top_token in list1:
            rank = list1.index(top_token)
        similarity = rank
    elif metric == "top_prob":
        top_token = list2[0] 
        prob = 0.0
        if top_token in list1:
            rank = list1.index(top_token)
            prob = token_list_1[rank]["norm_prob"]
        similarity = prob
    elif metric == "bi_top_prob":
        top_token = list2[0] 
        prob1 = 0.0
        if top_token in list1:
            rank = list1.index(top_token)
            prob1 = token_list_1[rank]["norm_prob"]
        # --- 
        top_token = list1[0] 
        prob2 = 0.0
        if top_token in list2:
            rank = list2.index(top_token)
            prob2 = token_list_2[rank]["norm_prob"]
        similarity = (prob1 + prob2)/2 
    return similarity


# print(i2b_data[0]["results"][2][:10])
# print(i2b_data[0]["results"][2][:10])

if __name__ == "__main__":
    
    # path_to_dir = "saved_logits/just_eval_1000/llama2-7b_shards/"
    # i2b_path = f"{path_to_dir}/llama2-i2b.pkl"
    # i2i_path = f"{path_to_dir}/llama2-i2i.pkl"
    import sys 
    
    # i2i_path = sys.argv[1]
    # i2b_path = sys.argv[2]
    pair_name = sys.argv[1] 
    i2i_path = sys.argv[2]
    i2b_path = sys.argv[3]

    with open(i2b_path, "rb") as f:
        i2b_data = pickle.load(f)

    with open(i2i_path, "rb") as f:
        i2i_data = pickle.load(f)
    curves = []
    for eid in tqdm(range(1000)):
        i2b_res = i2b_data[eid]["results"]
        i2i_res = i2i_data[eid]["results"]
        if len(i2b_res) != len(i2i_res):
            print()
        for i in range(len(i2b_res)):
            # print(compute_similarity(i2b_res[i], i2i_res[i], metric="jaccard", top_k=10))
            # i2b_res[i] = norm_prob(i2b_res[i])
            # i2i_res[i] = norm_prob(i2i_res[i]) 
            curves.append(
                {
                    "step": i,
                    "jaccard": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="jaccard", top_k=50
                    ),
                    "jaccard@10": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="jaccard", top_k=10
                    ), 
                    "nDCG": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="nDCG", top_k=50
                    ),
                    "nDCG@10": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="nDCG", top_k=10
                    ),
                    "KL": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="KL", top_k=50
                    ),
                    "KL@10": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="KL", top_k=10
                    ), 
                    "WS": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="weighted_sum", top_k=50
                    ),
                    "WS@10": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="weighted_sum", top_k=10
                    ),
                    "TR": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="top_rank", top_k=50
                    ),
                    "TP": compute_similarity(
                        i2b_res[i]["tokens"], i2i_res[i]["tokens"], metric="top_prob", top_k=50
                    ),
                }
            )
            # print(curves[-1])

    with open(f"saved_logits/{pair_name}_curves.pkl", "wb") as f:
        pickle.dump(curves, f)
