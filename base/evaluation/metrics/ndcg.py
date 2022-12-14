import math
from .metric import Metric

class NDCG(Metric):
    def __init__(self, k):
        self.name = "ndcg@{}".format(k)
        self.k = k
        self.less_is_better = False
        
    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0):
            return 0

        #print(f'Recommendations:\n{type(recommendations)}\n{len(recommendations)}\n{recommendations}\n')
        #print(f'Actual Actions:\n{type(actual_actions)}\n{actual_actions}\n')

        actual_set = set([action.item_id for action in actual_actions])
        recommended = [recommendation[0] for recommendation in recommendations[:self.k]]
        cool = set(recommended).intersection(actual_set)
        if len(cool) == 0:
            return 0
        ideal_rec = sorted(recommended, key = lambda x: not(x in actual_set))
        return NDCG.dcg(recommended, actual_set)/NDCG.dcg(ideal_rec, actual_set)
         

    @staticmethod
    def dcg(id_list, relevant_id_set):
        result = 0.0
        for idx in range(len(id_list)):
            i = idx + 1
            if (id_list[idx]) in relevant_id_set:
                result += 1 / math.log2(i+1)
        return result




