from base.recommenders.top_recommender import TopRecommender
from base.recommenders.lightfm import LightFMRecommender
from base.recommenders.filter_seen_recommender import FilterSeenRecommender
from base.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from base.evaluation.metrics.mrr import MRR
from base.evaluation.metrics.ndcg import NDCG
from base.evaluation.split_actions import LeaveOneOut
import numpy as np


DATASET = "BERT4rec.ml-1m"

USERS_FRACTIONS = [1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "MF-BPR": lambda: lightfm_recommender(30, 'bpr'),
}

MAX_TEST_USERS=6040

METRICS = [NDCG(10), MRR()]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

