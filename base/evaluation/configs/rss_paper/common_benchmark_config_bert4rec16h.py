from base.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from base.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from base.recommenders.dnn_sequential_recommender.models.caser import Caser
from base.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from base.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from base.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from base.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from base.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from base.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
from base.evaluation.samplers.random_sampler import RandomTargetItemSampler
from base.recommenders.metrics.ndcg import KerasNDCG
from base.recommenders.top_recommender import TopRecommender
from base.recommenders.svd import SvdRecommender
from base.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from base.recommenders.lightfm import LightFMRecommender
from base.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from base.losses.bce import BCELoss
from base.losses.lambda_gamma_rank import LambdaGammaRankLoss



from base.evaluation.metrics.ndcg import NDCG
from base.evaluation.metrics.mrr import MRR
from base.evaluation.metrics.map import MAP
from base.evaluation.metrics.hit import HIT

from tensorflow.keras.optimizers import Adam

from base.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def top_recommender():
    return TopRecommender()


def svd_recommender(k):
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    return LightFMRecommender(k, loss, num_threads=32)


def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter=SequenceContinuation, 
                 target_builder=FullMatrixTargetsBuilder,
                optimizer=Adam(),
                training_time_limit=3600, metric=KerasNDCG(40), 
                max_epochs=10000
                ):
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          debug=False
                                                          )

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

HISTORY_LEN=50

recommenders = {
    "bert4rec-16h": lambda: vanilla_bert4rec(3600 * 16) 
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

def get_recommenders(filter_seen: bool, filter_recommenders = set()):
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
                continue
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result
