import unittest
class TestOwnBERT4rec(unittest.TestCase):
    def test_bert_nlp__model(self):
        from base.recommenders.dnn_sequential_recommender.transformers import TFBertModel, BertConfig
        from base.recommenders.dnn_sequential_recommender.transformers import BertTokenizer

        print('test_bert_nlp__model')

        config = BertConfig()
        model = TFBertModel(config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer("With a record number of patients on hospital waiting lists in England", return_tensors = "tf")
        output = model(tokens)
        self.assertEqual(output[0].last_hidden_state.shape, (1, 14, 768))
        pass

    def test_bert4rec_model(self):
        from base.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec

        bert4rec = BERT4Rec()
        bert4rec.set_common_params(10, 10, None, None, 32, None)
        model = bert4rec.get_model()


    def test_bert4rec_recommender(self):
        print('test_bert4rec_recommender')
        import tempfile
        from base.utils.generator_limit import generator_limit
        from base.evaluation.split_actions import TemporalGlobal
        from base.evaluation.n_actions_for_user import n_actions_for_user
        from base.evaluation.evaluate_recommender import evaluate_recommender
        from base.evaluation.metrics.precision import Precision
        from base.evaluation.metrics.recall import Recall
        from base.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from base.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from base.recommenders.filter_seen_recommender import FilterSeenRecommender
        from base.recommenders.vanilla_bert4rec import VanillaBERT4Rec
        from base.tests.test_dnn_sequential import USER_ID
        from base.utils.generator_limit import generator_limit
        from base.datasets.mini_le import get_lfm_actions, get_tracks_catalog
        #from base.datasets.lfm2b import get_lfm_actions, get_tracks_catalog
        from base.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from base.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
        from base.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from base.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
        from base.recommenders.metrics.ndcg import KerasNDCG
        from base.losses.mean_ypred_ploss import MeanPredLoss
        #from base.recommenders.dnn_sequential_recommender.supcon.losses import ContrastiveLoss
        from base.losses.bpr import BPRLoss
        from base.losses.multi_similarity_loss import MultiSimilarityLoss
        from base.pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
        from base.evaluation.metrics.ndcg import NDCG
        from base.evaluation.metrics.mrr import MRR


        val_users = [ '16026', '21110', '75753', '17605', '55338', '24860', '29365']
        model = BERT4Rec(embedding_size=32)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=5*60, # Training time limit is in seconds
                                               loss=MeanPredLoss(),
                                               debug=True, 
                                               sequence_splitter=lambda: ItemsMasking(recency_importance=exponential_importance(0.8)), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=True, include_skips=True),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               )

        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        actions = generator_limit(get_lfm_actions(), 10000)
        split_actions = TemporalGlobal((70, 30))
        train, test = split_actions(actions)
        test = n_actions_for_user(test, 1)
        for action in train:
            recommender.add_action(action)
        recommender.rebuild_model()
        metrics = [MRR(), NDCG(5), NDCG(10), NDCG(20), Precision(1), Recall(1), Precision(5), Recall(5), Precision(10), Recall(10)]
        output_dir = tempfile.mkdtemp()
        result = evaluate_recommender(recommender, test, metrics, output_dir, "top_recommender")
        
        print(result)


if __name__ == "__main__":
    unittest.main()