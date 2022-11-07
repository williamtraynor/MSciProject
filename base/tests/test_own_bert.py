import unittest
class TestOwnBERT4rec(unittest.TestCase):
    def test_bert_nlp__model(self):
        from aprec.recommenders.dnn_sequential_recommender.transformers import TFBertModel, BertConfig
        from aprec.recommenders.dnn_sequential_recommender.transformers import BertTokenizer

        print('test_bert_nlp__model')

        config = BertConfig()
        model = TFBertModel(config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer("With a record number of patients on hospital waiting lists in England", return_tensors = "tf")
        output = model(tokens)
        self.assertEqual(output[0].last_hidden_state.shape, (1, 14, 768))
        pass

    def test_bert4rec_model(self):
        from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec

        bert4rec = BERT4Rec()
        bert4rec.set_common_params(10, 10, None, None, 32, None)
        model = bert4rec.get_model()


    def test_bert4rec_recommender(self):
        print('test_bert4rec_recommender')
        import tempfile
        from aprec.utils.generator_limit import generator_limit
        from aprec.evaluation.split_actions import TemporalGlobal
        from aprec.evaluation.n_actions_for_user import n_actions_for_user
        from aprec.evaluation.evaluate_recommender import evaluate_recommender
        from aprec.evaluation.metrics.precision import Precision
        from aprec.evaluation.metrics.recall import Recall
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
        from aprec.tests.test_dnn_sequential import USER_ID
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.mini_le import get_lfm_actions, get_tracks_catalog
        #from aprec.datasets.lfm2b import get_lfm_actions, get_tracks_catalog
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
        from aprec.recommenders.metrics.ndcg import KerasNDCG
        from aprec.losses.mean_ypred_ploss import MeanPredLoss
        #from aprec.recommenders.dnn_sequential_recommender.supcon.losses import ContrastiveLoss
        from aprec.losses.bpr import BPRLoss
        from aprec.losses.multi_similarity_loss import MultiSimilarityLoss
        from aprec.pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
        from aprec.evaluation.metrics.ndcg import NDCG
        from aprec.evaluation.metrics.mrr import MRR


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
        test = n_actions_for_user(test, 20)
        for action in train:
            recommender.add_action(action)
        recommender.rebuild_model()
        metrics = [MRR(), NDCG(5), NDCG(10), NDCG(20)]
        output_dir = tempfile.mkdtemp()
        result = evaluate_recommender(recommender, test, metrics, output_dir, "top_recommender")
        
        print(result)
        
        '''print('Rebuilding Model')
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print('Getting catalogue')
        catalog = get_tracks_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])'''


if __name__ == "__main__":
    unittest.main()