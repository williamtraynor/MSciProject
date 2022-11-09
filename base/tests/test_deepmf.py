import unittest

USER_ID = '120'

class TestDeepMF(unittest.TestCase):
    def test_deepmf_recommender(self):
        from base.recommenders.deep_mf import DeepMFRecommender
        from base.recommenders.filter_seen_recommender import FilterSeenRecommender
        from base.datasets.movielens20m import get_movielens20m_actions
        from base.utils.generator_limit import generator_limit

        mlp_recommender = DeepMFRecommender(100, 1000, steps=20)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()
