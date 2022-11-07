import numpy as np
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.history_vectorizer import HistoryVectorizer


class DefaultHistoryVectrizer(HistoryVectorizer):
    def __call__(self, user_actions):

        if len(user_actions) >= self.sequence_len:
            '''l = []
            for action in user_actions[-self.sequence_len:]:
                if len(action) > 2:
                    l.append(action[1] * multiplier(action[2]))
                else:
                    l.append(action[1] * 1)
            return np.array(l)'''
            return np.array([action[1] for action in user_actions[-self.sequence_len:]])
        else:
            n_special = self.sequence_len - len(user_actions)
            result_list = [self.padding_value] * n_special + [action[1] for action in user_actions]
            return np.array(result_list)


