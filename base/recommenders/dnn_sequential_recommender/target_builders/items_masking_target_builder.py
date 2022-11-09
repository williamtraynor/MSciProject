from random import Random

import numpy as np
from base.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder

class ItemsMaskingTargetsBuilder(TargetBuilder):
    def __init__(self, random_seed=31337, 
                       relative_positions_encoding = True, 
                       ignore_value=-100, #-100 is used by default in hugginface's BERT implementation
                       include_skips=False):
        self.random = Random()
        self.random.seed(random_seed) 
        self.targets = []
        self.ignore_value = ignore_value
        self.relative_positions_encoding = relative_positions_encoding
        self.positions = []
        self.include_skips = include_skips

    def build(self, user_targets):
        print("BUILDING TARGETS")
        #print(user_targets)
        targets = []
        positions = []
        p = True
        for seq_len, user in user_targets:
            user_positions = []
            user_target = [self.ignore_value] * self.sequence_len
            if self.relative_positions_encoding:
                split_pos = self.random.randint(self.sequence_len - seq_len, self.sequence_len - 1)
            else:
                split_pos = self.sequence_len - 1

            for i in range(self.sequence_len):
                user_positions.append(self.sequence_len - split_pos  + i) 

            positions.append(user_positions)

            # SECTION BELOW ADDED TO INCLUDE SKIPS TO EMBEDDINGS
            # SIMPLY MULTIPLY ITEM NUMBER BY 0 IF IT WAS SKIPPED
            if self.include_skips:
                for pos in user:
                    skipped = bool(pos[1][2])
                    #print(skipped)
                    multiplier = 1
                    if skipped:
                        user_target[pos[0]] = -100 #ignore if item was skipped
                    else:
                        user_target[pos[0]] = pos[1][1]  
            else:
                for pos in user:
                    user_target[pos[0]] = pos[1][1]

            if p:
                if len(user) > 1:
                    print('User: ', user)
                    for pos in user:
                        print('POS: ', pos)
                    print('USER TARGET: ', user_target)
                    p=False

            targets.append(user_target)

        self.positions = np.array(positions)
        self.targets = np.array(targets)


    def get_targets(self, start, end):
        return [self.targets[start:end], self.positions[start:end]], self.targets[start:end]
