import numpy as np
from tensorflow.keras import Model
import tensorflow as tf

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel

from aprec.recommenders.dnn_sequential_recommender.transformers.models.bert.configuration_bert import BertConfig
from aprec.recommenders.dnn_sequential_recommender.transformers.models.bert.modeling_tf_bert import TFBertForMaskedLM

class BERT4Rec(SequentialRecsysModel):
    def __init__(self, output_layer_activation = 'linear',
                 embedding_size = 64, max_history_len = 100,
                 attention_probs_dropout_prob = 0.2,
                 hidden_act = "gelu",
                 hidden_dropout_prob = 0.2,
                 initializer_range = 0.02,
                 intermediate_size = 128,
                 num_attention_heads = 2,
                 num_hidden_layers = 3,
                 type_vocab_size = 2, 
                ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.embedding_size = embedding_size
        self.max_history_length = max_history_len
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads 
        self.num_hidden_layers = num_hidden_layers 
        self.type_vocab_size = type_vocab_size      


    def get_model(self):
        bert_config = BertConfig(
            vocab_size = self.num_items + 2, # +1 for mask item, +1 for padding
            hidden_size = self.embedding_size,
            max_position_embeddings=2*self.max_history_length, 
            attention_probs_dropout_prob=self.attention_probs_dropout_prob, 
            hidden_act=self.hidden_act, 
            hidden_dropout_prob=self.hidden_dropout_prob, 
            initializer_range=self.initializer_range, 
            num_attention_heads=self.num_attention_heads, 
            num_hidden_layers=self.num_hidden_layers, 
            type_vocab_size=self.type_vocab_size, 
        )
        return BERT4RecModel(self.batch_size, self.output_layer_activation, bert_config, self.max_history_length)


class BERT4RecModel(Model):
    def __init__(self, batch_size, outputput_layer_activation, bert_config, sequence_length, 
                        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.output_layer_activation = outputput_layer_activation
        self.token_type_ids = tf.constant(tf.zeros(shape=(batch_size, bert_config.max_position_embeddings)))
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))
        self.bert =  TFBertForMaskedLM(bert_config)
        self.bert_config = bert_config

    def skip_values(self, skips):
        skips = np.array(skips)
        skips[skips == 1] = -1
        skips[skips == 0] = 1
        
        return list(skips)

    def create_input_embedding(self, sequences, skips):

        batch_size, seq_len = sequences.shape
        
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Embedding(self.bert_config.vocab_size, self.bert_config.hidden_size, input_length=seq_len))

        #model.compile('rmsprop', 'mse')

        seq_emb = model(sequences)
        skip_emb = model(skips)

        #seq_emb = model.predict(sequences)
        #skip_emb = model.predict(skips)

        return seq_emb + skip_emb

    def mask_skips_in_labels(self, labels, skips):

        skips = np.array(skips)
        labels = np.array(labels)

        labels[skips == 1] = -100
        
        return list(labels)


    # BERT4Rec Call Function
    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        labels = inputs[1]
        positions = inputs[2]
        skips = inputs[3]

        #labels = self.mask_skips_in_labels(labels, skips) # mask any negative samples i.e. skipped songs.

        # For below
        # 1 => skipped songs given value 1
        #      also played and pad items given same value
        # 0 => played songs given value 0 
        #      also played and pad items given same value
        one_hot_labels = (np.array(skips) == 0).astype(int)

        result = self.bert(input_ids=sequences, labels=one_hot_labels, position_ids=positions)          #, inputs_embeds=inputs_embeds) #,input_ids=sequences)

        return result.loss

    def score_all_items(self, inputs):
        sequence = inputs[0] 
        result = self.bert(sequence, position_ids=self.position_ids_for_pred).logits[:,-1,:-2]
        return result