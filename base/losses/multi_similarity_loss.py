from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from ..losses.generic_pair_loss import GenericPairLoss


class MultiSimilarityLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.add_to_recordable_attributes(
            list_of_names=["alpha", "beta", "base"], is_stat=False
        )

    # To fit in existing code may need to rename function below to __call__(...)
    def __call__(self, mat, pos_mask, neg_mask):

    #def __call__(self, y_true, y_pred):
    
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)
        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )

        
        
        return {
            "loss": {
                "losses": pos_loss + neg_loss + manual_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }
        

    def get_default_distance(self):
        return CosineSimilarity()
