import torch
from skorch import NeuralNetClassifier
class ShapeletRegularizedNet(NeuralNetClassifier):
  def __init__(self, *args, lambda_prototypes=0.05,lambda_linear_params=0.05, lambda_fused_lasso = 0.03,**kwargs):
    super().__init__(*args, **kwargs)
    self.lambda_prototypes = lambda_prototypes
    self.lambda_linear_params = lambda_linear_params
    self.lambda_fused_lasso = lambda_fused_lasso

  def get_loss(self, y_pred, y_true, X=None, training=False):
    loss_SoftMax = super().get_loss(y_pred, y_true, X=X, training=training)
    
    loss_prototypes = torch.norm(self.module_.prototypes,p=2)
    loss_weight_reg = 0
    for param in self.module_.linear_layer1.parameters():
      loss_weight_reg += param.norm(p=2).sum()

    # Add fused lasso penalty to the loss
    fused_lasso_penalty = 0
    prototypes = self.module_.prototypes
    for row in prototypes:
        row_penalty = torch.sum(
          torch.abs(row[1:] - row[:-1]))  # Compute the absolute differences between successive terms in a row
        fused_lasso_penalty += row_penalty  # total fused lasso penalty for all rows

    loss = (loss_SoftMax + self.lambda_prototypes * loss_prototypes
            + self.lambda_linear_params * loss_weight_reg + self.lambda_fused_lasso * fused_lasso_penalty)
    return loss