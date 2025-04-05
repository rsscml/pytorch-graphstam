import torch
import torch.nn.functional as F


class QuantileLoss:
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(self, quantiles=None):
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles = quantiles

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses


class RMSE:
    """
    Root-mean-square error
    Defined as ``(y_pred - target)**2``
    """
    def __init__(self, tensor_dims=3):
        super().__init__()
        self.tensor_dims = tensor_dims

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.tensor_dims == 3:
            target = torch.unsqueeze(target, dim=2)
        loss = torch.pow(y_pred - target, 2)

        return loss


class TweedieLoss:
    def __init__(self, p_list=None):
        super().__init__()
        if p_list is None:
            p_list = []
        self.p_list = p_list

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, p, scaler, log1p_transform):
        # convert all 2-d inputs to 3-d
        y_true = torch.unsqueeze(y_true, dim=2)
        scaler = torch.unsqueeze(scaler, dim=2)

        if len(self.p_list) > 0:
            pass
        else:
            self.p_list.append(p)

        """
        if log1p_transform:
            # log1p first, scale next
            # reverse process actual
            y_true = torch.expm1(y_true * scaler)
            # reverse proceess y_pred
            y_pred = torch.expm1(torch.exp(y_pred) * scaler)
            # take log of y_pred again
            y_pred = torch.log(y_pred + 1e-8)

            a = y_true * torch.exp(y_pred * (1 - p)) / (1 - p)
            b = torch.exp(y_pred * (2 - p)) / (2 - p)
            loss = -a + b

        """
        if log1p_transform:
            # scale first, log1p after
            y_true = torch.expm1(y_true) * scaler
            # reverse log of prediction y_pred
            y_pred = torch.exp(y_pred)
            # clamp predictions
            y_pred = torch.clamp(y_pred, min=-7, max=7)
            # get pred
            y_pred = torch.expm1(y_pred) * scaler
            # take log of y_pred again
            y_pred = torch.log(y_pred + 1e-8)

            loss = 0
            for pn in self.p_list:
                pn = torch.unsqueeze(pn, dim=2)
                a = y_true * torch.exp(y_pred * (1 - pn)) / (1 - pn)
                b = torch.exp(y_pred * (2 - pn)) / (2 - pn)
                loss += (-a + b)
        else:
            # no log1p
            # clamp predictions
            y_pred = torch.clamp(y_pred, min=-7, max=7)
            y_true = y_true * scaler
            loss = 0
            for pn in self.p_list:
                pn = torch.unsqueeze(pn, dim=2)
                a = y_true * torch.exp((y_pred + torch.log(scaler)) * (1 - pn)) / (1 - pn)
                b = torch.exp((y_pred + torch.log(scaler)) * (2 - pn)) / (2 - pn)
                loss += (-a + b)

            """
            a = y_true * torch.exp(y_pred * (1 - p)) / (1 - p)
            b = torch.exp(y_pred * (2 - p)) / (2 - p)
            loss = -a + b
            """
        return loss


class Huber:
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        self.loss_obj = torch.nn.HuberLoss(reduction='none', delta=self.delta)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_true = torch.unsqueeze(y_true, dim=2)
        loss = self.loss_obj(input=y_pred, target=y_true)
        return loss


class NormalizedMultiLoss:
    def __init__(self, alpha=0.5, beta=0.9, ema_iterations=10):
        self.alpha = alpha  # Weight for point loss
        self.beta = beta  # EMA decay factor
        self.max_ema_iterations = ema_iterations
        self.init_iterations = 0
        self.point_loss_ema = None
        self.total_loss_ema = None

    def loss(self, point_loss, total_loss):

        if self.init_iterations <= self.max_ema_iterations:
            self.init_iterations += 1
            # Update exponential moving averages
            if self.point_loss_ema is None:
                self.point_loss_ema = point_loss.detach().mean().item()
                self.total_loss_ema = total_loss.detach().mean().item()
            else:
                self.point_loss_ema = self.beta * self.point_loss_ema + (1 - self.beta) * point_loss.detach().mean().item()
                self.total_loss_ema = self.beta * self.total_loss_ema + (1 - self.beta) * total_loss.detach().mean().item()


        # Normalize losses by their EMAs
        normalized_point_loss = point_loss / (self.point_loss_ema + 1e-8)
        normalized_total_loss = total_loss / (self.total_loss_ema + 1e-8)

        # Weighted combination of normalized losses
        return self.alpha * normalized_point_loss + (1 - self.alpha) * normalized_total_loss
