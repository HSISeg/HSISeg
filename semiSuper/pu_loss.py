import chainer.functions as F
import numpy
from chainer import cuda, function, Variable
from chainer.utils import type_check
 

class PULoss(function.Function):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss_func_name='sigmoid', unlabeled = -1, loss=(lambda x: F.sigmoid(-x)), gamma=1, beta=0, nnPU=True):
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.loss_func_name = loss_func_name
        self.nnPU = nnPU
        self.positive = 1
        # self.unlabeled = -1
        self.unlabeled = unlabeled # For binary

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        t = t[:, None]
        positive, unlabeled = t == self.positive, t == self.unlabeled
        n_positive, n_unlabeled = max([1., xp.sum(positive)]), max([1., xp.sum(unlabeled)])
        self.x_in = Variable(x)
        t_positive = numpy.ones(x.shape, dtype=numpy.int32) # For binary
        t_negative = numpy.zeros(x.shape, dtype=numpy.int32) # For binary
        if self.loss_func_name == "sigmoid_cross_entropy":
            y_positive = self.loss_func(self.x_in, t_positive, reduce='no', normalize=False) # For binary
            y_unlabeled = self.loss_func(self.x_in, t_negative, reduce='no', normalize=False) # For binary
        elif self.loss_func_name == "tanh_cross_entropy":
            y_positive = self.loss_func(self.x_in, t_positive, reduce='no', normalize=False) # For binary
            y_unlabeled = self.loss_func(self.x_in, t_negative, reduce='no', normalize=False) # For binary
        else:
            y_positive = self.loss_func(self.x_in)
            y_unlabeled = self.loss_func(-self.x_in)
        positive_risk = F.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = F.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        objective = positive_risk + negative_risk
        if self.nnPU:
            if negative_risk.data < -self.beta:
                objective = positive_risk - self.beta
                self.x_out = -self.gamma * negative_risk
            else:
                self.x_out = objective
        else:
            self.x_out = objective
        self.loss = xp.array(objective.data, dtype=self.x_out.data.dtype)
        return self.loss,

    def backward(self, inputs, gy):
        self.x_out.backward()
        gx = gy[0].reshape(gy[0].shape + (1,) * (self.x_in.data.ndim - 1)) * self.x_in.grad
        return gx, None


def pu_loss(x, t, prior, loss=(lambda x: F.sigmoid(-x)), nnPU=True):
    """wrapper of loss function for non-negative/unbiased PU learning

        .. math::
            \\begin{array}{lc}
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) & {\\rm if nnPU learning}\\\\
            L_[\\pi E_1[l(f(x))]+E_X[l(-f(x))]-\\pi E_1[l(-f(x))] & {\\rm otherwise}
            \\end{array}

    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~chainer.function): loss function.
            The loss function should be non-increasing.
        nnPU (bool): Whether use non-negative PU learning or unbiased PU learning.
            In default setting, non-negative PU learning will be used.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            PU loss.

    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 32nd International Conference on Machine Learning. 2015.
    """
    return PULoss(prior=prior, loss=loss, nnPU=nnPU)(x, t)
