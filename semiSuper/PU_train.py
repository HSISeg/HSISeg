import six
import copy
import chainer
import Config


try:
    from matplotlib import use
    use('Agg')
except ImportError:
    pass

from chainer import Variable, functions as F
from chainer.training import extensions
from semiSuper.PU_model import MultiLayerPerceptron, CNN, BassNet
from semiSuper.pu_loss import PULoss
from semiSuper.tanh_cross_entropy import tanh_cross_entropy


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x), "sigmoid": lambda x: F.sigmoid(-x), "sigmoid_cross_entropy": F.sigmoid_cross_entropy, "tanh_cross_entropy":tanh_cross_entropy}
    return losses[loss_name]


def select_model(model_name):
    models = {"mlp": MultiLayerPerceptron, "cnn": CNN,"bass_net":BassNet}
    return models[model_name]


def make_optimizer(model, stepsize):
    optimizer = chainer.optimizers.Adam(alpha=stepsize)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))
    return optimizer


class MultiUpdater(chainer.training.StandardUpdater):

    def __init__(self, iterator, optimizer, model, converter=chainer.dataset.convert.concat_examples,
                 device=None, loss_func=None):
        assert(isinstance(model, dict))
        self.model = model
        assert(isinstance(optimizer, dict))
        if loss_func is None:
            loss_func = {k: v.target for k, v in optimizer.items()}
        assert(isinstance(loss_func, dict))
        super(MultiUpdater, self).__init__(iterator, optimizer, converter, device, loss_func)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizers = self.get_all_optimizers()
        models = self.model
        loss_funcs = self.loss_func
        if isinstance(in_arrays, tuple):
            x, t = tuple(Variable(x) for x in in_arrays)
            for key in optimizers:
                optimizers[key].update(models[key], x, t, loss_funcs[key])
        else:
            raise NotImplemented


class MultiEvaluator(chainer.training.extensions.Evaluator):
    default_name = 'test'

    def __init__(self, *args, **kwargs):
        super(MultiEvaluator, self).__init__(*args, **kwargs)

    def evaluate(self):
        iterator = self._iterators['main']
        targets = self.get_all_targets()

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()
        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(Variable(x)
                                    for x in in_arrays)
                    for k, target in targets.items():
                        target.error(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: Variable(x)
                               for key, x in six.iteritems(in_arrays)}
                    for k, target in targets.items():
                        target.error(**in_vars)
                else:
                    in_vars = Variable(in_arrays)
                    for k, target in targets.items():
                        target.error(in_vars)
            summary.add(observation)
        return summary.compute_mean()

def train(XYtrain, XYtest, prior, test_output_path, train_output_path, out):
    unlabeled_tag = Config.unlabeled_tag
    gpu = Config.gpu
    batchsize = Config.batchsize
    epoch = Config.epoch
    loss = Config.loss
    model = Config.model
    gamma = Config.gamma
    beta = Config.beta
    stepsize = Config.stepsize
    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    channel = XYtrain[0][0].shape[0]
    train_iter = chainer.iterators.SerialIterator(XYtrain, batchsize)
    test_iter = chainer.iterators.SerialIterator(XYtest, batchsize, repeat=False, shuffle=False)

    # model setup
    loss_type = select_loss(loss)
    selected_model = select_model(model)
    model = selected_model(prior, dim, channel, loss)
    models = {"nnPU": copy.deepcopy(model)}
    loss_funcs = {
        "nnPU": PULoss(prior, loss=loss_type, loss_func_name=loss, unlabeled=unlabeled_tag, nnPU=True,
                       gamma=gamma, beta=beta)}
    if gpu >= 0:
        for m in models.values():
            m.to_gpu(gpu)

    optimizers = {k: make_optimizer(v, stepsize) for k, v in models.items()}
    updater = MultiUpdater(train_iter, optimizers, models, device=gpu, loss_func=loss_funcs)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(MultiEvaluator(test_iter, models, device=gpu))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'nnPU/loss', 'test/nnPU/error','test/nnPU/precision', 'test/nnPU/recall', 'elapsed_time']))
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['nnPU/loss'], 'epoch', file_name=train_output_path))
        trainer.extend(
            extensions.PlotReport(['test/nnPU/error'], 'epoch', file_name=test_output_path))
    print("prior: {}".format(prior))
    print("stepsize: {}".format(stepsize))
    print("loss: {}".format(loss))
    print("epoch: {}".format(epoch))
    print("batchsize: {}".format(batchsize))
    print("model: {}".format(selected_model))
    print("beta: {}".format(beta))
    print("gamma: {}".format(gamma))
    # run training
    trainer.run()
    return models['nnPU']

