import chainer.functions as F
import numpy
from chainer import cuda, function, Variable
from chainer.utils import type_check
from chainer import function_node

class QuadFunction(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 4)
        x_type, a_type = in_types[:2]


        type_check.expect(
            x_type.dtype.kind == 'f',
            a_type.dtype.kind == 'f',
            x_type.ndim == 2,
            a_type.ndim == 1,
            x_type.shape[1] == a_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == a_type.shape[0],
            )
        if type_check.eval(n_in) == 4:
            c_type = in_types[3]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == a_type.shape[0],
            )

    def forward(self, inputs):
        x = inputs[0]
        W = inputs[1]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)

        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            gx = linear(gy, W.T)
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            gW = linear(gy.T, x.T)
            ret.append(chainer.functions.cast(gW, W.dtype))
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)

        return ret

