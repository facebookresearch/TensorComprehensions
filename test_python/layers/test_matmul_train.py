import tensor_comprehensions as tc

import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import unittest


class TestTrainMatMul(unittest.TestCase):

    def test_train_matmul(self):
        LANG = """
        def matmul(float(M,N) A, float(N,K) B) -> (output) {
          output(i, j) +=! A(i, kk) * B(kk, j)
        }
        def matmul_grad(float(M,N) A, float(N,K) B, float(M,K) O_grad) -> (A_grad, B_grad){
          A_grad(i, j) +=! O_grad(i, kk) * B(j, kk)
          B_grad(i, j) +=! O_grad(kk, j) * A(kk, i)
        }
        """

        matmul = tc.define(MATMUL_LANG, name="matmul", training=True, backward="matmul_grad")
        mat1 = Parameter(torch.randn(3, 4).cuda())
        mat2 = Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        out = matmul(mat1, mat2, options=[tc.Options("mlp"), tc.Options("mlp")])
        out.sum().backward()


if __name__ == '__main__':
unittest.main()
