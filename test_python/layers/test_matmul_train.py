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
        def matmul_grad(float(M,N) A, float(N,K) B, float(M,K) d_O) -> (d_A, d_B){
          d_A(i, j) +=! d_O(i, kk) * B(j, kk)
          d_B(i, j) +=! d_O(kk, j) * A(kk, i)
        }
        """

        matmul = tc.define(LANG, name="matmul", training=True, backward="matmul_grad")
        mat1 = Parameter(torch.randn(3, 4).cuda())
        mat2 = Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        out = matmul(mat1, mat2, options=[tc.CudaMappingOptions("mlp"), tc.CudaMappingOptions("mlp")])
        out.sum().backward()


if __name__ == '__main__':
    unittest.main()
