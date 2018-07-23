import tensor_comprehensions as tc
import torch

tc_name = "convolution"
tc_code = """
  def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
  O(n, m, h, w) +=! I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
  }
"""

N, C, H, W, O, kH, kW = 8, 2, 28, 28, 8, 1, 1 #32, 4, 56, 56, 16, 1, 1
I, W1 = torch.randn(N, C, H, W, device='cuda'), torch.randn(O, C, kH, kW, device='cuda')
inp = (I, W1)

opts = tc.MappingOptions("naive")
opts.mapToBlocks([2])
opts.mapToThreads([16, 7, 32])#tile([5])
opts.useSharedMemory(1)
opts.outerScheduleFusionStrategy("Preserve3Coincident")
opts.intraTileScheduleFusionStrategy("Max")
opts.tile([4,1,8])
opts.unroll(8)
opts.unrollCopyShared(0)
opts.useReadOnlyCache(0)
opts.fixParametersBeforeScheduling(0)
opts.usePrivateMemory(1)
opts.privateDepth(2)

tc.autotune(tc_code, tc_name, *inp, starting_options=opts)
