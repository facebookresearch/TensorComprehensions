Autograd with TC
================

To create a :code:`torch.autograd` function backed by TC one can just use the
:func:`make_autograd` helper function:

    .. code-block:: python

        conv = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
            O(n, m, h, w) +=!
                I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        }
        def convolution_igrad(float(M,C,KH,KW) W1, float(N,M,H,W) d_O)
            -> (d_I)
        {
            d_I(n, c, h, w) +=!
                d_O(  n, r_m, h - r_kh, w - r_kw) * W1(r_m, c, r_kh, r_kw)
        }
        def convolution_wgrad(float(N,C,H,W) I, float(N,M,H,W) d_O) -> (d_W1)
        {
            d_W1(m, c, kh, kw) +=!
                d_O(r_n,   m, r_h - kh, r_w - kw) *  I(r_n, c,  r_h,  r_w)
        }
        """

        N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
        T = tc.define(
            conv,
            tc.make_autotuned_options_factory(
                starting_options='naive',
                tuner_config=tuner_config))
        I, W = (
            torch.randn(N, C, H, W, device='cuda', requires_grad=True),
            torch.randn(O, C, kH, kW, device='cuda', requires_grad=True))

        def convolution_backward(I, W, d_O):
            d_I = T.convolution_igrad(W, d_O)
            d_O = T.convolution_wgrad(I, d_O)
            return (d_I, d_O)

        convolution_function = tc.make_autograd(
            T.convolution, convolution_backward)

        # First occurrence triggers tuning
        out = convolution_function(I, W)
        out.sum().backward()

        # Subsequent occurrences do not
        out = convolution_function(I, W)
        out.sum().backward()
