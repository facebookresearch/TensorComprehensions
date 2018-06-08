Installation in the Google Colaboratory environment
===================================================

If you want to install TC in a Google Colaboratory environment, copy/paste and run
the following code in the notebook. Please note, it will take 2-3 minutes to execute.

Step 1: Create new Notebook in the Google Research Colaboratory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open https://colab.research.google.com/ ,  create a new notebook and switch Runtime to GPU.

Step 2: Create a new Code Cell, with the following code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    !wget -c https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
    !chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
    !bash ./Anaconda3-5.1.0-Linux-x86_64.sh -b -f -p /usr/local
    !conda install -q -y --prefix /usr/local -c pytorch -c tensorcomp tensor_comprehensions
    
    import sys
    sys.path.append('/usr/local/lib/python3.6/site-packages/')
    
Step 3: Use TC normally, from Python/Torch environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, paste and execute the following code in a new Code Cell:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)
