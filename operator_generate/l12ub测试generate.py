import sys

with open(sys.argv[2], 'w') as f:
    
    f.write('''
from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np


def tg(a, b, kernel_name="tg"):
    tik_instance = tik.Tik()

    input_a_gm = tik_instance.Tensor("float16", (64, 1024), name='input_a_gm', scope=tik.scope_gm)
    output_r_gm = tik_instance.Tensor("float16", (32, 1024,), name='output_r_gm', scope=tik.scope_gm)

    src_ub = tik_instance.Tensor("float16", (64,1024), name="src_ub", scope=tik.scope_ubuf)
    dst_l1 = tik_instance.Tensor("float16", (64,1024), name="dst_ub", scope=tik.scope_cbuf)
  
    with tik_instance.for_range(0,100) as index:
      tik_instance.data_move( src_ub,dst_l1, 0, 8,'''+sys.argv[1]+''', 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_a_gm], outputs=[output_r_gm])

    return tik_instance
''')
