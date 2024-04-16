import sys

with open(sys.argv[2], 'w') as f:
    
    f.write('''from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np


def tg(a, b, kernel_name="tg"):
    tik_instance = tik.Tik()
    # soc_version = "Ascend910B"
    # tbe_platform.set_current_compile_soc_info(soc_version, core_type="AiCore")

    input_a_gm = tik_instance.Tensor("float16", (128, 1024,), name='input_a_gm', scope=tik.scope_gm)
    # input_b_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_b_gm', scope=tik.scope_gm, is_workspace=True)
    output_r_gm = tik_instance.Tensor("float16", (32, 1024,), name='output_r_gm', scope=tik.scope_gm)
    
    mul=16

    index = '''+sys.argv[1]+'''

    slice = [16, 32, 64, 128]
    loop = [512, 128, 32, 8]
    data = slice[index]
    loop = loop[index]
    loop = loop//mul
    size = 2 * data
    stride = (256 * 1024 // data - size) // 32

    src_a_ub = tik_instance.Tensor("float16", (100, 1024,), name='src_a_ub', scope=tik.scope_ubuf)
    # src_b_ub = tik_instance.Tensor("float32", (32, 1024,), name='src_b_ub', scope=tik.scope_ubuf)
    # dst_c_ub = tik_instance.Tensor("float16", (32, 1024,), name='dst_c_ub', scope=tik.scope_ubuf)

    with tik_instance.for_range(0, loop) as loop_index:
        tik_instance.data_move(src_a_ub, input_a_gm[loop_index * data * mul], 0, data, data * 2 // 32 * mul, stride, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_a_gm], outputs=[output_r_gm])

    return tik_instance

''')

            
  

