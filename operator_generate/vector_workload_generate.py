import sys

with open(sys.argv[2], 'w') as f:
    
    f.write('''
from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np


def tg(a, b, kernel_name="tg"):
    tik_instance = tik.Tik()
    # soc_version = "Ascend910B"
    # tbe_platform.set_current_compile_soc_info(soc_version, core_type="AiCore")

    input_a_gm = tik_instance.Tensor("float16", (64, 60, 1024), name='input_a_gm', scope=tik.scope_gm)
    # input_b_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_b_gm', scope=tik.scope_gm, is_workspace=True)
    output_r_gm = tik_instance.Tensor("float16", (32, 60, 1024,), name='output_r_gm', scope=tik.scope_gm)

    dtype="float16"
    index = '''+sys.argv[1]+'''
    with tik_instance.for_range(0, 30, block_num=30) as i:
        src_a_ub = tik_instance.Tensor("float16", (32, 1024,), name='src_a_ub', scope=tik.scope_ubuf)
        src_b_ub = tik_instance.Tensor("float16", (32, 1024,), name='src_b_ub', scope=tik.scope_ubuf)
        dst_c_ub = tik_instance.Tensor("float16", (32, 1024,), name='dst_c_ub', scope=tik.scope_ubuf)

        with tik_instance.for_range(0, 100, thread_num=2) as loop_index:
            tik_instance.data_move(src_a_ub, input_a_gm[64 * 1024 * i], 0, 1, index * 8, 0, 0)
            tik_instance.data_move(src_b_ub, input_a_gm[64 * 1024 * i + 32 * 1024], 0, 1, index * 8, 0, 0)
            tik_instance.vec_mul(128, dst_c_ub, src_a_ub, src_b_ub, index, 8, 8, 8)
            tik_instance.data_move(output_r_gm[32 * 1024], dst_c_ub, 0, 1, index * 8, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_a_gm], outputs=[output_r_gm])

    return tik_instance
''')
