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

    input_a_gm = tik_instance.Tensor("float16", (64, 1024), name='input_a_gm', scope=tik.scope_gm)
    # input_b_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_b_gm', scope=tik.scope_gm, is_workspace=True)
    output_r_gm = tik_instance.Tensor("float16", (32, 1024,), name='output_r_gm', scope=tik.scope_gm)

    dtype="int8"
    index = '''+sys.argv[1]+'''
    k=index%10+1
    m=index//10%10+1
    n=index//100+1
    matrix_cb = tik_instance.Tensor(  # 转移矩阵
        dtype, (k, 32, n*16),
        name="matrix_cb",
        scope=tik.scope_cbuf)

    input_r_cb_16 = tik_instance.Tensor(  # 实部输入
        dtype, (k, m*16, 32),
        name="input_r_cb_16",
        scope=tik.scope_cbuf)


    dst_r_cb = tik_instance.Tensor(  # 实部输出
        "int32", (n, m*16, 16),
        name="dst_r_cb_out",
        scope=tik.scope_cbuf_out)

#    matrix_cb2 = tik_instance.Tensor(  # 转移矩阵
#        dtype, (k, 16, n*16),
#        name="matrix_cb2",
#        scope=tik.scope_cbuf)

#    input_r_cb2_16 = tik_instance.Tensor(  # 实部输入
#        dtype, (k, m*16, 16),
#        name="input_r_cb2_16",
#        scope=tik.scope_cbuf)


#    dst_r_cb2 = tik_instance.Tensor(  # 实部输出
#        "float32", (n, m*16, 16),
#        name="dst_r_cb2_out",
#        scope=tik.scope_cbuf_out)

    with tik_instance.for_range(0, 100) as loop_index:
        tik_instance.matmul(dst_r_cb, input_r_cb_16, matrix_cb, m*16,k*32,n*16)
#        tik_instance.matmul(dst_r_cb, input_r_cb_16, matrix_cb, m*16,k*16,n*16,init_l1out=False)
#        tik_instance.matmul(dst_r_cb2, input_r_cb2_16, matrix_cb2, m*16,k*16,n*16)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_a_gm], outputs=[output_r_gm])

    return tik_instance
''')
