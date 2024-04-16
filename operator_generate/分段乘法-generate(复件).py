import sys

with open(sys.argv[2], 'w') as f:
    n_parts = str(2**int(sys.argv[1]))
    burst = 2048//int(n_parts)
    f.write('''from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np


def tg(a, b, kernel_name="tg"):
    tik_instance = tik.Tik()
    # soc_version = "Ascend910B"
    # tbe_platform.set_current_compile_soc_info(soc_version, core_type="AiCore")

    input_a_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_a_gm', scope=tik.scope_gm)
    input_b_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_b_gm', scope=tik.scope_gm,is_workspace=True)
    output_r_gm = tik_instance.Tensor("float16", (512, 1024,), name='output_r_gm', scope=tik.scope_gm)

    n_parts = '''+n_parts+'''
    # nblock = 112
    bursts = 2048
    burst = 2048//n_parts
    
    with tik_instance.for_range(0, 100) as j:
        src_a_ub = tik_instance.Tensor("float16", (32, 1024,), name='src_a_ub', scope=tik.scope_ubuf)
        src_b_ub = tik_instance.Tensor("float16", (32, 1024,), name='src_b_ub', scope=tik.scope_ubuf)
        dst_c_ub = tik_instance.Tensor("float16", (32, 1024,), name='dst_c_ub', scope=tik.scope_ubuf)
        # src_a_l1 = tik_instance.Tensor("float16", (16, 1024,), name='src_a_l1', scope=tik.scope_cbuf)
''')
    num=int(n_parts)+2
    for i in range(int(num)):
        f.write('        '+'with tik_instance.new_stmt_scope(disable_sync=True):'+'\n')
        if (i<num-2):
            f.write('           '+'tik_instance.data_move(src_a_ub['+str(i*burst*16)+'], input_a_gm['+str(i*burst*16)+'], 0, 1, burst, 0, 0)'+'\n')
            f.write('           '+'tik_instance.data_move(src_b_ub['+str(i*burst*16)+'], input_b_gm['+str(i*burst*16)+'], 0, 1, burst, 0, 0)'+'\n')
        if (0<i<num-1):
            f.write('           '+'tik_instance.vec_mul(128, dst_c_ub['+str((i-1)*burst*16)+'], src_a_ub['+str((i-1)*burst*16)+'], src_b_ub['+str((i-1)*burst*16)+'], burst//8, 8, 8, 8)'+'\n')
        if (1<i):
            f.write('           '+'tik_instance.data_move(output_r_gm['+str((i-2)*burst*16)+'], dst_c_ub['+str((i-2)*burst*16)+'], 0, 1, burst, 0, 0)'+'\n')

            
    f.write('''

        
            # tik_instance.data_move(src_a_l1[0], input_a_gm[0], 0, 1, bursts, 0, 0)
            # tik_instance.data_move(src_b_ub[0], input_b_gm[i * bursts * 16], 0, 1, bursts, 0, 0)
            # tik_instance.vec_mul(128, dst_c_ub, src_a_ub, src_b_ub, nblock, 8, 8, 8)
            # tik_instance.data_move(output_r_gm[i * bursts * 16], dst_c_ub[0], 0, 1, bursts, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_a_gm], outputs=[output_r_gm])

    return tik_instance

''')

