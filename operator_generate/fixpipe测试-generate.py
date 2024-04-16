import sys

with open(sys.argv[2], 'w') as f:
    
    f.write('''from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np


def tg(a, b, kernel_name="tg"):
    tik_instance = tik.Tik()
    # soc_version = "Ascend910B"
    # tbe_platform.set_current_compile_soc_info(soc_version, core_type="AiCore")

    input_a_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_a_gm', scope=tik.scope_gm)
    input_b_gm = tik_instance.Tensor("float16", (512, 1024,), name='input_b_gm', scope=tik.scope_gm, is_workspace=True)
    output_r_gm = tik_instance.Tensor("float16", (32, 1024,), name='output_r_gm', scope=tik.scope_gm)

    n_parts = 128
    # nblock = 112
    bursts = 2048
    burst = 2048 // n_parts

    src_a_l1out = tik_instance.Tensor("float32", (32, 1024,), name='src_a_l1out', scope=tik.scope_cbuf_out)
    #src_a_ub = tik_instance.Tensor("float32", (32, 1024,), name='src_a_ub', scope=tik.scope_ubuf)
    # src_b_ub = tik_instance.Tensor("float16", (32, 1024,), name='src_b_ub', scope=tik.scope_ubuf)
    # dst_c_ub = tik_instance.Tensor("float16", (32, 1024,), name='dst_c_ub', scope=tik.scope_ubuf)
    with tik_instance.for_range(0, 100) as num_index:
        tik_instance.fixpipe( output_r_gm, src_a_l1out, 8, '''+sys.argv[1]+''' , 0, 0,extend_params={"quantize_params": {"mode": "fp322fp16", "mode_param": None}})

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_a_gm], outputs=[output_r_gm])

    return tik_instance

''')

            
  

