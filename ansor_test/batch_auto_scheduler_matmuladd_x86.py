import os
import numpy as np
import tvm
from tvm import te, auto_scheduler
import time

# # MLP bert lart/ basic
# sizes=[
#     #Bert large
# [512,64,1024],      #BMATmul
# [512,4096,1024],    #MLP1
# [512,1024,4096],    #MLP2

#     #Bert basic
# [512,64,768],       #BMATmul
# [512,3072,768],     #MLP1
# [512,768,3072],     #MLP2
# ]

# From LLMs
sizes = [
    # Llama3
    [4096, 128, 4096],
    [128, 8192, 4096],
    [128, 4096, 8192],
    [4096, 4096, 4096],

    # Gemma27B
    [4608, 256, 4096],
    [256, 8192, 4608],
    [256, 4608, 8192],
    [4608, 4608, 36864],

    # Gemma9B
    [3584, 256, 4096],
    [256, 8192, 3584],
    [256, 3584, 8192],
    [3584, 3584, 14336],

    # Gemma7B
    [3072, 256, 4096],
    [256, 8192, 3072],
    [256, 3072, 8192],
    [3072, 3072, 24576],

    # Gemma2B
    [2048, 256, 4096],
    [256, 8192, 2048],
    [256, 2048, 8192],
    [2048, 2048, 16384],
]

def matmul_gflops(input_shape, exe_time):
    M, N, K = input_shape
    flops = 2 * M * N * K
    gflops = flops / exe_time / 1e9
    return gflops

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name = "A", dtype = dtype)
    B = te.placeholder((K, N), name = "B", dtype = dtype)
    C = te.placeholder((M, N), name = "C", dtype = dtype)

    k = te.reduce_axis((0, K), name = "k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis = k),
        name = "matmul",
        attrs = {"layout_free_placeholders": [B]},
    )

    #out = te.compute((M, N), lambda i, j: 0.674*matmul[i, j] + 0.013*C[i, j], name = "out")
    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name = "out")

    return [A, B, C, out]

def testmatmul_add():
    combined_res_file = 'results.txt'
    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        print("M=",M,"N=",N,"K=",L)
        target = tvm.target.Target("llvm -mcpu=core-avx2")
        # target = tvm.target.Target("llvm -mcpu=skylake-avx512")
        task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, L, "float32"), target=target)

        # Inspect the computational graph
        print("Computational DAG:")
        print(task.compute_dag)

        log_file = "cpu_testCase_" + str(i) +"matmul_add_M"+str(M)+"_N"+str(N)+"_K"+str(L)+".json"
        
        start_time = int(time.time())
        csv_file_path = log_file.replace('.json', '.csv')
        
        # write the start time to the csv file
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_file.write(f"start_time:{str(start_time)}\n")
            
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=1000,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )

        # Run auto-tuning (search)
        task.tune(tune_option)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)
        
        func = tvm.build(sch, args, target)
        
        a_np = np.random.uniform(size=(M, L)).astype(np.float32)
        b_np = np.random.uniform(size=(L, N)).astype(np.float32)
        c_np = np.random.uniform(size=(M, N)).astype(np.float32)
        out_np = a_np.dot(b_np) + c_np

        dev = tvm.cpu()
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        out_tvm = tvm.nd.empty(out_np.shape, device=dev)
        func(a_tvm, b_tvm, c_tvm, out_tvm)
        
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        # Check results
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
        
        # in seconds
        input_shape = (M, N, L)
        gflops = matmul_gflops(input_shape, np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results))
        
        # write the gflops to the csv file
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_file.write(f"gflops:{str(gflops)}\n")
            
        with open(combined_res_file, 'a') as f:
            f.write(f"Problem:{i} M:{M} N:{N} K:{L} GFLOPS:{gflops}\n")

if __name__ == '__main__':
    testmatmul_add()
