import numpy as np
import tvm
from tvm import te, auto_scheduler

import sys
import os

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

def main():
    argv = sys.argv
    if len(argv) != 2:
        print("Invalid input!")
        print("python mm_tvm.py THREADNUM!")
        exit(1)

    os.environ["TVM_NUM_THREADS"] = argv[1]
    with open("results.txt", "w") as f:
        f.write("Results\n")
    for M, N, K in sizes:
        print("M=%d N=%d K=%d" % (M, N, K))
        target = tvm.target.Target("llvm -mcpu=core-avx2")
        task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)

        #print("Computational DAG:")
        #print(task.compute_dag)

        log_file = f"matmul_add_{M}_{N}_{K}.json"
        runner = tvm.auto_scheduler.LocalRunner(
            timeout = 2000,
        )
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials = 1000,
            runner = runner,
            measure_callbacks = [auto_scheduler.RecordToFile(log_file)],
            verbose = 0,
        )

        # Run auto-tuning (search)
        task.tune(tune_option)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)


        func = tvm.build(sch, args, target)
        a_np = np.random.uniform(size=(M, K)).astype(np.float32)
        b_np = np.random.uniform(size=(K, N)).astype(np.float32)
        c_np = np.random.uniform(size=(M, N)).astype(np.float32)
        #out_np = 0.674*a_np.dot(b_np) + 0.013*c_np
        out_np = a_np.dot(b_np) + c_np

        dev = tvm.cpu()
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        out_tvm = tvm.nd.empty(out_np.shape, device=dev)
        func(a_tvm, b_tvm, c_tvm, out_tvm)

        # Check results
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

        # Evaluate execution time.
        evaluator = func.time_evaluator(func.entry_name, dev, number = 9)
        print(
            "ThreadNum=%s M=%d N=%d K=%d    Performance: %.0f GFLOPs/s" 
            % (os.environ["TVM_NUM_THREADS"], M, N, K, 2.0e-9*M*N*K/np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results))
        )
        
        # write to the results.txt
        
        with open("results.txt", "a") as f:
            f.write(
                "ThreadNum=%s M=%d N=%d K=%d    Performance: %.0f GFLOPs/s\n" 
                % (os.environ["TVM_NUM_THREADS"], M, N, K, 2.0e-9*M*N*K/np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results))
            )


if __name__ == "__main__":
    main()


