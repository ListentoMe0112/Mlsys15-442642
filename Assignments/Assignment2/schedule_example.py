import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def sum_broadcast(
    A: T.Buffer((256,), "float32"),
    B: T.Buffer((256,), "float32"),
) -> None:
    for i, k in T.grid(256, 256):
        with T.block("sum_broadcast"):
            vi = T.axis.spatial(256, i)
            vk = T.axis.reduce(256, k)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] += A[vk]

sch = tir.Schedule(sum_broadcast)

# Fetch the computation block of "sum_broadcast".
block = sch.get_block("sum_broadcast")
# Fetch the i, j loops of the computation.
i, j = sch.get_loops(block)
# Split i into two loops.
i_outer, i_inner = sch.split(i, factors=[2, 128])
sch.bind(i_outer, "blockIdx.x")
sch.bind(i_inner, "threadIdx.x")


def build_and_test(sch: tir.Schedule) -> None:
    import numpy as np

    # Build the scheduled function.
    f = tvm.build(sch.mod, target="cuda")

    # Create the NumPy array for testing.
    a_np = np.random.rand(256).astype("float32")
    b_np = np.broadcast_to(a_np.sum(keepdims=True), shape=(256,))

    # Run the function we scheduled and built.
    device = tvm.cuda()
    a_tvm = tvm.nd.array(a_np, device=device)
    b_tvm = tvm.nd.empty((256,), "float32", device=device)
    f(a_tvm, b_tvm)

    # Validate the result correctness.
    np.testing.assert_allclose(b_tvm.numpy(), b_np, atol=1e-5, rtol=1e-5)
    print("Test passed.")

    # Print out the CUDA source code of the function we scheduled.
    print(f"CUDA source code:\n{f.imported_modules[0].get_source()}")

A_shared = sch.cache_read(block, read_buffer_index=0, storage_scope="shared")
sch.compute_at(A_shared, i_inner)
ax0 = sch.get_loops(A_shared)[-1]
_, ax0_inner = sch.split(ax0, factors=[None, 128])
sch.bind(ax0_inner, "threadIdx.x")
sch.show()
# Run building and testing.
build_and_test(sch)

