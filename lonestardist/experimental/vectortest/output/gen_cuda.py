from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("gen_cuda.cuh", system = False)], parse = False),
Kernel("Work", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('galois::GAccumulator<float> ', 'local_sum'), ('float* /*ARRAY 4*/ *', 'p_some_value')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["atomicTestAdd(&p_some_value[src][2], 1.0f)"]),
CBlock(["local_sum += p_some_value[src][2]"]),
]),
]),
]),
Kernel("Work_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('galois::GAccumulator<float> &', 'local_sum'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("Work", ("ctx->gg", "__begin", "__end", "local_sum", "ctx->some_value.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("Work_allNodes_cuda", [('galois::GAccumulator<float> &', 'local_sum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["Work_cuda(0, ctx->gg.nnodes, local_sum, ctx)"]),
], host = True),
Kernel("Work_masterNodes_cuda", [('galois::GAccumulator<float> &', 'local_sum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["Work_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_sum, ctx)"]),
], host = True),
Kernel("Work_nodesWithEdges_cuda", [('galois::GAccumulator<float> &', 'local_sum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["Work_cuda(0, ctx->numNodesWithEdges, local_sum, ctx)"]),
], host = True),
])
