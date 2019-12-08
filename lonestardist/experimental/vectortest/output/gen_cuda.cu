/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void Work(CSRGraph graph, unsigned int __begin, unsigned int __end, galois::GAccumulator<float>  local_sum, float* /*ARRAY 4*/ * p_some_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      atomicTestAdd(&p_some_value[src][2], 1.0f);
      local_sum += p_some_value[src][2];
    }
  }
}
void Work_cuda(unsigned int  __begin, unsigned int  __end, galois::GAccumulator<float> & local_sum, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);
  Work <<<blocks, threads>>>(ctx->gg, __begin, __end, local_sum, ctx->some_value.data.gpu_wr_ptr());
  check_cuda_kernel;
}
void Work_allNodes_cuda(galois::GAccumulator<float> & local_sum, struct CUDA_Context*  ctx)
{
  Work_cuda(0, ctx->gg.nnodes, local_sum, ctx);
}
void Work_masterNodes_cuda(galois::GAccumulator<float> & local_sum, struct CUDA_Context*  ctx)
{
  Work_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_sum, ctx);
}
void Work_nodesWithEdges_cuda(galois::GAccumulator<float> & local_sum, struct CUDA_Context*  ctx)
{
  Work_cuda(0, ctx->numNodesWithEdges, local_sum, ctx);
}