/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"



#define TB_SIZE 256
#define LAMBDA .0001
//const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=True $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
bool enable_lb = false;
#include "kernels/reduce.cuh"
#include "pagerank_pull_cuda.cuh"
static const int __tb_SGD = TB_SIZE;
static const int __tb_InitializeGraph = TB_SIZE;


__global__ void BFS(CSRGraph graph, unsigned int __begin, unsigned int __end, double * residual_latent_vector, DynamicBitset& bitset_residual, double* latent_vector, HGAccumulator<double> error, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_SGD;
  __shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage error_ts;
  index_type src_end;
  index_type src_rup;
  // FP: "1 -> 2;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "2 -> 3;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  unsigned d_limit = DEGREE_LIMIT;
  // FP: "3 -> 4;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  // FP: "4 -> 5;
  __shared__ npsTy nps ;
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  error.thread_entry();
  // FP: "7 -> 8;
  src_end = __end;
  src_rup = ((__begin) + roundup(((__end) - (__begin)), (blockDim.x)));
  for (index_type src = __begin + tid; src < src_rup; src += nthreads)
  {
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "8 -> 9;
    bool pop  = src < __end && ((( src < (graph).nnodes ) && ( (graph).getOutDegree(src) < DEGREE_LIMIT)) ? true: false);
    // FP: "9 -> 10;
    if (pop)
    {
    }
    // FP: "11 -> 12;
    // FP: "14 -> 15;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "15 -> 16;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "16 -> 17;
    _np_closure[threadIdx.x].src = src;
    // FP: "17 -> 18;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "20 -> 21;
    // FP: "21 -> 22;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "22 -> 23;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "23 -> 24;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "26 -> 27;
    __syncthreads();
    // FP: "27 -> 28;
    while (true)
    {

      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }

      __syncthreads();

      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {

        __syncthreads();

        break;
      }

      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }

      __syncthreads();

      int ns = nps.tb.start;
      int ne = nps.tb.size;

      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }

      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;

      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;

          double edge_rating = graph.getAbsWeight(jj);
          dst = graph.getAbsDestination(jj);
          double old_dp = 0;
          for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
            old_dp += latent_vector[src * LATENT_VECTOR_SIZE + i] * latent_vector[dst * LATENT_VECTOR_SIZE + i];
          
          double cur_error = edge_rating - old_dp;
          error.reduce(cur_error * cur_error);

          for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {

            double prevUser  = latent_vector[dst * LATENT_VECTOR_SIZE + i];
            double prevMovie = latent_vector[src * LATENT_VECTOR_SIZE + i];

            atomicTestAdd(
                &residual_latent_vector[dst * LATENT_VECTOR_SIZE + i],
                double(step_size * (cur_error * prevMovie - LAMBDA * prevUser)));
    
            atomicTestAdd(
                &residual_latent_vector[src * LATENT_VECTOR_SIZE + i],
                double(step_size * (cur_error * prevUser - LAMBDA * prevMovie)));
          }
          bitset_residual.set(src);
          bitset_residual.set(dst);

        }
      }

      __syncthreads();
    }

    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any(_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
      {
        if (_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;
          nps.warp.src[warpid] = threadIdx.x;
          _np.start = 0;
          _np.size = 0;
        }
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        assert(nps.warp.src[warpid] < __kernel_tb_size);
        src = _np_closure[nps.warp.src[warpid]].src;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type jj;
          jj = _np_w_start +_np_ii;
          {
            index_type dst;
            double edge_rating = graph.getAbsWeight(jj);
            dst = graph.getAbsDestination(jj);
            double old_dp = 0;
            for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
              old_dp += latent_vector[src * LATENT_VECTOR_SIZE + i] * latent_vector[dst * LATENT_VECTOR_SIZE + i];
            
            double cur_error = edge_rating - old_dp;
            error.reduce(cur_error * cur_error);
  
            for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
  
              double prevUser  = latent_vector[dst * LATENT_VECTOR_SIZE + i];
              double prevMovie = latent_vector[src * LATENT_VECTOR_SIZE + i];
  
              atomicTestAdd(
                  &residual_latent_vector[dst * LATENT_VECTOR_SIZE + i],
                  double(step_size * (cur_error * prevMovie - LAMBDA * prevUser)));
      
              atomicTestAdd(
                  &residual_latent_vector[src * LATENT_VECTOR_SIZE + i],
                  double(step_size * (cur_error * prevUser - LAMBDA * prevMovie)));
            }
            bitset_residual.set(src);
            bitset_residual.set(dst);
  
          }
        }
      }
      __syncthreads();

    }

    __syncthreads();
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      __syncthreads();
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          double edge_rating = graph.getAbsWeight(jj);
          dst = graph.getAbsDestination(jj);
          double old_dp = 0;
          for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
            old_dp += latent_vector[src * LATENT_VECTOR_SIZE + i] * latent_vector[dst * LATENT_VECTOR_SIZE + i];
          
          double cur_error = edge_rating - old_dp;
          error.reduce(cur_error * cur_error);

          for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {

            double prevUser  = latent_vector[dst * LATENT_VECTOR_SIZE + i];
            double prevMovie = latent_vector[src * LATENT_VECTOR_SIZE + i];

            atomicTestAdd(
                &residual_latent_vector[dst * LATENT_VECTOR_SIZE + i],
                double(step_size * (cur_error * prevMovie - LAMBDA * prevUser)));
    
            atomicTestAdd(
                &residual_latent_vector[src * LATENT_VECTOR_SIZE + i],
                double(step_size * (cur_error * prevUser - LAMBDA * prevMovie)));
          }
          bitset_residual.set(src);
          bitset_residual.set(dst);

        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  error.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(error_ts);
}



__global__ void Inspect_SGD(CSRGraph graph, unsigned int __begin, unsigned int __end, double * residual_latent_vector, DynamicBitset& bitset_residual, double* latent_vector, HGAccumulator<unsigned int> error, unsigned int step_size, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    int index;
    bool pop  = src < __end && ((( src < (graph).nnodes ) && ( (graph).getOutDegree(src) >= DEGREE_LIMIT)) ? true: false);
    if (pop)
    {
    }
    if (!pop)
    {
      continue;
    }
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
    }
  }
}

__global__ void SGD_InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, double* residual_latent_vector, double* latent_vector)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;

  unsigned int seed = TID_1D;

  curandState s;

    // seed a random number generator
  curand_init(seed, 0, 0, &s);

  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      double rand = (curand_uniform_double(&s) * 2.0) - 1.0;
      for(int i = 0; i < LATENT_VECTOR_SIZE; i++) {
        residual_latent_vector[src * LATENT_VECTOR_SIZE + i] = 0;
        latent_vector[src * LATENT_VECTOR_SIZE + i] = rand;
      }
    }
  }
}

void SDG_InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  t_work.init_thread_work(ctx->gg.nnodes);
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  SGD_InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, local_src_node, ctx->residual_latent_vector.data.gpu_wr_ptr(), ctx->latent_vector.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void SGD_InitializeGraph_allNodes_cuda(struct CUDA_Context*  ctx)
{
  SGD_InitializeGraph_cuda(0, ctx->gg.nnodes,  ctx);
}

void SGD_InitializeGraph_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  SGD_InitializeGraph_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned,  ctx);
}

void SGD_InitializeGraph_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  SGD_InitializeGraph_cuda(0, ctx->numNodesWithEdges, ctx);
}



__global__ void SGD_mergeResidual(CSRGraph graph, unsigned int __begin, unsigned int __end, double * residual_latent_vector, DynamicBitset& bitset_residual, double* latent_vector)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
        for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
            latent_vector[src * LATENT_VECTOR_SIZE + i] += residual_latent_vector[src * LATENT_VECTOR_SIZE + i];
            residual_latent_vector[src * LATENT_VECTOR_SIZE + i] = 0;
            bitset_residual.set(src);
      }
    }
  }
}

void SGD_mergeResidual_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<unsigned int> _active_vertices;
  kernel_sizing(blocks, threads);
  Shared<unsigned int> active_verticesval  = Shared<unsigned int>(1);
  *(active_verticesval.cpu_wr_ptr()) = 0;
  _active_vertices.rv = active_verticesval.gpu_wr_ptr();
  SGD_mergeResidual <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->residual_latent_vector.data.gpu_wr_ptr(), *(ctx->residual_latent_vector.is_updated.gpu_rd_ptr()), ctx->latent_vector.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  check_cuda_kernel;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
}

void SGD_mergeResidual_allNodes_cuda(struct CUDA_Context*  ctx)
{
  SGD_mergeResidual_cuda(0, ctx->gg.nnodes, ctx);
}

void SGD_mergeResidual_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  SGD_mergeResidual_cuda(ctx->beginMaster, ctx);
}

void SGD_mergeResidual_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  SGD_mergeResidual_cuda(0, ctx->numNodesWithEdges, active_vertices, local_alpha, local_tolerance, ctx);
}

void SGD_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & error, unsigned int step_size, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<unsigned int> _error;
  kernel_sizing(blocks, threads);
  Shared<double> error_val  = Shared<double>(0.0);
  *(error_val.cpu_wr_ptr()) = 0;
  _error.rv = error_val.gpu_wr_ptr();
  if (enable_lb)
  {
    t_work.reset_thread_work();
    Inspect_SGD <<<blocks, __tb_SGD>>>(ctx->gg, __begin, __end, ctx->residual_latent_vector.data.gpu_wr_ptr(), *(ctx->residual_latent_vector.is_updated.gpu_rd_ptr()), ctx->latent_vector.data.gpu_wr_ptr(), _error, step_size, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
    cudaDeviceSynchronize();
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      //SGD_TB_LB <<<blocks, __tb_SGD>>>(ctx->gg, __begin, __end, ctx->residual_latent_vector.data.gpu_wr_ptr(), *(ctx->residual_latent_vector.is_updated.gpu_rd_ptr()), _error, step_size, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
    }
  }
  SGD <<<blocks, __tb_SGD>>>(ctx->gg, __begin, __end, ctx->residual_latent_vector.data.gpu_wr_ptr(), *(ctx->residual_latent_vector.is_updated.gpu_rd_ptr()), _error, step_size, enable_lb);
  cudaDeviceSynchronize();
  check_cuda_kernel;
  error = *(error_val.cpu_rd_ptr());
}

void SGD_allNodes_cuda(double& error, unsigned int step_size, struct CUDA_Context*  ctx)
{
  SGD_cuda(0, ctx->gg.nnodes, error, step_size,  ctx);
}

void SGD_masterNodes_cuda(double& error, unsigned int step_size, struct CUDA_Context*  ctx)
{
  SGD_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, error, step_size, ctx);
}

void SGD_nodesWithEdges_cuda(double& error, unsigned int step_size, struct CUDA_Context*  ctx)
{
  SGD_cuda(0, ctx->numNodesWithEdges, error, step_size, ctx);
}