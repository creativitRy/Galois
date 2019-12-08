#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void get_bitset_some_value_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_some_value_reset_cuda(struct CUDA_Context* ctx);
void bitset_some_value_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
float* /*ARRAY 4*/ get_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID, float* /*ARRAY 4*/ v);
void add_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID, float* /*ARRAY 4*/ v);
bool min_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID, float* /*ARRAY 4*/ v);
void batch_get_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, float* /*ARRAY 4*/ i);
void batch_get_reset_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, float* /*ARRAY 4*/ i);
void batch_set_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode);
void batch_set_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_some_value_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float* /*ARRAY 4*/ v);

void Work_cuda(unsigned int __begin, unsigned int __end, galois::GAccumulator<float> & local_sum, struct CUDA_Context* ctx);
void Work_allNodes_cuda(galois::GAccumulator<float> & local_sum, struct CUDA_Context* ctx);
void Work_masterNodes_cuda(galois::GAccumulator<float> & local_sum, struct CUDA_Context* ctx);
void Work_nodesWithEdges_cuda(galois::GAccumulator<float> & local_sum, struct CUDA_Context* ctx);
