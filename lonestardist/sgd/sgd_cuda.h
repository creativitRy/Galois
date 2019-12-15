#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void get_bitset_residual_latent_vector_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_residual_latent_vector_reset_cuda(struct CUDA_Context* ctx);
void bitset_residual_latent_vector_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
std::vector<double> get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID);
double get_element_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex);
void set_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v);
void set_element_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex, double v);
void reset_vector_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, double v);
void pair_wise_avg_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v);
void pair_wise_add_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v);
void batch_get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, double* i);
void batch_get_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, double* i);
void batch_set_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_set_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_avg_array_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_avg_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_add_array_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_add_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, double v);

void get_bitset_latent_vector_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_latent_vector_reset_cuda(struct CUDA_Context* ctx);
void bitset_latent_vector_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
std::vector<double> get_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID);
double get_element_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex);
void set_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v);
void set_element_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex, double v);
void reset_vector_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, double v);
void pair_wise_avg_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v);
void pair_wise_add_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v);
void batch_get_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, double* i);
void batch_get_reset_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, double* i);
void batch_set_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_set_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_avg_array_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_avg_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_add_array_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_pair_wise_add_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_latent_vector_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, double v);

void SGD_mergeResidual_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx);
void SGD_mergeResidual_allNodes_cuda(struct CUDA_Context*  ctx);
void SGD_mergeResidual_masterNodes_cuda(struct CUDA_Context*  ctx);
void SGD_mergeResidual_nodesWithEdges_cuda(struct CUDA_Context*  ctx);
void SDG_InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx);
void SGD_InitializeGraph_allNodes_cuda(struct CUDA_Context*  ctx);
void SGD_InitializeGraph_masterNodes_cuda(struct CUDA_Context*  ctx);
void SGD_InitializeGraph_nodesWithEdges_cuda(struct CUDA_Context*  ctx);
void SGD_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & error, unsigned int step_size, struct CUDA_Context*  ctx);
void SGD_allNodes_cuda(double& error, unsigned int step_size, struct CUDA_Context*  ctx);
void SGD_masterNodes_cuda(double& error, unsigned int step_size, struct CUDA_Context*  ctx);
void SGD_nodesWithEdges_cuda(double& error, unsigned int step_size, struct CUDA_Context*  ctx);
