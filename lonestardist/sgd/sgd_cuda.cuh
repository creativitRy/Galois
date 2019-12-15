#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "bfs_pull_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

#define LATENT_VECTOR_SIZE 20

struct CUDA_Context : public CUDA_Context_Common {
    struct CUDA_Context_Field<double*> residual_latent_vector;
    struct CUDA_Context_Field<double*> latent_vector;
};

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context* ) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
	size_t mem_usage = mem_usage_CUDA_common(g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->residual_latent_vector, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
    load_graph_CUDA_field_array(ctx, &ctx->residual_latent_vector, num_hosts, LATENT_VECTOR_SIZE);
    load_graph_CUDA_field_array(ctx, &ctx->latent_vector, num_hosts, LATENT_VECTOR_SIZE);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->residual_latent_vector.data.zero_gpu();
	ctx->latent_vector.data.zero_gpu();
}

void get_bitset_residual_latent_vector_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->residual_latent_vector.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_residual_latent_vector_reset_cuda(struct CUDA_Context* ctx) {
	ctx->residual_latent_vector.is_updated.cpu_rd_ptr()->reset();
}

void bitset_residual_latent_vector_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->residual_latent_vector, begin, end);
}

std::vector<galois::CopyableAtomic<double>> get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID) {
	double **residual_latent_vector = ctx->residual_latent_vector.data.cpu_rd_ptr();
	return residual_latent_vector[LID];
}

void set_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, double* v) {
	double **residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < LATENT_VECTOR_SIZE; i++) {
        residual_latent_vector[i] = v[i];
    }
}

void pair_wise_add_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	double **residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < LATENT_VECTOR_SIZE; i++) {
        residual_latent_vector[i] += v[i];
    }
}

void batch_get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<double*, sharedMaster, false>(ctx, &ctx->residual_latent_vector, from_id, v);
}

void batch_get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<double*, sharedMaster, false>(ctx, &ctx->residual_latent_vector, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<double*, sharedMirror, false>(ctx, &ctx->residual_latent_vector, from_id, v);
}

void batch_get_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<double*, sharedMirror, false>(ctx, &ctx->residual_latent_vector, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, double* i) {
	batch_get_shared_field<double*, sharedMirror, true>(ctx, &ctx->residual_latent_vector, from_id, v, i);
}

void batch_get_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, double* i) {
	batch_get_shared_field<double*, sharedMirror, true>(ctx, &ctx->residual_latent_vector, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<double*, sharedMirror, setOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode);
}

void batch_set_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<double*, sharedMaster, setOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode);
}

void batch_add_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<double*, sharedMirror, addOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode);
}

void batch_add_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<double*, sharedMaster, addOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode);
}

void batch_min_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<double*, sharedMirror, minOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode);
}

void batch_min_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<double*, sharedMaster, minOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode);
}

void batch_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, double* v) {
	reset_data_field<double*>(&ctx->residual_latent_vector, begin, end, v);
}

