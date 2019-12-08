#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "gen_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<float* /*ARRAY 4*/> some_value;
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
	mem_usage += mem_usage_CUDA_field(&ctx->some_value, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_array_field(ctx, &ctx->some_value, num_hosts, 4);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->some_value.data.zero_gpu();
}

void get_bitset_some_value_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->some_value.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_some_value_reset_cuda(struct CUDA_Context* ctx) {
	ctx->some_value.is_updated.cpu_rd_ptr()->reset();
}

void bitset_some_value_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->some_value, begin, end);
}

float* /*ARRAY 4*/ get_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID) {
	float* /*ARRAY 4*/ *some_value = ctx->some_value.data.cpu_rd_ptr();
	return some_value[LID];
}

void set_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID, float* /*ARRAY 4*/ v) {
	float* /*ARRAY 4*/ *some_value = ctx->some_value.data.cpu_wr_ptr();
	int i;
for(i = 0; i < 4; ++i)
		some_value[LID][i] = v[i];
}

void add_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID, float* /*ARRAY 4*/ v) {
	float* /*ARRAY 4*/ *some_value = ctx->some_value.data.cpu_wr_ptr();
	int i;
for(i = 0; i < 4; ++i)
		some_value[LID][i] += v[i];
}

bool min_node_some_value_cuda(struct CUDA_Context* ctx, unsigned LID, float* /*ARRAY 4*/ v) {
	float* /*ARRAY 4*/ *some_value = ctx->some_value.data.cpu_wr_ptr();
	if (some_value[LID] > v){
		some_value[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float* /*ARRAY 4*/, sharedMaster, false>(ctx, &ctx->some_value, from_id, v);
}

void batch_get_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float* /*ARRAY 4*/, sharedMaster, false>(ctx, &ctx->some_value, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float* /*ARRAY 4*/, sharedMirror, false>(ctx, &ctx->some_value, from_id, v);
}

void batch_get_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float* /*ARRAY 4*/, sharedMirror, false>(ctx, &ctx->some_value, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, float* /*ARRAY 4*/ i) {
	batch_get_shared_field<float* /*ARRAY 4*/, sharedMirror, true>(ctx, &ctx->some_value, from_id, v, i);
}

void batch_get_reset_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, float* /*ARRAY 4*/ i) {
	batch_get_shared_field<float* /*ARRAY 4*/, sharedMirror, true>(ctx, &ctx->some_value, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float* /*ARRAY 4*/, sharedMirror, setOp>(ctx, &ctx->some_value, from_id, v, data_mode);
}

void batch_set_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float* /*ARRAY 4*/, sharedMaster, setOp>(ctx, &ctx->some_value, from_id, v, data_mode);
}

void batch_add_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float* /*ARRAY 4*/, sharedMirror, addOp>(ctx, &ctx->some_value, from_id, v, data_mode);
}

void batch_add_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float* /*ARRAY 4*/, sharedMaster, addOp>(ctx, &ctx->some_value, from_id, v, data_mode);
}

void batch_min_mirror_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float* /*ARRAY 4*/, sharedMirror, minOp>(ctx, &ctx->some_value, from_id, v, data_mode);
}

void batch_min_node_some_value_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float* /*ARRAY 4*/, sharedMaster, minOp>(ctx, &ctx->some_value, from_id, v, data_mode);
}

void batch_reset_node_some_value_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float* /*ARRAY 4*/ v) {
	reset_data_field<float* /*ARRAY 4*/>(&ctx->some_value, begin, end, v);
}

