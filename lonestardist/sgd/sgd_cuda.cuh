#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "sgd_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

#define RESIDUAL_LATENT_VECTOR_SIZE 20
#define LATENT_VECTOR_SIZE 20

struct CUDA_Context : public CUDA_Context_Common {
    struct CUDA_Context_Field<double> residual_latent_vector;
    struct CUDA_Context_Field<double> latent_vector;
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
	mem_usage += mem_usage_CUDA_vector_field(&ctx->residual_latent_vector, g, num_hosts, RESIDUAL_LATENT_VECTOR_SIZE);
	mem_usage += mem_usage_CUDA_vector_field(&ctx->latent_vector, g, num_hosts, LATENT_VECTOR_SIZE);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
    load_graph_CUDA_vector_field(ctx, &ctx->residual_latent_vector, num_hosts, RESIDUAL_LATENT_VECTOR_SIZE);
    load_graph_CUDA_vector_field(ctx, &ctx->latent_vector, num_hosts, LATENT_VECTOR_SIZE);
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

std::vector<double> get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID) {
	double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_rd_ptr();
    std::vector<double> v;
    v.resize(RESIDUAL_LATENT_VECTOR_SIZE);
    memcpy(&v[0], &residual_latent_vector[LID], sizeof(double) * RESIDUAL_LATENT_VECTOR_SIZE);
	return v;
}

double get_element_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex) {
    double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_rd_ptr();
    return residual_latent_vector[LID + vecIndex];
}

void set_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v) {
	double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    memcpy(&residual_latent_vector[LID], &v[0], sizeof(double) * RESIDUAL_LATENT_VECTOR_SIZE);
}

void set_element_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex, double v) {
	double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    residual_latent_vector[LID + vecIndex] = v;
}

void reset_vector_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, double v) {
	double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < RESIDUAL_LATENT_VECTOR_SIZE; ++i)
        residual_latent_vector[LID + i] = v;
}

void pair_wise_avg_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v) {
    double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < RESIDUAL_LATENT_VECTOR_SIZE; ++i)
        residual_latent_vector[LID + i] = (residual_latent_vector[LID + i] + v[i]) / (double) 2;
}

void pair_wise_add_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v) {
    double *residual_latent_vector = ctx->residual_latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < RESIDUAL_LATENT_VECTOR_SIZE; ++i)
        residual_latent_vector[LID + i] = (double) (residual_latent_vector[LID + i] + v[i]);
}

void batch_get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
    batch_get_shared_vector_field<double, sharedMaster, false>(ctx, &ctx->residual_latent_vector, from_id, v, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_get_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
    batch_get_shared_vector_field<double, sharedMaster, false>(ctx, &ctx->residual_latent_vector, from_id, v, v_size, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_get_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
    batch_get_shared_vector_field<double, sharedMirror, false>(ctx, &ctx->residual_latent_vector, from_id, v, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_get_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
    batch_get_shared_vector_field<double, sharedMirror, false>(ctx, &ctx->residual_latent_vector, from_id, v, v_size, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_get_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, double i) {
	batch_get_shared_vector_field<double, sharedMirror, true>(ctx, &ctx->residual_latent_vector, from_id, v, RESIDUAL_LATENT_VECTOR_SIZE, i);
}

void batch_get_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, double i) {
	batch_get_shared_vector_field<double, sharedMirror, true>(ctx, &ctx->residual_latent_vector, from_id, v, v_size, data_mode, RESIDUAL_LATENT_VECTOR_SIZE, i);
}

void batch_set_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMirror, setArrOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_set_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMaster, setArrOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_pair_wise_avg_array_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMirror, avgArrOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_pair_wise_avg_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMaster, avgArrOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_pair_wise_add_array_mirror_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMirror, addArrOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_pair_wise_add_array_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMaster, addArrOp>(ctx, &ctx->residual_latent_vector, from_id, v, data_mode, RESIDUAL_LATENT_VECTOR_SIZE);
}

void batch_reset_node_residual_latent_vector_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, double v) {
	reset_data_field_vector<double>(&ctx->residual_latent_vector, begin, end, v, RESIDUAL_LATENT_VECTOR_SIZE);
}

void get_bitset_latent_vector_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->latent_vector.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_latent_vector_reset_cuda(struct CUDA_Context* ctx) {
	ctx->latent_vector.is_updated.cpu_rd_ptr()->reset();
}

void bitset_latent_vector_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->latent_vector, begin, end);
}

std::vector<double> get_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID) {
	double *latent_vector = ctx->latent_vector.data.cpu_rd_ptr();
    std::vector<double> v;
    v.resize(LATENT_VECTOR_SIZE);
    memcpy(&v[0], &latent_vector[LID], sizeof(double) * LATENT_VECTOR_SIZE);
	return v;
}

double get_element_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex) {
    double *latent_vector = ctx->latent_vector.data.cpu_rd_ptr();
    return latent_vector[LID + vecIndex];
}

void set_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v) {
	double *latent_vector = ctx->latent_vector.data.cpu_wr_ptr();
    memcpy(&latent_vector[LID], &v[0], sizeof(double) * LATENT_VECTOR_SIZE);
}

void set_element_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, unsigned vecIndex, double v) {
	double *latent_vector = ctx->latent_vector.data.cpu_wr_ptr();
    latent_vector[LID + vecIndex] = v;
}

void reset_vector_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, double v) {
	double *latent_vector = ctx->latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        latent_vector[LID + i] = v;
}

void pair_wise_avg_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v) {
    double *latent_vector = ctx->latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        latent_vector[LID + i] = (latent_vector[LID + i] + v[i]) / (double) 2;
}

void pair_wise_add_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned LID, std::vector<double> v) {
    double *latent_vector = ctx->latent_vector.data.cpu_wr_ptr();
    for(int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        latent_vector[LID + i] = (double) (latent_vector[LID + i] + v[i]);
}

void batch_get_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
    batch_get_shared_vector_field<double, sharedMaster, false>(ctx, &ctx->latent_vector, from_id, v, LATENT_VECTOR_SIZE);
}

void batch_get_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
    batch_get_shared_vector_field<double, sharedMaster, false>(ctx, &ctx->latent_vector, from_id, v, v_size, data_mode, LATENT_VECTOR_SIZE);
}

void batch_get_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
    batch_get_shared_vector_field<double, sharedMirror, false>(ctx, &ctx->latent_vector, from_id, v, LATENT_VECTOR_SIZE);
}

void batch_get_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
    batch_get_shared_vector_field<double, sharedMirror, false>(ctx, &ctx->latent_vector, from_id, v, v_size, data_mode, LATENT_VECTOR_SIZE);
}

void batch_get_reset_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, double i) {
	batch_get_shared_vector_field<double, sharedMirror, true>(ctx, &ctx->latent_vector, from_id, v, LATENT_VECTOR_SIZE, i);
}

void batch_get_reset_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, double i) {
	batch_get_shared_vector_field<double, sharedMirror, true>(ctx, &ctx->latent_vector, from_id, v, v_size, data_mode, LATENT_VECTOR_SIZE, i);
}

void batch_set_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMirror, setArrOp>(ctx, &ctx->latent_vector, from_id, v, data_mode, LATENT_VECTOR_SIZE);
}

void batch_set_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMaster, setArrOp>(ctx, &ctx->latent_vector, from_id, v, data_mode, LATENT_VECTOR_SIZE);
}

void batch_pair_wise_avg_array_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMirror, avgArrOp>(ctx, &ctx->latent_vector, from_id, v, data_mode, LATENT_VECTOR_SIZE);
}

void batch_pair_wise_avg_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMaster, avgArrOp>(ctx, &ctx->latent_vector, from_id, v, data_mode, LATENT_VECTOR_SIZE);
}

void batch_pair_wise_add_array_mirror_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMirror, addArrOp>(ctx, &ctx->latent_vector, from_id, v, data_mode, LATENT_VECTOR_SIZE);
}

void batch_pair_wise_add_array_node_latent_vector_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_vector_field<double, sharedMaster, addArrOp>(ctx, &ctx->latent_vector, from_id, v, data_mode, LATENT_VECTOR_SIZE);
}

void batch_reset_node_latent_vector_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, double v) {
	reset_data_field_vector<double>(&ctx->latent_vector, begin, end, v, LATENT_VECTOR_SIZE);
}

