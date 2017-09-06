/**-*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Macros for easy construction of sync structures for distributed Galois 
 * programs. Define a reductions for all fields as well as a broadcast. If
 * using bitsets, then define the bitset as well (see the bitset section
 * for more details on this).
 *
 * A NodeData struct must be declared before including this file. It must 
 * have the field names that you pass into these macros.
 * You need to add a semicolon at the end of the macros since they expand into
 * structs.
 *
 * Example:
 *
 * // This is declared in your main file.
 * struct NodeData {
 *   std::atomic<unsigned int> current_degree;
 * }
 *
 * // These are written in wherever you want your sync structs to go. It should
 * // go after the declaration of your NodeData struct.
 * GALOIS_SYNC_STRUCTURE_REDUCE_ADD(current_degree, unsigned int);
 * GALOIS_SYNC_STRUCTURE_BROADCAST(current_degree, unsigned int);
 *
 * 
 * WARNING: "bool" is NOT supported as a field type: the code uses the data
 * operation on C vectors, and a vector<bool> has a specialized version of
 * the data operation that causes problems. Use a uint8_t instead.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#ifndef SYNC_STRUCT_MACROS
#define SYNC_STRUCT_MACROS


////////////////////////////////////////////////////////////////////////////////
// Field flag structure
////////////////////////////////////////////////////////////////////////////////

class FieldFlags {
private:
  uint8_t _s2s;
  uint8_t _s2d;
  uint8_t _d2s;
  uint8_t _d2d;
public:
  /**
   * Field Flags constructor. Sets all flags to false
   */
  FieldFlags() {
    _s2s = false;
    _s2d = false;
    _d2s = false;
    _d2d = false;
  }

  bool src_to_src() {
    return _s2s;
  }

  bool src_to_dst() {
    return _s2d;
  }

  bool dst_to_src() {
    return _d2s;
  }

  bool dst_to_dst() {
    return _d2d;
  }

  void set_write_src() {
    _s2s = true;
    _s2d = true;
  }

  void set_write_dst() {
    _d2s = true;
    _d2d = true;
  }

  void set_write_any() {
    _s2s = true;
    _s2d = true;
    _d2s = true;
    _d2d = true;
  }

  void clear_read_src() {
    _s2s = false;
    _d2s = false;
  }

  void clear_read_dst() {
    _s2d = false;
    _d2d = false;
  }
  
  void clear_all() {
    _s2s = false;
    _s2d = false;
    _d2s = false;
    _d2d = false;
  }
};

// Deprecated; use the class above with the sync on demand call that takes
// flags as an argument
#define GALOIS_SYNC_STRUCTURE_FLAGS(fieldname)\
static uint8_t _##fieldname##_s2s = false;\
static uint8_t _##fieldname##_s2d = false;\
static uint8_t _##fieldname##_d2s = false;\
static uint8_t _##fieldname##_d2d = false;\
\
struct Flags_##fieldname {\
  static bool src_to_src() {\
    return _##fieldname##_s2s;\
  }\
\
  static bool src_to_dst() {\
    return _##fieldname##_s2d;\
  }\
\
  static bool dst_to_src() {\
    return _##fieldname##_d2s;\
  }\
\
  static bool dst_to_dst() {\
    return _##fieldname##_d2d;\
  }\
\
  static void set_write_src() {\
    _##fieldname##_s2s = true;\
    _##fieldname##_s2d = true;\
  }\
\
  static void set_write_dst() {\
    _##fieldname##_d2s = true;\
    _##fieldname##_d2d = true;\
  }\
\
  static void set_write_any() {\
    _##fieldname##_s2s = true;\
    _##fieldname##_s2d = true;\
    _##fieldname##_d2s = true;\
    _##fieldname##_d2d = true;\
  }\
\
  static void clear_read_src() {\
    _##fieldname##_s2s = false;\
    _##fieldname##_d2s = false;\
  }\
\
  static void clear_read_dst() {\
    _##fieldname##_s2d = false;\
    _##fieldname##_d2d = false;\
  }\
};

////////////////////////////////////////////////////////////////////////////////
// Reduce Add
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, y, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      add_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { Galois::add(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_add_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, (ValTy)0);\
    }\
    else if (personality == CPU)\
    Galois::set(node.fieldname, (ValTy)0);\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { Galois::add(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    Galois::set(node.fieldname, (ValTy)0);\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD_ARRAY(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, y, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      add_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { Galois::add(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_add_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, (ValTy)0);\
    }\
    else if (personality == CPU)\
    Galois::set(fieldname[node_id], (ValTy)0);\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD_ARRAY(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { Galois::add(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    Galois::set(fieldname[node_id], (ValTy)0);\
  }\
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Reduce Set
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { Galois::set(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { Galois::set(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET_ARRAY(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { Galois::set(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET_ARRAY(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { Galois::set(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Reduce Min
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      return y < min_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    }\
    assert(personality == CPU);\
    { return y < Galois::min(node.fieldname, y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_min_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                        data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { return y < Galois::min(node.fieldname, y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN_ARRAY(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      return y < min_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    }\
    assert(personality == CPU);\
    { return y < Galois::min(fieldname[node_id], y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_min_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                        data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN_ARRAY(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { return y < Galois::min(fieldname[node_id], y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(fieldname, fieldtype) \
struct Reduce_pair_wise_avg_array_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { Galois::pairWiseAvg_vec(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    { Galois::resetVec(node.fieldname); }\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(fieldname, fieldtype) \
struct Reduce_pair_wise_avg_array_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  unsigned long long int *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { Galois::pairWiseAvg_vec(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    { Galois::resetVec(node.fieldname); }\
  }\
}
#endif


////////////////////////////////////////////////////////////////////////////////
// Broadcast
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_BROADCAST(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            unsigned long long int *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    if (personality == GPU_CUDA)\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    else if (personality == CPU)\
    node.fieldname = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                               data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
}
#else
#define GALOIS_SYNC_STRUCTURE_BROADCAST(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    return node.fieldname;\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            unsigned long long int *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    node.fieldname = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_BROADCAST_ARRAY(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            unsigned long long int *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    if (personality == GPU_CUDA)\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    else if (personality == CPU)\
    fieldname[node_id] = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                               data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
}
#else
#define GALOIS_SYNC_STRUCTURE_BROADCAST_ARRAY(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            unsigned long long int *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    fieldname[node_id] = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           unsigned long long int *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Bitset struct
////////////////////////////////////////////////////////////////////////////////

// Important: Bitsets are expected to have the following naming scheme:
// bitset_<fieldname>
// In addition, you will have to declare and appropriately resize the bitset 
// in your main program as well as set the bitset appropriately (i.e. when you
// do a write to a particular node).

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_BITSET(fieldname)\
struct Bitset_##fieldname {\
  static bool is_valid() {\
    return true;\
  }\
\
  static Galois::DynamicBitSet& get() {\
    if (personality == GPU_CUDA) \
      get_bitset_##fieldname##_cuda(cuda_ctx,\
        (unsigned long long int *)bitset_##fieldname.get_vec().data());\
    return bitset_##fieldname;\
  }\
\
  static void reset_range(size_t begin, size_t end) {\
    if (personality == GPU_CUDA) {\
      bitset_##fieldname##_reset_cuda(cuda_ctx, begin, end);\
    } else\
      { assert (personality == CPU);\
      bitset_##fieldname.reset(begin, end); }\
  }\
}
#else
// no GPU code
#define GALOIS_SYNC_STRUCTURE_BITSET(fieldname)\
struct Bitset_##fieldname {\
  static bool is_valid() {\
    return true;\
  }\
\
  static Galois::DynamicBitSet& get() {\
    return bitset_##fieldname;\
  }\
\
  static void reset_range(size_t begin, size_t end) {\
    bitset_##fieldname.reset(begin, end);\
  }\
}
#endif

#endif // header guard
