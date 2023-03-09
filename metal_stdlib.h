//
//  metal_stdlib.h
//  OpenCL Metal Stdlib
//
//  Created by Philip Turner on 2/26/23.
//

#ifndef metal_stdlib_h
#define metal_stdlib_h

#define EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, C_TYPE, AIR_TYPE) \
__attribute__((__overloadable__)) C_TYPE EXPR(C_TYPE data) \
  __asm("air." #EXPR "." #AIR_TYPE); \

#define EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, C_TYPE, AIR_TYPE) \
__attribute__((__overloadable__)) C_TYPE EXPR(C_TYPE data, ushort delta) \
  __asm("air." #EXPR "." #AIR_TYPE); \

#define BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x) { \
  return METAL_EXPR(x); \
} \

#define BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x, uint delta) { \
  return METAL_EXPR(x, ushort(delta)); \
} \

#define OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x, uint clustersize) \
  __attribute__((enable_if(clustersize == 4 || clustersize == 32, "Cluster size not supported."))) \
{ \
  if (clustersize == 4) { \
    return quad_##METAL_SUFFIX(x); \
  } else { \
    return simd_##METAL_SUFFIX(x); \
  } \
} \

#define OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x, int delta, uint clustersize) \
  __attribute__((enable_if(clustersize == 4 || clustersize == 32, "Cluster size not supported."))) \
{ \
  if (clustersize == 4) { \
    return quad_##METAL_SUFFIX(x, ushort(delta)); \
  } else { \
    return simd_##METAL_SUFFIX(x, ushort(delta)); \
  } \
} \

#if OPENCL_USE_SUBGROUP_EXTENDED_TYPES

#define EXPOSE_FUNCTION_I_SCALAR_ARGS_1(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, char, s.i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, short, s.i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, int, s.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, uchar, u.i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, ushort, u.i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, uint, u.i32) \

#define EXPOSE_FUNCTION_I_VECTOR_ARGS_1(EXPR, VEC_SIZE) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, char##VEC_SIZE, s.v##VEC_SIZE##i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, short##VEC_SIZE, s.v##VEC_SIZE##i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, int##VEC_SIZE, s.v##VEC_SIZE##i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, uchar##VEC_SIZE, u.v##VEC_SIZE##i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, ushort##VEC_SIZE, u.v##VEC_SIZE##i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, uint##VEC_SIZE, u.v##VEC_SIZE##i32) \

#define EXPOSE_FUNCTION_I_ARGS_1(EXPR) \
EXPOSE_FUNCTION_I_SCALAR_ARGS_1(EXPR) \
EXPOSE_FUNCTION_I_VECTOR_ARGS_1(EXPR,2) \
EXPOSE_FUNCTION_I_VECTOR_ARGS_1(EXPR,3) \
EXPOSE_FUNCTION_I_VECTOR_ARGS_1(EXPR,4) \

#define EXPOSE_FUNCTION_F_ARGS_1(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, float, f32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, float2, v2f32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, float3, v3f32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, float4, v4f32) \

#define EXPOSE_FUNCTION_SCALAR_ARGS_2(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, char, s.i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, short, s.i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, int, s.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, uchar, u.i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, ushort, u.i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, uint, u.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, float, f32) \

#define EXPOSE_FUNCTION_VECTOR_ARGS_2(EXPR, VEC_SIZE) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, char##VEC_SIZE, s.v##VEC_SIZE##i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, short##VEC_SIZE, s.v##VEC_SIZE##i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, int##VEC_SIZE, s.v##VEC_SIZE##i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, uchar##VEC_SIZE, u.v##VEC_SIZE##i8) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, ushort##VEC_SIZE, u.v##VEC_SIZE##i16) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, uint##VEC_SIZE, u.v##VEC_SIZE##i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, float##VEC_SIZE, v##VEC_SIZE##f32) \

#define EXPOSE_FUNCTION_ARGS_2(EXPR) \
EXPOSE_FUNCTION_SCALAR_ARGS_2(EXPR) \
EXPOSE_FUNCTION_VECTOR_ARGS_2(EXPR,2) \
EXPOSE_FUNCTION_VECTOR_ARGS_2(EXPR,3) \
EXPOSE_FUNCTION_VECTOR_ARGS_2(EXPR,4) \

#define BRIDGE_FUNCTION_I_SCALAR_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, char) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, uchar) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, short) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, ushort) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, int) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, uint) \

#define BRIDGE_FUNCTION_I_VECTOR_ARGS_1(METAL_EXPR, OPENCL_EXPR, VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, char##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, uchar##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, short##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, ushort##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, int##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, uint##VEC_SIZE) \

#define BRIDGE_FUNCTION_I_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_I_SCALAR_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_I_VECTOR_ARGS_1(METAL_EXPR, OPENCL_EXPR, 2) \
BRIDGE_FUNCTION_I_VECTOR_ARGS_1(METAL_EXPR, OPENCL_EXPR, 3) \
BRIDGE_FUNCTION_I_VECTOR_ARGS_1(METAL_EXPR, OPENCL_EXPR, 4) \

#define BRIDGE_FUNCTION_F_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, float) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, float2) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, float3) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, float4) \

#define BRIDGE_FUNCTION_SCALAR_ARGS_2(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, char) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, uchar) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, short) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, ushort) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, int) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, uint) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_VECTOR_ARGS_2(METAL_EXPR, OPENCL_EXPR, VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, char##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, uchar##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, short##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, ushort##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, int##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, uint##VEC_SIZE) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, float##VEC_SIZE) \

#define BRIDGE_FUNCTION_ARGS_2(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_SCALAR_ARGS_2(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_VECTOR_ARGS_2(METAL_EXPR, OPENCL_EXPR, 2) \
BRIDGE_FUNCTION_VECTOR_ARGS_2(METAL_EXPR, OPENCL_EXPR, 3) \
BRIDGE_FUNCTION_VECTOR_ARGS_2(METAL_EXPR, OPENCL_EXPR, 4) \

#define CLUSTERED_I_SCALAR_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, char) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, uchar) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, short) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, ushort) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, int) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, uint) \

#define CLUSTERED_I_VECTOR_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, char##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, uchar##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, short##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, ushort##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, int##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, uint##VEC_SIZE) \

#define BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
CLUSTERED_I_SCALAR_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
CLUSTERED_I_VECTOR_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, 2) \
CLUSTERED_I_VECTOR_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, 3) \
CLUSTERED_I_VECTOR_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, 4) \

#define BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, float) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, float2) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, float3) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, float4) \

#define CLUSTERED_SCALAR_ARGS_2(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, char) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, uchar) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, short) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, ushort) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, int) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, uint) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, float) \

#define CLUSTERED_VECTOR_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, char##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, uchar##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, short##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, ushort##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, int##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, uint##VEC_SIZE) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, float##VEC_SIZE) \

#define BRIDGE_FUNCTION_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR) \
CLUSTERED_SCALAR_ARGS_2(METAL_SUFFIX, OPENCL_EXPR) \
CLUSTERED_VECTOR_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, 2) \
CLUSTERED_VECTOR_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, 3) \
CLUSTERED_VECTOR_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, 4) \

#else // OPENCL_USE_SUBGROUP_EXTENDED_TYPES

#define EXPOSE_FUNCTION_I_ARGS_1(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, int, s.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, uint, u.i32) \

#define EXPOSE_FUNCTION_F_ARGS_1(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, float, f32) \

#define EXPOSE_FUNCTION_ARGS_2(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, int, s.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, uint, u.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, float, f32) \

#define BRIDGE_FUNCTION_I_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, int) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, uint) \

#define BRIDGE_FUNCTION_F_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_ARGS_2(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, int) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, uint) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, int) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, uint) \

#define BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, int) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, uint) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, float) \

#endif // OPENCL_USE_SUBGROUP_EXTENDED_TYPES

// Declarations

// `I` = integer reduction
// `F` = floating-point reduction
// `B` = boolean reduction

#define DECLARE_I_REDUCTION_BASE(METAL_OP) \
EXPOSE_FUNCTION_I_ARGS_1(quad_##METAL_OP) \
EXPOSE_FUNCTION_I_ARGS_1(simd_##METAL_OP) \

#define DECLARE_F_REDUCTION_BASE(METAL_OP) \
EXPOSE_FUNCTION_F_ARGS_1(quad_##METAL_OP) \
EXPOSE_FUNCTION_F_ARGS_1(simd_##METAL_OP) \

#define DECLARE_REDUCTION_BASE(METAL_OP) \
DECLARE_I_REDUCTION_BASE(METAL_OP) \
DECLARE_F_REDUCTION_BASE(METAL_OP) \

#define DECLARE_SHUFFLE_BASE(METAL_OP) \
EXPOSE_FUNCTION_ARGS_2(quad_##METAL_OP) \
EXPOSE_FUNCTION_ARGS_2(simd_##METAL_OP) \

DECLARE_REDUCTION_BASE(sum)
DECLARE_REDUCTION_BASE(prefix_inclusive_sum)
DECLARE_REDUCTION_BASE(prefix_exclusive_sum)
DECLARE_REDUCTION_BASE(min)
DECLARE_REDUCTION_BASE(max)

DECLARE_REDUCTION_BASE(product)
DECLARE_REDUCTION_BASE(prefix_inclusive_product)
DECLARE_REDUCTION_BASE(prefix_exclusive_product)
DECLARE_I_REDUCTION_BASE(and)
DECLARE_I_REDUCTION_BASE(or)
DECLARE_I_REDUCTION_BASE(xor)

DECLARE_SHUFFLE_BASE(broadcast)
DECLARE_REDUCTION_BASE(broadcast_first)

DECLARE_SHUFFLE_BASE(shuffle)
DECLARE_SHUFFLE_BASE(shuffle_xor)
DECLARE_SHUFFLE_BASE(shuffle_up)
DECLARE_SHUFFLE_BASE(shuffle_down)
DECLARE_SHUFFLE_BASE(shuffle_rotate_up)
DECLARE_SHUFFLE_BASE(shuffle_rotate_down)

#define DECLARE_I_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_I_ARGS_1(simd_##METAL_OP, sub_group_##OPENCL_OP) \

#define DECLARE_F_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_F_ARGS_1(simd_##METAL_OP, sub_group_##OPENCL_OP) \

#define DECLARE_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_I_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_F_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \

#define DECLARE_SHUFFLE_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_ARGS_2(simd_##METAL_OP, sub_group_##OPENCL_OP) \

DECLARE_REDUCTION_UNIFORM(sum, reduce_add)
DECLARE_REDUCTION_UNIFORM(prefix_inclusive_sum, scan_inclusive_add)
DECLARE_REDUCTION_UNIFORM(prefix_exclusive_sum, scan_exclusive_add)
DECLARE_REDUCTION_UNIFORM(min, reduce_min)
DECLARE_REDUCTION_UNIFORM(max, reduce_max)

DECLARE_SHUFFLE_UNIFORM(shuffle, shuffle)
DECLARE_SHUFFLE_UNIFORM(shuffle_xor, shuffle_xor)
DECLARE_SHUFFLE_UNIFORM(shuffle_up, shuffle_up)
DECLARE_SHUFFLE_UNIFORM(shuffle_down, shuffle_down)
DECLARE_SHUFFLE_UNIFORM(shuffle_rotate_down, rotate)

DECLARE_SHUFFLE_UNIFORM(broadcast, broadcast)
DECLARE_REDUCTION_UNIFORM(broadcast_first, broadcast_first)

#define DECLARE_I_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_I_ARGS_1(simd_##METAL_OP, sub_group_non_uniform_##OPENCL_OP) \

#define DECLARE_F_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_F_ARGS_1(simd_##METAL_OP, sub_group_non_uniform_##OPENCL_OP) \

#define DECLARE_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_I_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_F_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \

#define DECLARE_SHUFFLE_NON_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_ARGS_2(simd_##METAL_OP, sub_group_non_uniform_##OPENCL_OP) \

DECLARE_REDUCTION_NON_UNIFORM(sum, reduce_add)
DECLARE_REDUCTION_NON_UNIFORM(prefix_inclusive_sum, scan_inclusive_add)
DECLARE_REDUCTION_NON_UNIFORM(prefix_exclusive_sum, scan_exclusive_add)
DECLARE_REDUCTION_NON_UNIFORM(min, reduce_min)
DECLARE_REDUCTION_NON_UNIFORM(max, reduce_max)

DECLARE_REDUCTION_NON_UNIFORM(product, reduce_mul)
DECLARE_REDUCTION_NON_UNIFORM(prefix_inclusive_product, scan_inclusive_mul)
DECLARE_REDUCTION_NON_UNIFORM(prefix_exclusive_product, scan_exclusive_mul)
DECLARE_I_REDUCTION_NON_UNIFORM(and, reduce_and)
DECLARE_I_REDUCTION_NON_UNIFORM(or, reduce_or)
DECLARE_I_REDUCTION_NON_UNIFORM(xor, reduce_xor)

DECLARE_SHUFFLE_NON_UNIFORM(broadcast, broadcast)

#define DECLARE_I_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1(METAL_OP, sub_group_clustered_##OPENCL_OP) \

#define DECLARE_F_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1(METAL_OP, sub_group_clustered_##OPENCL_OP) \

#define DECLARE_B_REDUCTION_CLUSTERED(OP) \
__attribute__((__overloadable__)) \
int sub_group_non_uniform_reduce_logical_##OP(int predicate) { \
return simd_##OP(select(0, 1, predicate != 0)); \
} \
\
__attribute__((__overloadable__)) \
int sub_group_clustered_reduce_logical_##OP(int predicate, uint clustersize) \
  __attribute__((enable_if(clustersize == 4 || clustersize == 32, "Cluster size not supported."))) \
{ \
  int x = select(0, 1, predicate != 0); \
  if (clustersize == 4) { \
    return quad_##OP(x); \
  } else { \
    return simd_##OP(x); \
  } \
} \

#define DECLARE_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
DECLARE_I_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
DECLARE_F_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \

#define DECLARE_SHUFFLE_CLUSTERED(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_CLUSTERED_ARGS_2(METAL_OP, sub_group_clustered_##OPENCL_OP) \

DECLARE_REDUCTION_CLUSTERED(sum, reduce_add)
DECLARE_REDUCTION_CLUSTERED(min, reduce_min)
DECLARE_REDUCTION_CLUSTERED(max, reduce_max)

DECLARE_REDUCTION_CLUSTERED(product, reduce_mul)
DECLARE_I_REDUCTION_CLUSTERED(and, reduce_and)
DECLARE_I_REDUCTION_CLUSTERED(or, reduce_or)
DECLARE_I_REDUCTION_CLUSTERED(xor, reduce_xor)
DECLARE_B_REDUCTION_CLUSTERED(and)
DECLARE_B_REDUCTION_CLUSTERED(or)
DECLARE_B_REDUCTION_CLUSTERED(xor)

DECLARE_SHUFFLE_CLUSTERED(shuffle_rotate_down, rotate)

#define EXPOSE_BALLOT(FUNC_EXPR, IN_EXPR, OUT_EXPR, AIR_EXPR) \
__attribute__((__overloadable__)) OUT_EXPR FUNC_EXPR(IN_EXPR) \
  __asm("air." #FUNC_EXPR #AIR_EXPR); \

EXPOSE_BALLOT(simd_is_first, , bool, )
EXPOSE_BALLOT(simd_all, bool expr, bool, )
EXPOSE_BALLOT(simd_any, bool expr, bool, )
EXPOSE_BALLOT(simd_ballot, bool expr, ulong, .i64)
EXPOSE_BALLOT(simd_active_threads_mask, , ulong, .i64)

EXPOSE_BALLOT(quad_is_first, , bool, )
EXPOSE_BALLOT(quad_all, bool expr, bool, )
EXPOSE_BALLOT(quad_any, bool expr, bool, )
EXPOSE_BALLOT(quad_ballot, bool expr, ushort, )
EXPOSE_BALLOT(quad_active_threads_mask, , ushort, )

int sub_group_elect() {
  return select(0, 1, simd_is_first());
}

int sub_group_all(int predicate) {
  return select(0, 1, simd_all(predicate != 0));
}

int sub_group_any(int predicate) {
  return select(0, 1, simd_any(predicate != 0));
}

int sub_group_non_uniform_all(int predicate) {
  return select(0, 1, simd_all(predicate != 0));
}

int sub_group_non_uniform_any(int predicate) {
  return select(0, 1, simd_any(predicate != 0));
}

uint4 sub_group_ballot(int predicate) {
  uint4 output = uint4(0);
  output.x = simd_ballot(predicate != 0);
  return output;
}

#undef EXPOSE_FUNCTION_OVERLOAD_ARGS_1
#undef EXPOSE_FUNCTION_OVERLOAD_ARGS_2
#undef BRIDGE_FUNCTION_OVERLOAD_ARGS_1
#undef BRIDGE_FUNCTION_OVERLOAD_ARGS_2
#undef OVERLOAD_CLUSTERED_ARGS_1
#undef OVERLOAD_CLUSTERED_ARGS_2

#undef EXPOSE_FUNCTION_I_SCALAR_ARGS_1
#undef EXPOSE_FUNCTION_I_VECTOR_ARGS_1
#undef EXPOSE_FUNCTION_I_ARGS_1
#undef EXPOSE_FUNCTION_F_ARGS_1
#undef EXPOSE_FUNCTION_SCALAR_ARGS_2
#undef EXPOSE_FUNCTION_VECTOR_ARGS_2
#undef EXPOSE_FUNCTION_ARGS_2

#undef BRIDGE_FUNCTION_I_SCALAR_ARGS_1
#undef BRIDGE_FUNCTION_I_VECTOR_ARGS_1
#undef BRIDGE_FUNCTION_I_ARGS_1
#undef BRIDGE_FUNCTION_F_ARGS_1
#undef BRIDGE_FUNCTION_SCALAR_ARGS_2
#undef BRIDGE_FUNCTION_VECTOR_ARGS_2
#undef BRIDGE_FUNCTION_ARGS_2

#undef CLUSTERED_I_SCALAR_ARGS_1
#undef CLUSTERED_I_VECTOR_ARGS_1
#undef BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1
#undef BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1
#undef CLUSTERED_SCALAR_ARGS_2
#undef CLUSTERED_VECTOR_ARGS_2
#undef BRIDGE_FUNCTION_CLUSTERED_ARGS_2

#undef DECLARE_I_REDUCTION_BASE
#undef DECLARE_F_REDUCTION_BASE
#undef DECLARE_REDUCTION_BASE
#undef DECLARE_SHUFFLE_BASE

#undef DECLARE_I_REDUCTION_UNIFORM
#undef DECLARE_F_REDUCTION_UNIFORM
#undef DECLARE_REDUCTION_UNIFORM
#undef DECLARE_SHUFFLE_UNIFORM

#undef DECLARE_I_REDUCTION_NON_UNIFORM
#undef DECLARE_F_REDUCTION_NON_UNIFORM
#undef DECLARE_REDUCTION_NON_UNIFORM
#undef DECLARE_SHUFFLE_NON_UNIFORM

#undef DECLARE_I_REDUCTION_CLUSTERED
#undef DECLARE_F_REDUCTION_CLUSTERED
#undef DECLARE_B_REDUCTION_CLUSTERED
#undef DECLARE_REDUCTION_CLUSTERED
#undef DECLARE_SHUFFLE_CLUSTERED

#endif /* metal_stdlib_h */
