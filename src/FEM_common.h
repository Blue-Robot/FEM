#include "common.h"

extern "C" void step (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		uint *fv,
		FN_TYPE *fv_weights,
		uint *t,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *heW,
		float3 *grads,
		uint *parts_n,
		uint *parts_e,
		uint *halo_faces,
		uint *halo_faces_keys,
		uint blocks,
		uint threads,
		double dt,
		uint smem_size
		);
