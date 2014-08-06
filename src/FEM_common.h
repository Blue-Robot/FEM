#include "common.h"

extern "C" void step (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		uint *fv,
		FN_TYPE *fv_weights,
		uint fv_pitchInBytes,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *vertex_weights,
		uint vv_pitchInBytes,
		uint vv_size,
		float4 *grads,
		uint he_pitchInBytes,
		uint *parts_n,
		uint *parts_e,
		uint *halo_faces,
		uint *halo_faces_keys,
		uint blocks,
		uint threads,
		double dt,
		uint smem_size
		);
