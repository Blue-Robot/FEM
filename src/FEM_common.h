#include "common.h"

extern "C" void step (
		float2 *fn_src,
		float2 *fn_dst,
		uint *dev_halo_sizes,
		uint *halo_vertices,
		uint hv_pitchInBytes,
		uint *fv,
		FN_TYPE *fv_weights,
		uint fv_pitchInBytes,
		uint *nbr,
		FN_TYPE *vtxW,
		uint vw_pitchInBytes,
		FN_TYPE *vertex_weights,
		uint vv_pitchInBytes,
		uint vv_size,
		float4 *grads,
		uint he_pitchInBytes,
		uint *parts_n,
		uint *block_face_count,
		uint blocks,
		uint threads,
		double dt,
		uint smem_size
		);
