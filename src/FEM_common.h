#include "common.h"

extern "C" void step (
		double2 *fn_src,
		double2 *fn_dst,
		uint *fv,
		FN_TYPE *fv_weights,
		uint fv_pitchInBytes,
		uint fvw_pitchInBytes,
		uint *nbr,
		FN_TYPE *vtxW,
		uint vw_pitchInBytes,
		FN_TYPE *vertex_weights,
		uint vv_pitchInBytes,
		uint vws_pitchInBytes,
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

extern "C" void format (
		double2 *fn,
		cudaGraphicsResource_t *vbo_res,
		int vertices
		);
