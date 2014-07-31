#include "common.h"

extern "C" void step (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		uint *fv,
		uint *nbr,
		FN_TYPE *vertex_weights,
		uint vv_pitchInBytes,
		FN_TYPE *vtxW,
		float3 *grads,
		float3 *nfGrads,
		float3 *cfGrads,
		uint *vertex_faces,
		FN_TYPE *face_weights,
		uint vf_pitchInBytes,
		uint *parts_n,
		uint *parts_e,
		uint *halo_faces,
		uint hf_pitchInBytes,
		uint blocks,
		uint threads,
		double dt
		);
