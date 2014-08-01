#include "common.h"

extern "C" void step (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		uint *face_vertices,
		uint fv_pitchInBytes,
		uint *nbr,
		FN_TYPE *vertex_weights,
		uint vv_pitchInBytes,
		FN_TYPE *vtxW,
		float3 *he_grads,
		uint he_pitchInBytes,
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
