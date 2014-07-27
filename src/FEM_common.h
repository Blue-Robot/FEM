#include "common.h"

extern "C" void step (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		uint *fv,
		uint *t,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *heW,
		float3 *grads,
		float3 *nfGrads,
		float3 *cfGrads,
		uint *f,
		uint *faces,
		FN_TYPE *fW,
		uint *parts_n,
		uint *parts_e,
		uint *halo_faces,
		uint *halo_faces_keys,
		uint blocks,
		uint threads,
		double dt
		);
