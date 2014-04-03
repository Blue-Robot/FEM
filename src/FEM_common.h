#include "common.h"

extern "C" void computeLaplacian(
		FN_TYPE *nFn,
		FN_TYPE *nLap,
		uint *t,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *heW,
		uint vertices,
		uint threads
		);


extern "C" void computeFaceGradients(
		uint *fv,
		FN_TYPE *fn,
		float3 *grads,
		float3 *fGrads,
		uint faces,
		uint threads
		);


extern "C" void computeVertexGradients(
		float3 *fGrads,
		float3 *vGrads,
		uint *t,
		uint *faces,
		FN_TYPE *fW,
		uint vertices,
		uint threads
		);


extern "C" void update(
		FN_TYPE *nFn,
		FN_TYPE *cFn,
		FN_TYPE *nLap,
		FN_TYPE *cLap,
		float3 *nVtxGrad,
		float3 *cVtxGrad,
		double dt,
		uint vertices,
		uint threads
		);
