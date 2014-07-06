#include "common.h"

extern "C" void computeLaplacian(
		FN_TYPE *nFn,
		FN_TYPE *cFn,
		FN_TYPE *nLap,
		FN_TYPE *cLap,
		uint *t,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *heW,
		uint* parts,
		uint vertices,
		uint blocks,
		uint threads
		);


extern "C" void computeFaceGradients(
		uint *fv,
		FN_TYPE *nFn,
		FN_TYPE *cFn,
		float3 *grads,
		float3 *nfGrads,
		float3 *cfGrads,
		uint faces,
		uint threads
		);


extern "C" void computeVertexGradients(
		float3 *nfGrads,
		float3 *cfGrads,
		float3 *nvGrads,
		float3 *cvGrads,
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
