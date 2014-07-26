#include "common.h"

extern "C" void computeLaplacianAndFaceGradients (
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		FN_TYPE *nLap,
		FN_TYPE *cLap,
		uint *fv,
		uint *t,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *heW,
		float3 *grads,
		float3 *nfGrads,
		float3 *cfGrads,
		float3 *nvGrads,
		float3 *cvGrads,
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

extern "C" void computeLaplacian(
		FN_TYPE *nFn,
		FN_TYPE *cFn,
		FN_TYPE *nLap,
		FN_TYPE *cLap,
		uint *t,
		uint *nbr,
		FN_TYPE *vtxW,
		FN_TYPE *heW,
		uint *halo_vertices,
		uint *halo_vertices_keys,
		uint *parts,
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
		uint *halo_faces,
		uint *halo_faces_keys,
		uint *parts,
		uint faces,
		uint blocks,
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
		uint* parts,
		uint vertices,
		uint blocks,
		uint threads
		);



extern "C" void update(
		FN_TYPE *nFn_src,
		FN_TYPE *cFn_src,
		FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst,
		FN_TYPE *nLap,
		FN_TYPE *cLap,
		float3 *nVtxGrad,
		float3 *cVtxGrad,
		double dt,
		uint vertices,
		uint threads
		);
