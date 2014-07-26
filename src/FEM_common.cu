#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"

const FN_TYPE nMax = 1;
const FN_TYPE D = 0.25;
const FN_TYPE r = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S = 1;

__global__ void stepKernel(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, FN_TYPE *nLap, FN_TYPE *cLap, uint *fv, uint *t,
		uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, float3 *grads, float3 *nfGrads,
		float3 *cfGrads, float3 *nvGrads, float3 *cvGrads, uint *f, uint *faces,
		FN_TYPE *fW, uint *vertex_parts, uint *face_parts, uint *halo_faces,
		uint *halo_faces_keys, double dt) {

	/* face gradients *************************************/
	int i = face_parts[blockIdx.x] + threadIdx.x;

	if (i >= face_parts[blockIdx.x + 1]) {

		i = i - face_parts[blockIdx.x + 1] + halo_faces_keys[blockIdx.x];
		if (i >= halo_faces_keys[blockIdx.x + 1])
			return;
		i = halo_faces[i];
	}

	FN_TYPE nv1 = nFn_src[fv[i * 3 + 2]];
	FN_TYPE nv12 = nFn_src[fv[i * 3]] - nv1;
	FN_TYPE nv13 = nFn_src[fv[i * 3 + 1]] - nv1;
	FN_TYPE cv1 = cFn_src[fv[i * 3 + 2]];
	FN_TYPE cv12 = cFn_src[fv[i * 3]] - cv1;
	FN_TYPE cv13 = cFn_src[fv[i * 3 + 1]] - cv1;

	float3 grad12 = grads[i * 2];
	float3 grad13 = grads[i * 2 + 1];

	nfGrads[i] = grad12 * nv12 + grad13 * nv13;
	cfGrads[i] = grad12 * cv12 + grad13 * cv13;

	__syncthreads();

	// Adjust i
	i = vertex_parts[blockIdx.x] + threadIdx.x;

	// Kill unnecessary threads
	if (i >= vertex_parts[blockIdx.x + 1])
		return;

	/* vertex gradients ***********************************/
	float3 ng = make_float3(0.0f, 0.0f, 0.0f);
	float3 cg = make_float3(0.0f, 0.0f, 0.0f);
	FN_TYPE wg = 0;

	int end = f[i + 1];

	for (int j = f[i]; j < end; j++) {
		uint face = faces[j];
		FN_TYPE w = fW[j];
		ng += w * nfGrads[face];
		cg += w * cfGrads[face];
		wg += w;
	}
	if (wg > 0) {
		nvGrads[i] = ng / wg;
		cvGrads[i] = cg / wg;
	} else {
		nvGrads[i] = make_float3(0.0f, 0.0f, 0.0f);
		cvGrads[i] = make_float3(0.0f, 0.0f, 0.0f);
	}

	/* laplacian ******************************************/
	FN_TYPE vW = vtxW[i];
	FN_TYPE n = nFn_src[i] * vW;
	FN_TYPE c = cFn_src[i] * vW;

	end = t[i + 1];
	for (int j = t[i]; j < end; j++) {
		int nIdx = nbr[j];
		FN_TYPE hW = heW[j];
		n += nFn_src[nIdx] * hW;
		c += cFn_src[nIdx] * hW;
	}
	nLap[i] = n;
	cLap[i] = c;


	/* update *********************************************/
	float3 nVG = nvGrads[i];
	float3 cVG = cvGrads[i];
	FN_TYPE dotP = dot(nVG, cVG);

	FN_TYPE dauN = D * nLap[i] - alpha * nFn_src[i] * cLap[i] - alpha * dotP
			+ S * r * nFn_src[i] * (nMax - nFn_src[i]);
	FN_TYPE dauC = cLap[i] + S * (nFn_src[i] / (1 + nFn_src[i]) - cFn_src[i]);

	nFn_dst[i] = dt * dauN + nFn_src[i];
	cFn_dst[i] = dt * dauC + cFn_src[i];
}

extern "C" void step(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, FN_TYPE *nLap, FN_TYPE *cLap, uint *fv, uint *t,
		uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, float3 *grads, float3 *nfGrads,
		float3 *cfGrads, float3 *nvGrads, float3 *cvGrads, uint *f, uint *faces,
		FN_TYPE *fW, uint *parts_n, uint *parts_e, uint *halo_faces,
		uint *halo_faces_keys, uint blocks, uint threads, double dt) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block>>>(nFn_src, cFn_src, nFn_dst, cFn_dst, nLap, cLap,
			fv, t, nbr, vtxW, heW, grads, nfGrads, cfGrads, nvGrads, cvGrads, f,
			faces, fW, parts_n, parts_e, halo_faces, halo_faces_keys, dt);

}
