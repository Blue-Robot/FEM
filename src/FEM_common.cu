#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"

const FN_TYPE nMax = 1;
const FN_TYPE D = 0.25;
const FN_TYPE r = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S = 1;

__global__ void stepKernel(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, uint *fv, FN_TYPE *fv_weights, uint *t,
		uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, float3 *grads, float3 *nvGrads,
		float3 *cvGrads, FN_TYPE *wg, uint *f, uint *faces,
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
	int first = fv[i * 3];
	int second = fv[i * 3 + 1];
	int third = fv[i * 3 + 2];

	FN_TYPE nv1 = nFn_src[third];
	FN_TYPE nv12 = nFn_src[first] - nv1;
	FN_TYPE nv13 = nFn_src[second] - nv1;
	FN_TYPE cv1 = cFn_src[third];
	FN_TYPE cv12 = cFn_src[first] - cv1;
	FN_TYPE cv13 = cFn_src[second] - cv1;

	float3 grad12 = grads[i * 2];
	float3 grad13 = grads[i * 2 + 1];

	float3 nvGrad;
	float3 cvGrad;
	if (first >= vertex_parts[blockIdx.x] && first < vertex_parts[blockIdx.x+1]) {
	nvGrad = (grad12 * nv12 + grad13 * nv13)*fv_weights[i * 3];
	atomicAdd(&nvGrads[first].x, nvGrad.x);
	atomicAdd(&nvGrads[first].y, nvGrad.y);
	atomicAdd(&nvGrads[first].z, nvGrad.z);


	cvGrad = (grad12 * cv12 + grad13 * cv13)*fv_weights[i * 3];
	atomicAdd(&cvGrads[first].x, cvGrad.x);
	atomicAdd(&cvGrads[first].y, cvGrad.y);
	atomicAdd(&cvGrads[first].z, cvGrad.z);

	atomicAdd(&wg[first], fv_weights[i * 3]);
	}

	if (second >= vertex_parts[blockIdx.x] && second < vertex_parts[blockIdx.x+1]) {
	nvGrad = (grad12 * nv12 + grad13 * nv13)*fv_weights[i * 3 + 1];
	atomicAdd(&nvGrads[second].x, nvGrad.x);
	atomicAdd(&nvGrads[second].y, nvGrad.y);
	atomicAdd(&nvGrads[second].z, nvGrad.z);

	cvGrad = (grad12 * cv12 + grad13 * cv13)*fv_weights[i * 3 + 1];
	atomicAdd(&cvGrads[second].x, cvGrad.x);
	atomicAdd(&cvGrads[second].y, cvGrad.y);
	atomicAdd(&cvGrads[second].z, cvGrad.z);

	atomicAdd(&wg[second], fv_weights[i * 3 + 1]);
	}


	if (third >= vertex_parts[blockIdx.x] && third < vertex_parts[blockIdx.x+1]) {
	nvGrad = (grad12 * nv12 + grad13 * nv13)*fv_weights[i * 3 + 2];
	atomicAdd(&nvGrads[third].x, nvGrad.x);
	atomicAdd(&nvGrads[third].y, nvGrad.y);
	atomicAdd(&nvGrads[third].z, nvGrad.z);

	cvGrad = (grad12 * cv12 + grad13 * cv13)*fv_weights[i * 3 + 2];
	atomicAdd(&cvGrads[third].x, cvGrad.x);
	atomicAdd(&cvGrads[third].y, cvGrad.y);
	atomicAdd(&cvGrads[third].z, cvGrad.z);

	atomicAdd(&wg[third], fv_weights[i * 3 + 2]);
	}


	__syncthreads();

	// Adjust i
	i = vertex_parts[blockIdx.x] + threadIdx.x;

	// Kill unnecessary threads
	if (i >= vertex_parts[blockIdx.x + 1])
		return;

	/* vertex gradients ***********************************/

	FN_TYPE dotP = dot(nvGrads[i] / wg[i], cvGrads[i] / wg[i]);
	if (wg[i] <= 0)
		dotP = 0;

	/* laplacian ******************************************/
	FN_TYPE vW = vtxW[i];
	FN_TYPE n = nFn_src[i] * vW;
	FN_TYPE c = cFn_src[i] * vW;

	int end = t[i + 1];
	for (int j = t[i]; j < end; j++) {
		int nIdx = nbr[j];
		FN_TYPE hW = heW[j];
		n += nFn_src[nIdx] * hW;
		c += cFn_src[nIdx] * hW;
	}


	/* update *********************************************/
	FN_TYPE dauN = D * n - alpha * nFn_src[i] * c - alpha * dotP
			+ S * r * nFn_src[i] * (nMax - nFn_src[i]);
	FN_TYPE dauC = c + S * (nFn_src[i] / (1 + nFn_src[i]) - cFn_src[i]);

	nFn_dst[i] = dt * dauN + nFn_src[i];
	cFn_dst[i] = dt * dauC + cFn_src[i];
}

extern "C" void step(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, uint *fv, FN_TYPE *fv_weights, uint *t,
		uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, float3 *grads, float3 *nvGrads,
		float3 *cvGrads, FN_TYPE *wg, uint *f, uint *faces,
		FN_TYPE *fW, uint *parts_n, uint *parts_e, uint *halo_faces,
		uint *halo_faces_keys, uint blocks, uint threads, double dt) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block>>>(nFn_src, cFn_src, nFn_dst, cFn_dst,
			fv, fv_weights, t, nbr, vtxW, heW, grads, nvGrads, cvGrads, wg, f,
			faces, fW, parts_n, parts_e, halo_faces, halo_faces_keys, dt);

}
