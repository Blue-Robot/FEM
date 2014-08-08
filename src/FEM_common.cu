#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"

const FN_TYPE nMax = 1;
const FN_TYPE D = 0.25;
const FN_TYPE r = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S = 1;

__global__ void stepKernel(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, float4 *nv_grads, float4 *cv_grads, FN_TYPE *wg, uint *fv, FN_TYPE *fv_weights, uint fv_pitch,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitch, FN_TYPE *vertex_weights,
		uint vv_pitch, uint vv_size, float4 *grads, uint he_pitch,
		uint *vertex_parts, uint *block_face_count, double dt) {

	int i = vertex_parts[blockIdx.x] + threadIdx.x;
	nv_grads[i] = make_float4(0.0, 0.0, 0.0, 0.0);
	cv_grads[i] = make_float4(0.0, 0.0, 0.0, 0.0);
	wg[i] = 0.0;

	__syncthreads();

	/* face gradients *************************************/
	if (threadIdx.x >= block_face_count[blockIdx.x])
		return;

	int fn_index[3] = {fv[blockIdx.x*3*fv_pitch + threadIdx.x], fv[blockIdx.x*3*fv_pitch + fv_pitch + threadIdx.x], fv[blockIdx.x*3*fv_pitch + 2*fv_pitch + threadIdx.x]};

	FN_TYPE nv1 = nFn_src[fn_index[2]];
	FN_TYPE nv12 = nFn_src[fn_index[0]] - nv1;
	FN_TYPE nv13 = nFn_src[fn_index[1]] - nv1;
	FN_TYPE cv1 = cFn_src[fn_index[2]];
	FN_TYPE cv12 = cFn_src[fn_index[0]] - cv1;
	FN_TYPE cv13 = cFn_src[fn_index[1]] - cv1;


	float3 grad12 = make_float3(grads[blockIdx.x*2*he_pitch + threadIdx.x]);
	float3 grad13 = make_float3(grads[blockIdx.x*2*he_pitch + he_pitch + threadIdx.x]);



	for (int j = 0; j < 3; j++) {
		if(fn_index[j] >= vertex_parts[blockIdx.x] && fn_index[j] < vertex_parts[blockIdx.x+1]) {
			float3 nvGrad = (grad12 * nv12 + grad13 * nv13)*fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x];
			atomicAdd(&nv_grads[fn_index[j]].x, nvGrad.x);
			atomicAdd(&nv_grads[fn_index[j]].y, nvGrad.y);
			atomicAdd(&nv_grads[fn_index[j]].z, nvGrad.z);

			float3 cvGrad = (grad12 * cv12 + grad13 * cv13)*fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x];
			atomicAdd(&cv_grads[fn_index[j]].x, cvGrad.x);
			atomicAdd(&cv_grads[fn_index[j]].y, cvGrad.y);
			atomicAdd(&cv_grads[fn_index[j]].z, cvGrad.z);

			atomicAdd(&wg[fn_index[j]], fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x]);
		}
	}

	__syncthreads();

	// Adjust i

	// Kill unnecessary threads
	if (i >= vertex_parts[blockIdx.x + 1])
		return;

	/* vertex gradients ***********************************/

	FN_TYPE dotP = dot(make_float3(nv_grads[i]) / wg[i], make_float3(cv_grads[i]) / wg[i]);
	if (wg[i] <= 0)
		dotP = 0;

	/* laplacian ******************************************/
	FN_TYPE vW = vtxW[blockIdx.x*vw_pitch + threadIdx.x];
	FN_TYPE n = nFn_src[i] * vW;
	FN_TYPE c = cFn_src[i] * vW;

	int end = nbr[blockIdx.x*(vv_size+1)*vv_pitch+threadIdx.x];
	for (int j = 0; j < end; j++) {
		int nIdx = nbr[blockIdx.x*(vv_size+1)*vv_pitch+vv_pitch*(j+1) + threadIdx.x];
		FN_TYPE hW = vertex_weights[blockIdx.x*vv_size*vv_pitch+vv_pitch*j + threadIdx.x];
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

extern "C" void step(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst, FN_TYPE *cFn_dst, float4 *nv_grads, float4 *cv_grads, FN_TYPE *wg,
		uint *fv, FN_TYPE *fv_weights, uint fv_pitchInBytes,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitchInBytes, FN_TYPE *vertex_weights, uint vv_pitchInBytes, uint vv_size, float4 *grads, uint he_pitchInBytes,
		uint *parts_n, uint *block_face_count,
		uint blocks, uint threads, double dt, uint smem_size) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block>>>(nFn_src, cFn_src, nFn_dst, cFn_dst, nv_grads, cv_grads, wg,
			fv, fv_weights, fv_pitchInBytes/sizeof(uint), nbr, vtxW, vw_pitchInBytes/sizeof(uint), vertex_weights, vv_pitchInBytes/sizeof(uint), vv_size, grads, he_pitchInBytes/sizeof(float4), parts_n, block_face_count, dt);

}
