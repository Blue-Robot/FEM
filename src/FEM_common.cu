#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"

const FN_TYPE nMax = 1;
const FN_TYPE D = 0.25;
const FN_TYPE r = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S = 1;

extern __shared__ FN_TYPE s_mem[];
__global__ void stepKernel(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, uint *fv, FN_TYPE *fv_weights, uint fv_pitch,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitch, FN_TYPE *vertex_weights, uint vv_pitch, uint vv_size, float4 *grads, uint he_pitch, uint *vertex_parts, uint *block_face_count, double dt) {

	uint size = vertex_parts[blockIdx.x+1] - vertex_parts[blockIdx.x];
	float3 *s_nvGrads = (float3 *)&s_mem[0];
	float3 *s_cvGrads = &s_nvGrads[size];
	FN_TYPE *s_wg = (FN_TYPE *)&s_cvGrads[size];
	for (int i = threadIdx.x; i < 7*size; i += blockDim.x) {
		s_mem[i] = 0.0;
	}

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

	fn_index[0] -= vertex_parts[blockIdx.x];
	fn_index[1] -= vertex_parts[blockIdx.x];
	fn_index[2] -= vertex_parts[blockIdx.x];

	float3 grad12 = make_float3(grads[blockIdx.x*2*he_pitch + threadIdx.x]);
	float3 grad13 = make_float3(grads[blockIdx.x*2*he_pitch + he_pitch + threadIdx.x]);

	float3 nvGrad = grad12 * nv12 + grad13 * nv13;
	float3 cvGrad = grad12 * cv12 + grad13 * cv13;


	FN_TYPE weight = fv_weights[blockIdx.x*3*fv_pitch + threadIdx.x];

	for (int i = 0; i < 3; i++) {
		atomicAdd(&s_nvGrads[fn_index[0]].x+i, *(&nvGrad.x+i)*weight);
		atomicAdd(&s_cvGrads[fn_index[0]].x+i, *(&cvGrad.x+i)*weight);
	}
	atomicAdd(&s_wg[fn_index[0]], weight);


	if (fn_index[1] < 0 || fn_index[1] >= size)
		return;

	weight = fv_weights[blockIdx.x*3*fv_pitch + fv_pitch + threadIdx.x];

	atomicAdd(&s_nvGrads[fn_index[1]].x+0, *(&nvGrad.x+0)*weight);
	atomicAdd(&s_nvGrads[fn_index[1]].x+1, *(&nvGrad.x+1)*weight);
	atomicAdd(&s_nvGrads[fn_index[1]].x+2, *(&nvGrad.x+2)*weight);

	atomicAdd(&s_cvGrads[fn_index[1]].x+0, *(&cvGrad.x+0)*weight);
	atomicAdd(&s_cvGrads[fn_index[1]].x+1, *(&cvGrad.x+1)*weight);
	atomicAdd(&s_cvGrads[fn_index[1]].x+2, *(&cvGrad.x+2)*weight);

	atomicAdd(&s_wg[fn_index[1]], weight);

	if (fn_index[2] < 0 || fn_index[2] >= size)
		return;

	weight = fv_weights[blockIdx.x*3*fv_pitch + 2*fv_pitch + threadIdx.x];

	atomicAdd(&s_nvGrads[fn_index[2]].x+0, *(&nvGrad.x+0)*weight);
	atomicAdd(&s_nvGrads[fn_index[2]].x+1, *(&nvGrad.x+1)*weight);
	atomicAdd(&s_nvGrads[fn_index[2]].x+2, *(&nvGrad.x+2)*weight);

	atomicAdd(&s_cvGrads[fn_index[2]].x+0, *(&cvGrad.x+0)*weight);
	atomicAdd(&s_cvGrads[fn_index[2]].x+1, *(&cvGrad.x+1)*weight);
	atomicAdd(&s_cvGrads[fn_index[2]].x+2, *(&cvGrad.x+2)*weight);

	atomicAdd(&s_wg[fn_index[2]], weight);



	// Adjust i
	int i = vertex_parts[blockIdx.x] + threadIdx.x;

	// Kill unnecessary threads
	if (i >= vertex_parts[blockIdx.x + 1])
		return;

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

	/* vertex gradients ***********************************/
	__syncthreads();
	FN_TYPE dotP = dot(s_nvGrads[threadIdx.x] / s_wg[threadIdx.x], s_cvGrads[threadIdx.x] / s_wg[threadIdx.x]);
	if (s_wg[threadIdx.x] <= 0) {
		dotP = 0;
	}

	/* update *********************************************/
	FN_TYPE dauN = D * n - alpha * nFn_src[i] * c - alpha * dotP
			+ S * r * nFn_src[i] * (nMax - nFn_src[i]);
	FN_TYPE dauC = c + S * (nFn_src[i] / (1 + nFn_src[i]) - cFn_src[i]);

	nFn_dst[i] = dt * dauN + nFn_src[i];
	cFn_dst[i] = dt * dauC + cFn_src[i];
}

extern "C" void step(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, uint *fv, FN_TYPE *fv_weights, uint fv_pitchInBytes,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitchInBytes, FN_TYPE *vertex_weights, uint vv_pitchInBytes, uint vv_size, float4 *grads, uint he_pitchInBytes,
		uint *parts_n, uint *block_face_count,
		uint blocks, uint threads, double dt, uint smem_size) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block, smem_size>>>(nFn_src, cFn_src, nFn_dst, cFn_dst,
			fv, fv_weights, fv_pitchInBytes/sizeof(uint), nbr, vtxW, vw_pitchInBytes/sizeof(uint), vertex_weights, vv_pitchInBytes/sizeof(uint), vv_size, grads, he_pitchInBytes/sizeof(float4), parts_n, block_face_count, dt);

}
