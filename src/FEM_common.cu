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
		uint *nbr, FN_TYPE *vtxW, uint vw_pitch, FN_TYPE *vertex_weights, uint vv_pitch, uint vv_size, float4 *grads, uint he_pitch, uint *vertex_parts, uint *halo_vertices, uint hv_pitch, uint* halo_parts, uint *block_face_count, double dt) {

	uint size = vertex_parts[blockIdx.x+1] - vertex_parts[blockIdx.x];
	float3 *s_nvGrads = (float3 *)&s_mem[0];
	float3 *s_cvGrads = &s_nvGrads[size];
	FN_TYPE *s_wg = (FN_TYPE *)&s_cvGrads[size];
	FN_TYPE *s_nFn = &s_wg[size];
	FN_TYPE *s_cFn = &s_nFn[size+halo_parts[blockIdx.x]];
	for (int i = threadIdx.x; i < 7*size; i += blockDim.x) {
		s_mem[i] = 0.0;
	}
	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		s_nFn[i] = nFn_src[vertex_parts[blockIdx.x] + i];
		s_cFn[i] = cFn_src[vertex_parts[blockIdx.x] + i];
	}
	for (int i = threadIdx.x; i < halo_parts[blockIdx.x]; i += blockDim.x) {
		s_nFn[i + size] = nFn_src[halo_vertices[blockIdx.x*hv_pitch + i]];
		s_cFn[i + size] = cFn_src[halo_vertices[blockIdx.x*hv_pitch + i]];
	}


	__syncthreads();

	/* face gradients *************************************/
	if (threadIdx.x >= block_face_count[blockIdx.x])
		return;

	int fn_index[3] = {fv[blockIdx.x*3*fv_pitch + threadIdx.x], fv[blockIdx.x*3*fv_pitch + fv_pitch + threadIdx.x], fv[blockIdx.x*3*fv_pitch + 2*fv_pitch + threadIdx.x]};

	FN_TYPE nv1 = s_nFn[fn_index[2]];
	FN_TYPE nv12 = s_nFn[fn_index[0]] - nv1;
	FN_TYPE nv13 = s_nFn[fn_index[1]] - nv1;
	FN_TYPE cv1 = s_cFn[fn_index[2]];
	FN_TYPE cv12 = s_cFn[fn_index[0]] - cv1;
	FN_TYPE cv13 = s_cFn[fn_index[1]] - cv1;

	float3 grad12 = make_float3(grads[blockIdx.x*2*he_pitch + threadIdx.x]);
	float3 grad13 = make_float3(grads[blockIdx.x*2*he_pitch + he_pitch + threadIdx.x]);

	for (int j = 0; j < 3; j++) {
		if (fn_index[j] >= 0 && fn_index[j] < size) {
			float3 nvGrad = (grad12 * nv12 + grad13 * nv13)*fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x];
			atomicAdd(&s_nvGrads[fn_index[j]].x, nvGrad.x);
			atomicAdd(&s_nvGrads[fn_index[j]].y, nvGrad.y);
			atomicAdd(&s_nvGrads[fn_index[j]].z, nvGrad.z);

			float3 cvGrad = (grad12 * cv12 + grad13 * cv13)*fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x];
			atomicAdd(&s_cvGrads[fn_index[j]].x, cvGrad.x);
			atomicAdd(&s_cvGrads[fn_index[j]].y, cvGrad.y);
			atomicAdd(&s_cvGrads[fn_index[j]].z, cvGrad.z);

			atomicAdd(&s_wg[fn_index[j]], fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x]);
		}
	}

	__syncthreads();

	// Kill unnecessary threads
	if (vertex_parts[blockIdx.x] + threadIdx.x >= vertex_parts[blockIdx.x + 1])
		return;

	/* vertex gradients ***********************************/

	FN_TYPE dotP = dot(s_nvGrads[threadIdx.x] / s_wg[threadIdx.x], s_cvGrads[threadIdx.x] / s_wg[threadIdx.x]);
	if (s_wg[threadIdx.x] <= 0)
		dotP = 0;

	/* laplacian ******************************************/
	FN_TYPE vW = vtxW[blockIdx.x*vw_pitch + threadIdx.x];
	FN_TYPE n = s_nFn[threadIdx.x] * vW;
	FN_TYPE c = s_cFn[threadIdx.x] * vW;

	int end = nbr[blockIdx.x*(vv_size+1)*vv_pitch+threadIdx.x];
	for (int j = 0; j < end; j++) {
		int nIdx = nbr[blockIdx.x*(vv_size+1)*vv_pitch+vv_pitch*(j+1) + threadIdx.x];
		FN_TYPE hW = vertex_weights[blockIdx.x*vv_size*vv_pitch+vv_pitch*j + threadIdx.x];
		n += s_nFn[nIdx] * hW;
		c += s_cFn[nIdx] * hW;
	}


	/* update *********************************************/
	FN_TYPE dauN = D * n - alpha * s_nFn[threadIdx.x] * c - alpha * dotP
			+ S * r * s_nFn[threadIdx.x] * (nMax - s_nFn[threadIdx.x]);
	FN_TYPE dauC = c + S * (s_nFn[threadIdx.x] / (1 + s_nFn[threadIdx.x]) - s_cFn[threadIdx.x]);

	nFn_dst[vertex_parts[blockIdx.x] + threadIdx.x] = dt * dauN + s_nFn[threadIdx.x];
	cFn_dst[vertex_parts[blockIdx.x] + threadIdx.x] = dt * dauC + s_cFn[threadIdx.x];
}

extern "C" void step(FN_TYPE *nFn_src, FN_TYPE *cFn_src, FN_TYPE *nFn_dst,
		FN_TYPE *cFn_dst, uint *fv, FN_TYPE *fv_weights, uint fv_pitchInBytes,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitchInBytes, FN_TYPE *vertex_weights, uint vv_pitchInBytes, uint vv_size, float4 *grads, uint he_pitchInBytes,
		uint *parts_n, uint *halo_vertices, uint hv_pitchInBytes, uint *halo_parts, uint *block_face_count,
		uint blocks, uint threads, double dt, uint smem_size) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block, smem_size>>>(nFn_src, cFn_src, nFn_dst, cFn_dst,
			fv, fv_weights, fv_pitchInBytes/sizeof(uint), nbr, vtxW, vw_pitchInBytes/sizeof(uint), vertex_weights, vv_pitchInBytes/sizeof(uint), vv_size, grads, he_pitchInBytes/sizeof(float4), parts_n, halo_vertices, hv_pitchInBytes/sizeof(uint), halo_parts, block_face_count, dt);

}
