#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"

// THRUST
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

const FN_TYPE nMax = 1;
const FN_TYPE D = 0.25;
const FN_TYPE r = 1.52201704740629;
const FN_TYPE alpha = 12.0228703901698;
const FN_TYPE S = 1;

extern __shared__ FN_TYPE s_mem[];
__global__ void stepKernel(float2 *fn_src, float2 *fn_dst,
		uint *fv, FN_TYPE *fv_weights, uint fv_pitch,
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

	float2 v1 = fn_src[fn_index[2]];
	float2 v12 = fn_src[fn_index[0]] - v1;
	float2 v13 = fn_src[fn_index[1]] - v1;

	fn_index[0] -= vertex_parts[blockIdx.x];
	fn_index[1] -= vertex_parts[blockIdx.x];
	fn_index[2] -= vertex_parts[blockIdx.x];

	float3 grad12 = make_float3(grads[blockIdx.x*2*he_pitch + threadIdx.x]);
	float3 grad13 = make_float3(grads[blockIdx.x*2*he_pitch + he_pitch + threadIdx.x]);

	float3 nvGrad = grad12 * v12.x + grad13 * v13.x;
	float3 cvGrad = grad12 * v12.y + grad13 * v13.y;

	for (int j = 0; j < 3; j++) {
		if (fn_index[j] >= 0 && fn_index[j] < size) {
			FN_TYPE weight = fv_weights[blockIdx.x*3*fv_pitch + j*fv_pitch + threadIdx.x];

			atomicAdd(&s_nvGrads[fn_index[j]].x, nvGrad.x*weight);
			atomicAdd(&s_nvGrads[fn_index[j]].y, nvGrad.y*weight);
			atomicAdd(&s_nvGrads[fn_index[j]].z, nvGrad.z*weight);

			atomicAdd(&s_cvGrads[fn_index[j]].x, cvGrad.x*weight);
			atomicAdd(&s_cvGrads[fn_index[j]].y, cvGrad.y*weight);
			atomicAdd(&s_cvGrads[fn_index[j]].z, cvGrad.z*weight);

			atomicAdd(&s_wg[fn_index[j]], weight);
		}
	}



	// Adjust i
	int i = vertex_parts[blockIdx.x] + threadIdx.x;

	// Kill unnecessary threads
	if (i >= vertex_parts[blockIdx.x + 1])
		return;

	/* laplacian ******************************************/
	double vW = vtxW[blockIdx.x*vw_pitch + threadIdx.x];
	float2 lap = fn_src[i] * vW;

	int end = nbr[blockIdx.x*(vv_size+1)*vv_pitch+threadIdx.x];
	for (int j = 0; j < end; j++) {
		int nIdx = nbr[blockIdx.x*(vv_size+1)*vv_pitch+vv_pitch*(j+1) + threadIdx.x];
		double hW = vertex_weights[blockIdx.x*vv_size*vv_pitch+vv_pitch*j + threadIdx.x];
		lap += fn_src[nIdx] * hW;
	}

	/* vertex gradients ***********************************/
	__syncthreads();
	double dotP = dot(s_nvGrads[threadIdx.x] / s_wg[threadIdx.x], s_cvGrads[threadIdx.x] / s_wg[threadIdx.x]);
	if (s_wg[threadIdx.x] <= 0) {
		dotP = 0;
	}

	/* update *********************************************/
	double dauN = D * lap.x - alpha * fn_src[i].x * lap.y - alpha * dotP
			+ S * r * fn_src[i].x * (nMax - fn_src[i].x);
	double dauC = lap.y + S * (fn_src[i].x / (1 + fn_src[i].x) - fn_src[i].y);

	fn_dst[i].x = dt * dauN + fn_src[i].x > 0 ? dt * dauN + fn_src[i].x : 0.0;
	fn_dst[i].y = dt * dauC + fn_src[i].y > 0 ? dt * dauC + fn_src[i].y : 0.0;
}

extern "C" void step(float2 *fn_src, float2 *fn_dst,
		uint *fv, FN_TYPE *fv_weights, uint fv_pitchInBytes,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitchInBytes, FN_TYPE *vertex_weights, uint vv_pitchInBytes, uint vv_size, float4 *grads, uint he_pitchInBytes,
		uint *parts_n, uint *block_face_count,
		uint blocks, uint threads, double dt, uint smem_size) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block, smem_size>>>(fn_src, fn_dst,
			fv, fv_weights, fv_pitchInBytes/sizeof(uint), nbr, vtxW, vw_pitchInBytes/sizeof(uint), vertex_weights, vv_pitchInBytes/sizeof(uint), vv_size, grads, he_pitchInBytes/sizeof(float4), parts_n, block_face_count, dt);

}
const float sigma = 0.2;

__global__ void formatKernel(float2 *fn, float4 *vbo, float2 *min, float2 *max, int offset) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	float n = (fn[i].x-(*min).x)/((*max).x-(*min).x);
	n = ((1.0f/(1.0f + expf(-((n-0.5f)/sigma)))) - (1.0f/(1.0f + expf(-((0.0f-0.5f)/sigma)))))/((1.0f/(1.0f + expf(-((1.0f-0.5f)/sigma)))) - (1.0f/(1.0f + expf(-((0.0f-0.5f)/sigma)))));
	//n = n < 0.5 ? pow(n,3.0f) : pow(n,1/3.0f);
	float c = 1-n;
	float b = 1-n;

	vbo[i + offset] = make_float4(n, c, b, 1.0);
//	if (i == 0)
//		printf("%f %f\n", (*max).x, (*min).x);
}

struct comp_x
{
  __host__ __device__
  bool operator()(float2 lhs, float2 rhs)
  {
    return lhs.x < rhs.x;
  }
};


extern "C" void format (float2 *fn, cudaGraphicsResource_t *vbo_res, int vertices) {
	thrust::device_ptr<float2> dptr(fn);
	thrust::device_ptr<float2> dresptrmax = thrust::max_element(dptr, dptr + vertices, comp_x());
	thrust::device_ptr<float2> dresptrmin = thrust::min_element(dptr, dptr + vertices, comp_x());

	float2 *max = raw_pointer_cast(dresptrmax);
	float2 *min = raw_pointer_cast(dresptrmin);

	float4 *vboptr;
	size_t num_bytes;

	cudaGraphicsMapResources(1, vbo_res, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&vboptr, &num_bytes, *vbo_res);

//	float2 *test = new float2[vertices];
//	cudaMemcpy(test, fn, vertices*sizeof(float2), cudaMemcpyDeviceToHost);
//	for(int i = 0; i < vertices; i++) {
//		printf("%f ", test[i].x);
//	}
//	printf("\n");

	formatKernel<<<vertices,1>>>(fn, vboptr, min, max, vertices);

	cudaGraphicsUnmapResources(1, vbo_res, 0);
}
