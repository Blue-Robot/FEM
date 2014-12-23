#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"

// THRUST
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

const FN_TYPE nMax = 1;
const FN_TYPE D = 0.25;
const FN_TYPE r = 59.453790914308100;
const FN_TYPE alpha = 269;
const FN_TYPE S = 1;

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

inline __host__ __device__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}

inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double3 make_double3(float3 a)
{
    return make_double3((double)a.x, (double)a.y, (double)a.z);
}

inline __host__ __device__ double3 make_double3(float4 a)
{
    return make_double3((double)a.x, (double)a.y, (double)a.z);
}




extern __shared__ FN_TYPE s_mem[];
__global__ void stepKernel(double2 *fn_src, double2 *fn_dst,
		uint *fv, FN_TYPE *fv_weights, uint fv_pitch, uint fvw_pitch,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitch, FN_TYPE *vertex_weights, uint vv_pitch, uint vws_pitch, uint vv_size, float4 *grads, uint he_pitch, uint *vertex_parts, uint *block_face_count, double dt) {

	uint size = vertex_parts[blockIdx.x+1] - vertex_parts[blockIdx.x];
	float3 *s_nvGrads = (float3 *)&s_mem[0];
	float3 *s_cvGrads = &s_nvGrads[size];
	FN_TYPE *s_wg = (FN_TYPE *)&s_cvGrads[size];
	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		s_nvGrads[i] = make_float3(0.0f,0.0f,0.0f);
		s_cvGrads[i] = make_float3(0.0f,0.0f,0.0f);
		s_wg[i] = 0.0;
	}

	__syncthreads();

	/* face gradients *************************************/
	if (threadIdx.x >= block_face_count[blockIdx.x])
		return;

	int fn_index[3] = {fv[blockIdx.x*3*fv_pitch + threadIdx.x], fv[blockIdx.x*3*fv_pitch + fv_pitch + threadIdx.x], fv[blockIdx.x*3*fv_pitch + 2*fv_pitch + threadIdx.x]};

	double2 v1 = fn_src[fn_index[2]];
	double2 v12 = fn_src[fn_index[0]] - v1;
	double2 v13 = fn_src[fn_index[1]] - v1;

	fn_index[0] -= vertex_parts[blockIdx.x];
	fn_index[1] -= vertex_parts[blockIdx.x];
	fn_index[2] -= vertex_parts[blockIdx.x];

	float3 grad12 = make_float3(grads[blockIdx.x*2*he_pitch + threadIdx.x]);
	float3 grad13 = make_float3(grads[blockIdx.x*2*he_pitch + he_pitch + threadIdx.x]);

	float3 nvGrad = grad12 * v12.x + grad13 * v13.x;
	float3 cvGrad = grad12 * v12.y + grad13 * v13.y;

	for (int j = 0; j < 3; j++) {
		if (fn_index[j] >= 0 && fn_index[j] < size) {
			FN_TYPE weight = fv_weights[blockIdx.x*3*fvw_pitch + j*fvw_pitch + threadIdx.x];

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
	double2 lap = fn_src[i] * vW;

	int end = nbr[blockIdx.x*(vv_size+1)*vv_pitch+threadIdx.x];
	for (int j = 0; j < end; j++) {
		int nIdx = nbr[blockIdx.x*(vv_size+1)*vv_pitch+vv_pitch*(j+1) + threadIdx.x];
		double hW = vertex_weights[blockIdx.x*vv_size*vws_pitch + vws_pitch*j + threadIdx.x];
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

extern "C" void step(double2 *fn_src, double2 *fn_dst,
		uint *fv, FN_TYPE *fv_weights, uint fv_pitchInBytes, uint fvw_pitchInBytes,
		uint *nbr, FN_TYPE *vtxW, uint vw_pitchInBytes, FN_TYPE *vertex_weights, uint vv_pitchInBytes, uint vws_pitchInBytes, uint vv_size, float4 *grads, uint he_pitchInBytes,
		uint *parts_n, uint *block_face_count,
		uint blocks, uint threads, double dt, uint smem_size) {

	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);

	stepKernel<<<grid, block, smem_size>>>(fn_src, fn_dst,
			fv, fv_weights, fv_pitchInBytes/sizeof(uint), fvw_pitchInBytes/sizeof(FN_TYPE), nbr, vtxW, vw_pitchInBytes/sizeof(FN_TYPE), vertex_weights, vv_pitchInBytes/sizeof(uint), vws_pitchInBytes/sizeof(FN_TYPE), vv_size, grads, he_pitchInBytes/sizeof(float4), parts_n, block_face_count, dt);

}
const double sigma = 0.2;

__global__ void formatKernel(double2 *fn, double4 *vbo, double2 *min, double2 *max, int offset) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	double n = (fn[i].x-(*min).x)/((*max).x-(*min).x);
	//n = ((1.0f/(1.0f + expf(-((n-0.5f)/sigma)))) - (1.0f/(1.0f + expf(-((0.0f-0.5f)/sigma)))))/((1.0f/(1.0f + expf(-((1.0f-0.5f)/sigma)))) - (1.0f/(1.0f + expf(-((0.0f-0.5f)/sigma)))));
	//n = n < 0.5 ? pow(n,3.0f) : pow(n,1/3.0f);
	double c = 1-n;
	double b = 1-n;


	vbo[i + offset] = make_double4(n, c, b, 1.0);
}

struct comp_x
{
  __host__ __device__
  bool operator()(double2 lhs, double2 rhs)
  {
    return lhs.x < rhs.x;
  }
};


extern "C" void format (double2 *fn, cudaGraphicsResource_t *vbo_res, int vertices) {
	thrust::device_ptr<double2> dptr(fn);
	thrust::device_ptr<double2> dresptrmax = thrust::max_element(dptr, dptr + vertices, comp_x());
	thrust::device_ptr<double2> dresptrmin = thrust::min_element(dptr, dptr + vertices, comp_x());

	double2 *max = raw_pointer_cast(dresptrmax);
	double2 *min = raw_pointer_cast(dresptrmin);

	double4 *vboptr;
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
