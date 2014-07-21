#include "FEM_common.h"
#include <stdio.h>
#include "helper_math.h"


const FN_TYPE nMax  = 1;

const FN_TYPE D     = 0.25;
const FN_TYPE r     = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S     = 1;

__global__ void computeLaplacianKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint *parts, uint vertices);
__global__ void computeFaceGradientsKernel(uint *fv, FN_TYPE *nFn, FN_TYPE *cFn, float3 *grads, float3 *nfGrads, float3 *cfGrads, uint *halo_faces, uint *halo_faces_keys, uint *parts, uint faces);
__global__ void computeVertexGradientsKernel(float3 *nfGrads, float3 *cfGrads, float3 *nvGrads, float3 *cvGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices);
__global__ void updateKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, float3 *nVtxGrad, float3 *cVtxGrad, double dt, uint vertices);

extern "C" void computeLaplacian(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint* parts, uint vertices, uint blocks, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);
	computeLaplacianKernel<<<grid, block>>>(nFn, cFn, nLap, cLap, t, nbr, vtxW, heW, parts, vertices);
}

__global__ void computeLaplacianKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint *parts, uint vertices){
	int i = parts[blockIdx.x] + threadIdx.x;

	if (i >= parts[blockIdx.x+1]) return;

	FN_TYPE vW = vtxW[i];
	FN_TYPE n = nFn[i]*vW;
	FN_TYPE c = cFn[i]*vW;

	int end = t[i+1];
	for (int j = t[i]; j < end; j++) {
		int nIdx = nbr[j];
		FN_TYPE hW = heW[j];
		n += nFn[nIdx]*hW;
		c += cFn[nIdx]*hW;
	}
	nLap[i] = n;
	cLap[i] = c;
}

extern "C" void computeFaceGradients(uint *fv, FN_TYPE *nFn, FN_TYPE *cFn, float3 *grads, float3 *nfGrads, float3 *cfGrads, uint *halo_faces, uint *halo_faces_keys, uint *parts, uint faces, uint blocks, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(blocks, 1, 1);
	computeFaceGradientsKernel<<<grid, block>>>(fv, nFn, cFn, grads, nfGrads, cfGrads, halo_faces, halo_faces_keys, parts, faces);
}

__global__ void computeFaceGradientsKernel(uint *fv, FN_TYPE *nFn, FN_TYPE *cFn, float3 *grads, float3 *nfGrads, float3 *cfGrads, uint *halo_faces, uint *halo_faces_keys, uint *parts, uint faces) {
	int i = parts[blockIdx.x] + threadIdx.x;

	if (i >= parts[blockIdx.x+1]) {

		i = i - parts[blockIdx.x+1] + halo_faces_keys[blockIdx.x];
		if (i >= halo_faces_keys[blockIdx.x+1])
			return;
		i = halo_faces[i];
	}

	FN_TYPE nv1 = nFn[fv[i*3+2]];
	FN_TYPE nv12 = nFn[fv[i*3]] - nv1;
	FN_TYPE nv13 = nFn[fv[i*3+1]] - nv1;
	FN_TYPE cv1 = cFn[fv[i*3+2]];
	FN_TYPE cv12 = cFn[fv[i*3]] - cv1;
	FN_TYPE cv13 = cFn[fv[i*3+1]] - cv1;

	float3 grad12 = grads[i*2];
	float3 grad13 = grads[i*2+1];

	nfGrads[i] = grad12*nv12 + grad13*nv13;
	cfGrads[i] = grad12*cv12 + grad13*cv13;
}

extern "C" void computeVertexGradients(float3 *nfGrads, float3 *cfGrads, float3 *nvGrads, float3 *cvGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)vertices/threads), 1, 1);
	computeVertexGradientsKernel<<<grid, block>>>(nfGrads, cfGrads, nvGrads, cvGrads, t, faces, fW, vertices);
}

__global__ void computeVertexGradientsKernel(float3 *nfGrads, float3 *cfGrads, float3 *nvGrads, float3 *cvGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= vertices) return;

	float3 ng = make_float3(0.0f, 0.0f, 0.0f);
	float3 cg = make_float3(0.0f, 0.0f, 0.0f);
	FN_TYPE wg = 0;

	int end = t[i+1];
	for (int j = t[i]; j < end; j++) {
		uint face = faces[j];
		FN_TYPE w = fW[j];
		ng += w*nfGrads[face];
		cg += w*cfGrads[face];
		wg += w;
	}
	if (wg > 0) {
		nvGrads[i] = ng/wg;
		cvGrads[i] = cg/wg;
	} else {
		nvGrads[i] = make_float3(0.0f, 0.0f, 0.0f);
		cvGrads[i] = make_float3(0.0f, 0.0f, 0.0f);
	}
}

extern "C" void update(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, float3 *nVtxGrad, float3 *cVtxGrad, double dt, uint vertices, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)vertices/threads), 1, 1);
	updateKernel<<<grid, block>>>(nFn, cFn, nLap, cLap, nVtxGrad, cVtxGrad, dt, vertices);
}

__global__ void updateKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, float3 *nVtxGrad, float3 *cVtxGrad, double dt, uint vertices) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= vertices) return;

	float3 nVG = nVtxGrad[i];
	float3 cVG = cVtxGrad[i];
	FN_TYPE dotP = dot(nVG, cVG);

	FN_TYPE dauN = D*nLap[i] - alpha*nFn[i]*cLap[i] - alpha*dotP + S*r*nFn[i]*(nMax - nFn[i]);
	FN_TYPE dauC = cLap[i] + S*(nFn[i]/(1+nFn[i]) - cFn[i]);

	nFn[i] += dt*dauN;
	cFn[i] += dt*dauC;
}
