#include "FEM_common.h"
#include <stdio.h>


const FN_TYPE nMax  = 1;

const FN_TYPE D     = 0.25;
const FN_TYPE r     = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S     = 1;

__global__ void computeLaplacianKernel(FN_TYPE *nFn, FN_TYPE *nLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint vertices);
__global__ void computeFaceGradientsKernel(uint *fv, FN_TYPE *fn, float3 *grads, float3 *fGrads, uint faces);
__global__ void computeVertexGradientsKernel(float3 *fGrads, float3 *vGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices);
__global__ void updateKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, float3 *nVtxGrad, float3 *cVtxGrad, double dt, uint vertices);

extern "C" void computeLaplacian(FN_TYPE *nFn, FN_TYPE *nLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint vertices, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)vertices/threads), 1, 1);
	computeLaplacianKernel<<<grid, block>>>(nFn, nLap, t, nbr, vtxW, heW, vertices);
}

__global__ void computeLaplacianKernel(FN_TYPE *nFn, FN_TYPE *nLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint vertices){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= vertices) return;

	FN_TYPE vW = vtxW[i];
	FN_TYPE n = nFn[i]*vW;

	int end = t[i+1];
	for (int j = t[i]; j < end; j++) {
		int nIdx = nbr[j];
		FN_TYPE hW = heW[j];
		n += nFn[nIdx]*hW;
	}
	nLap[i] = n;
}

extern "C" void computeFaceGradients(uint *fv, FN_TYPE *fn, float3 *grads, float3 *fGrads, uint faces, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)faces/threads), 1, 1);
	computeFaceGradientsKernel<<<grid, block>>>(fv, fn, grads, fGrads, faces);
}

__global__ void computeFaceGradientsKernel(uint *fv, FN_TYPE *fn, float3 *grads, float3 *fGrads, uint faces) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= faces) return;

	FN_TYPE v1 = fn[fv[i*3+2]];
	FN_TYPE v2 = fn[fv[i*3]];
	FN_TYPE v3 = fn[fv[i*3+1]];

	float3 grad12 = grads[i*2];
	float3 grad13 = grads[i*2+1];

	float3 fGrad = make_float3(
			grad12.x*(v2-v1) + grad13.x*(v3-v1),
			grad12.y*(v2-v1) + grad13.y*(v3-v1),
			grad12.z*(v2-v1) + grad13.z*(v3-v1)
	);
	fGrads[i] = fGrad;
}

extern "C" void computeVertexGradients(float3 *fGrads, float3 *vGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)vertices/threads), 1, 1);
	computeVertexGradientsKernel<<<grid, block>>>(fGrads, vGrads, t, faces, fW, vertices);
}

__global__ void computeVertexGradientsKernel(float3 *fGrads, float3 *vGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= vertices) return;

	float3 g = make_float3(0.0f, 0.0f, 0.0f);

	int end = t[i+1];
	float3 f;
	FN_TYPE w;
	FN_TYPE wg = 0;
	for (int j = t[i]; j < end; j++) {
		f = fGrads[faces[j]];
		w = fW[j];
		g = make_float3(g.x + w*f.x, g.y + w*f.y, g.z + w*f.z);
		wg += w;
	}
	if (wg > 0) {
		vGrads[i] = make_float3(g.x/wg, g.y/wg, g.z/wg);
	} else {
		vGrads[i] = make_float3(0.0f, 0.0f, 0.0f);
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
	FN_TYPE dot = nVG.x*cVG.x + nVG.y*cVG.y + nVG.z*cVG.z;

	FN_TYPE dauN = D*nLap[i] - alpha*nFn[i]*cLap[i] - alpha*dot + S*r*nFn[i]*(nMax - nFn[i]);
	FN_TYPE dauC = cLap[i] + S*(nFn[i]/(1+nFn[i]) - cFn[i]);

	nFn[i] += dt*dauN;
	cFn[i] += dt*dauC;
}
