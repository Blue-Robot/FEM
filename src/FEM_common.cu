#include "FEM_common.h"
#include <stdio.h>


const FN_TYPE nMax  = 1;

const FN_TYPE D     = 0.25;
const FN_TYPE r     = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S     = 1;

__global__ void computeLaplacianKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint vertices);
__global__ void computeFaceGradientsKernel(uint *fv, FN_TYPE *nFn, FN_TYPE *cFn, float3 *grads, float3 *nfGrads, float3 *cfGrads, uint faces);
__global__ void computeVertexGradientsKernel(float3 *nfGrads, float3 *cfGrads, float3 *nvGrads, float3 *cvGrads, uint *t, uint *faces, FN_TYPE *fW, uint vertices);
__global__ void updateKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, float3 *nVtxGrad, float3 *cVtxGrad, double dt, uint vertices);

extern "C" void computeLaplacian(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint vertices, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)vertices/threads), 1, 1);
	computeLaplacianKernel<<<grid, block>>>(nFn, cFn, nLap, cLap, t, nbr, vtxW, heW, vertices);
}

__global__ void computeLaplacianKernel(FN_TYPE *nFn, FN_TYPE *cFn, FN_TYPE *nLap, FN_TYPE *cLap, uint *t, uint *nbr, FN_TYPE *vtxW, FN_TYPE *heW, uint vertices){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= vertices) return;

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

extern "C" void computeFaceGradients(uint *fv, FN_TYPE *nFn, FN_TYPE *cFn, float3 *grads, float3 *nfGrads, float3 *cfGrads, uint faces, uint threads) {
	dim3 block(threads, 1, 1);
	dim3 grid(ceil((double)faces/threads), 1, 1);
	computeFaceGradientsKernel<<<grid, block>>>(fv, nFn, cFn, grads, nfGrads, cfGrads, faces);
}

__global__ void computeFaceGradientsKernel(uint *fv, FN_TYPE *nFn, FN_TYPE *cFn, float3 *grads, float3 *nfGrads, float3 *cfGrads, uint faces) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= faces) return;

	FN_TYPE nv1 = nFn[fv[i*3+2]];
	FN_TYPE nv2 = nFn[fv[i*3]];
	FN_TYPE nv3 = nFn[fv[i*3+1]];
	FN_TYPE cv1 = cFn[fv[i*3+2]];
	FN_TYPE cv2 = cFn[fv[i*3]];
	FN_TYPE cv3 = cFn[fv[i*3+1]];

	float3 grad12 = grads[i*2];
	float3 grad13 = grads[i*2+1];

	nfGrads[i] = make_float3(
			grad12.x*(nv2-nv1) + grad13.x*(nv3-nv1),
			grad12.y*(nv2-nv1) + grad13.y*(nv3-nv1),
			grad12.z*(nv2-nv1) + grad13.z*(nv3-nv1)
	);
	cfGrads[i] = make_float3(
			grad12.x*(cv2-cv1) + grad13.x*(cv3-cv1),
			grad12.y*(cv2-cv1) + grad13.y*(cv3-cv1),
			grad12.z*(cv2-cv1) + grad13.z*(cv3-cv1)
	);
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

	int end = t[i+1];
	float3 nf;
	float3 cf;
	FN_TYPE w;
	FN_TYPE wg = 0;
	for (int j = t[i]; j < end; j++) {
		nf = nfGrads[faces[j]];
		cf = cfGrads[faces[j]];
		w = fW[j];
		ng = make_float3(ng.x + w*nf.x, ng.y + w*nf.y, ng.z + w*nf.z);
		cg = make_float3(cg.x + w*cf.x, cg.y + w*cf.y, cg.z + w*cf.z);
		wg += w;
	}
	if (wg > 0) {
		nvGrads[i] = make_float3(ng.x/wg, ng.y/wg, ng.z/wg);
		cvGrads[i] = make_float3(cg.x/wg, cg.y/wg, cg.z/wg);
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
	FN_TYPE dot = nVG.x*cVG.x + nVG.y*cVG.y + nVG.z*cVG.z;

	FN_TYPE dauN = D*nLap[i] - alpha*nFn[i]*cLap[i] - alpha*dot + S*r*nFn[i]*(nMax - nFn[i]);
	FN_TYPE dauC = cLap[i] + S*(nFn[i]/(1+nFn[i]) - cFn[i]);

	nFn[i] += dt*dauN;
	cFn[i] += dt*dauC;
}
