// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// C/C++ Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <string>
#include <math.h>
#include <list>

// OpenMesh Includes
#include <IO/MeshIO.hh>
#include <getopt.h>

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <rendercheck_gl.h>
#include <cuda_profiler_api.h>

#include <curand.h>

// File Includes
#include "common.h"
#include "mesh_function.h"
#include "FEM_common.h"
#include "partitioning_common.h"

// Model
const FN_TYPE nMax = 1;
const FN_TYPE betaMax = 0.01f;
const FN_TYPE D     = 0.25;
const FN_TYPE r     = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S     = 1;
double dt;

// General
string file_name;
bool verbose = false;
bool debug = false;

// OpenMesh
SimpleTriMesh originalMesh;
SimpleTriMesh orderedMesh;
MeshStats mStats;
uint numAngles;
uint numVtx;
uint numFaces;

// Host Device
FN_TYPE* nFn;
FN_TYPE* cFn;
FN_TYPE *heWeights;
//FN_TYPE *vtxWeights;
FN_TYPE *faceWeights;

uint *nbrTracker;
uint *neighbors;
uint *faceTracker;
uint *vertexFaces;
uint *faceVertices;
FN_TYPE *fv_weights;
float3 *gradients;

uint *node_parts;
uint *element_parts;

std::vector<uint> halo_faces;
std::vector<uint> halo_vertices;
uint *halo_faces_keys;
uint *halo_vertices_keys;

FN_TYPE *vertex_weights;
uint *nbr_v;
FN_TYPE *vv_weights;
uint max_vertex_count;
uint vv_max_neighbors;

uint *face_vertices;
FN_TYPE *fv_weights_new;
float4 *he_grads;
uint max_face_count;

uint *block_face_count;

//batch configuration
int start_n = 365; // number of partitions to start with
int end_n = 385; // number of partitions to end with

using namespace OpenMesh;

int main(int argc, char **argv);

void initializeCUDA();
void loadMesh();
void initializeData(int n);
void initializeGPUData(int n);

double GPUrun(int n);
void CPUrun(FN_TYPE *test_nFn, FN_TYPE *test_cFn, int num_steps);


int main(int argc, char **argv) {
	file_name = argv[1];
	if (argc >=3 && !strcmp(argv[2], "-d"))
		debug = true;

	if (!verbose)
		std::cout.rdbuf(NULL);

	initializeCUDA();
	loadMesh();

	double best_time;
	int best_n=-1;
	for (int i = start_n; i <= end_n; i++) {
		initializeData(i);
		initializeGPUData(i);

		double time = GPUrun(i);
		if ((best_n == -1 || best_time > time) && time > 0) {
			best_n = i;
			best_time = time;
		}

	}
	if (!debug)
		printf("Best time with %d partitions is %.1f!\n", best_n, best_time);

	return 0;
}
void loadMesh() {
	OpenMesh::IO::Options opt(OpenMesh::IO::Options::VertexColor | OpenMesh::IO::Options::VertexNormal);

	originalMesh.request_vertex_normals();
	OpenMesh::IO::read_mesh(originalMesh, file_name, opt);
}

void initializeData(int n) {
	partition(originalMesh, &orderedMesh, &node_parts, &element_parts, n);

	numAngles = orderedMesh.n_halfedges();
	numVtx = orderedMesh.n_vertices();
	numFaces = orderedMesh.n_faces();

	initMeshStats(orderedMesh, mStats);

	nFn = new FN_TYPE[numVtx];
	cFn = new FN_TYPE[numVtx];
	FN_TYPE* beta = new FN_TYPE[numVtx];

	for (unsigned int i = 0; i < numVtx; ++i) {
		beta[i] = -0.5 * betaMax + rand() * betaMax / RAND_MAX;
		nFn[i] = nMax - beta[i];
		cFn[i] = 1 / (1 + nFn[i]);
	}

	dt = mStats.maxEdgeLen * mStats.maxEdgeLen * 0.1;
}

void initializeCUDA() {
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaDeviceReset());
	checkCudaErrors(cudaSetDevice(gpuGetMaxGflopsDeviceId()));

	//checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
	//checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
}

void initializeGPUData(int n) {
	VertexHandle *mesh = new VertexHandle[numVtx];
	SimpleTriMesh::VertexIter vIter, vEnd(orderedMesh.vertices_end());

	for (vIter = orderedMesh.vertices_begin(); vIter != vEnd; ++vIter) {
		mesh[vIter.handle().idx()] = vIter.handle();

	}

	// TO-DO: bring vertices in fast order

	// Set up tracking for vertices
	// Pointing on data references and weights
	nbrTracker = new uint[numVtx + 1];
	nbrTracker[0] = 0;
	neighbors = new uint[numAngles];
	heWeights = new FN_TYPE[numAngles];

	// Set up vertex faces
	faceTracker = new uint[numVtx + 1];
	faceTracker[0] = 0;
	vertexFaces = new uint[numFaces * 3];
	faceWeights = new FN_TYPE[numFaces * 3];

	for (int i = 0; i < numVtx; i++) {
		SimpleTriMesh::VertexHandle v = mesh[i];
		SimpleTriMesh::VertexHandle vNbr;
		SimpleTriMesh::FaceHandle f;
		nbrTracker[i + 1] = 0;
		faceTracker[i + 1] = 0;

		SimpleTriMesh::VertexOHalfedgeIter heIter;
		for (heIter = orderedMesh.voh_begin(v); heIter; ++heIter) {
			vNbr = orderedMesh.to_vertex_handle(heIter.handle());
			neighbors[nbrTracker[i] + nbrTracker[i + 1]] = vNbr.idx(); //add neighbor reference
			heWeights[nbrTracker[i] + nbrTracker[i + 1]] = mStats.wHej[heIter.handle().idx()]; // add weight value
			nbrTracker[i + 1]++; // update tracker

			if (!orderedMesh.is_boundary(heIter.handle())) {
				f = orderedMesh.face_handle(heIter.handle());
				vertexFaces[faceTracker[i] + faceTracker[i + 1]] = f.idx();
				faceWeights[faceTracker[i] + faceTracker[i + 1]] = mStats.meshAngle[orderedMesh.prev_halfedge_handle( heIter.handle()).idx()];
				faceTracker[i + 1]++;
			}
		}
		nbrTracker[i + 1] += nbrTracker[i];
		faceTracker[i + 1] += faceTracker[i];
	}

	// Set up faces
	faceVertices = new uint[numFaces * 3];
	fv_weights = new FN_TYPE[numFaces * 3];
	for (int i = 0; i < numFaces; i++) {
		FaceHandle f = orderedMesh.face_handle(i);

		int counter = 0;
		for (SimpleTriMesh::FaceHalfedgeIter fhIter = orderedMesh.fh_begin(f); fhIter != orderedMesh.fh_end(f); ++fhIter) {
			VertexHandle v = orderedMesh.to_vertex_handle(fhIter.handle());
			faceVertices[i*3 + counter] = v.idx();
			fv_weights[i*3 + counter] = mStats.meshAngle[fhIter.handle().idx()];
			counter++;
		}
	}

	// Set up gradients
	gradients = new float3[numFaces * 2];
	OpenMesh::Vec3f vec;
	for (int i = 0; i < numFaces; i++) {
		vec = mStats.gradVec12[i];
		gradients[i * 2] = make_float3(vec.values_[0], vec.values_[1], vec.values_[2]);

		vec = mStats.gradVec13[i];
		gradients[i * 2 + 1] = make_float3(vec.values_[0], vec.values_[1], vec.values_[2]);
	}

	// set up helo
	halo_faces_keys = new uint[n+1]; halo_faces_keys[0] = 0;
	halo_vertices_keys = new uint[n+1]; halo_vertices_keys[0] = 0;
	halo_faces.clear();
	halo_vertices.clear();

	// fill face helo array
	for (int i = 0; i < n; i++) { // loop over partitions

		std::set<uint> halo_faces_part;
		std::set<uint> halo_vertices_part;
		for (int j = node_parts[i]; j < node_parts[i+1]; j++) { // loop over vertices of partition i
			VertexHandle v = orderedMesh.vertex_handle(j);

			// loop over neighbor faces
			for (SimpleTriMesh::VertexFaceIter vfIter = orderedMesh.vf_begin(v); vfIter != orderedMesh.vf_end(v); ++vfIter) {
				FaceHandle faNe = vfIter.handle();

				if (faNe.idx() < element_parts[i] || faNe.idx() >= element_parts[i+1]) {
					halo_faces_part.insert(faNe.idx());
				}
			}

			// loop over neighbor vertices
			for (SimpleTriMesh::VertexVertexIter vvIter = orderedMesh.vv_begin(v); vvIter != orderedMesh.vv_end(v); ++vvIter) {
				VertexHandle veNe = vvIter.handle();

				if (veNe.idx() < node_parts[i] || veNe.idx() >= node_parts[i+1]) {
					halo_vertices_part.insert(veNe.idx());
				}
			}
		}
		halo_faces.insert(halo_faces.end(), halo_faces_part.begin(), halo_faces_part.end());
		halo_vertices.insert(halo_vertices.end(), halo_vertices_part.begin(), halo_vertices_part.end());
		halo_faces_keys[i+1] = halo_faces.size();
		halo_vertices_keys[i+1] = halo_vertices.size();
	}


	/* -------- New Code ----------*/
	block_face_count = new uint[n];
	for (int i = 0; i < n; i++) {
		block_face_count[i] = element_parts[i+1] - element_parts[i] + halo_faces_keys[i+1] - halo_faces_keys[i];
	}

	max_vertex_count = 0;
	for (int i = 0; i < n; i++) {
		uint vertex_count = node_parts[i+1] - node_parts[i];
		max_vertex_count = std::max(max_vertex_count, vertex_count);
	}

	vertex_weights = new FN_TYPE[n*max_vertex_count];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < node_parts[i+1] - node_parts[i]; j++) {
			VertexHandle v = orderedMesh.vertex_handle(j + node_parts[i]);

			vertex_weights[i*max_vertex_count + j] = mStats.wVtx[v.idx()];
		}
	}

	vv_max_neighbors = 0;
	for (int i = 0; i < numVtx; i++) {
		VertexHandle v = orderedMesh.vertex_handle(i);

		uint vv_neighbors = 0;
		for (SimpleTriMesh::VertexOHalfedgeIter heIter = orderedMesh.voh_begin(v); heIter != orderedMesh.voh_end(v); ++heIter) {
			vv_neighbors++;
		}
		vv_max_neighbors = std::max(vv_neighbors, vv_max_neighbors);
	}

	nbr_v = new uint[n*(vv_max_neighbors+1)*max_vertex_count]; std::fill_n(nbr_v, n*(vv_max_neighbors+1)*max_vertex_count, 0);
	vv_weights = new FN_TYPE[n*vv_max_neighbors*max_vertex_count]; std::fill_n(vv_weights, n*vv_max_neighbors*max_vertex_count, 0.0);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < node_parts[i+1] - node_parts[i]; j++) {
			VertexHandle v = orderedMesh.vertex_handle(j+node_parts[i]);

			uint counter = 0;
			for (SimpleTriMesh::VertexOHalfedgeIter heIter = orderedMesh.voh_begin(v); heIter != orderedMesh.voh_end(v); ++heIter) {
				VertexHandle nbr = orderedMesh.to_vertex_handle(heIter.handle());
				nbr_v[i*(vv_max_neighbors+1)*max_vertex_count + (counter+1)*max_vertex_count + j] = nbr.idx();
				vv_weights[i*vv_max_neighbors*max_vertex_count + counter*max_vertex_count + j] = mStats.wHej[heIter.handle().idx()];
				counter++;
			}
			nbr_v[i*(vv_max_neighbors+1)*max_vertex_count + j] = counter;
		}
	}

	// Set up faces
	max_face_count = 0;
	for (int i = 0; i < n; i++) {
		uint he_size = element_parts[i+1] - element_parts[i] + halo_faces_keys[i+1] - halo_faces_keys[i];
		max_face_count = std::max(max_face_count, he_size);
	}

	he_grads = new float4[n*2*max_face_count];
	face_vertices = new uint[n*3*max_face_count]; std::fill_n(face_vertices, n*3*max_face_count, 0);
	fv_weights_new = new FN_TYPE[n*3*max_face_count]; std::fill_n(fv_weights_new, n*3*max_face_count, 0.0);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < element_parts[i+1] - element_parts[i]; j++) {
			FaceHandle f = orderedMesh.face_handle(j+element_parts[i]);

			// halfedge weights
			OpenMesh::Vec3f vec12 = mStats.gradVec12[f.idx()];
			he_grads[i*2*max_face_count + j] = make_float4(vec12.values_[0], vec12.values_[1], vec12.values_[2], 0.0);

			OpenMesh::Vec3f vec13 = mStats.gradVec13[f.idx()];
			he_grads[i*2*max_face_count + max_face_count + j] = make_float4(vec13.values_[0], vec13.values_[1], vec13.values_[2], 0.0);

			// face vertices indices and weights
			int counter = 0;
			for (SimpleTriMesh::FaceHalfedgeIter fhIter = orderedMesh.fh_begin(f); fhIter != orderedMesh.fh_end(f); ++fhIter) {
				VertexHandle v = orderedMesh.to_vertex_handle(fhIter.handle());
				face_vertices[i*3*max_face_count + counter*max_face_count + j] = v.idx();
				fv_weights_new[i*3*max_face_count + counter*max_face_count + j] = mStats.meshAngle[fhIter.handle().idx()];
				counter++;
			}
		}

		for (int j = 0; j < halo_faces_keys[i+1] - halo_faces_keys[i]; j++) {
			FaceHandle f = orderedMesh.face_handle(halo_faces[j+halo_faces_keys[i]]);

			// halfedge weights
			OpenMesh::Vec3f vec12 = mStats.gradVec12[f.idx()];
			he_grads[i*2*max_face_count + j + element_parts[i+1] - element_parts[i]] = make_float4(vec12.values_[0], vec12.values_[1], vec12.values_[2], 0.0);

			OpenMesh::Vec3f vec13 = mStats.gradVec13[f.idx()];
			he_grads[i*2*max_face_count + max_face_count + j + element_parts[i+1] - element_parts[i]] = make_float4(vec13.values_[0], vec13.values_[1], vec13.values_[2], 0.0);

			// face vertices indices and weights
			int counter = 0;
			for (SimpleTriMesh::FaceHalfedgeIter fhIter = orderedMesh.fh_begin(f); fhIter != orderedMesh.fh_end(f); ++fhIter) {
				VertexHandle v = orderedMesh.to_vertex_handle(fhIter.handle());
				face_vertices[i*3*max_face_count + counter*max_face_count + j + element_parts[i+1] - element_parts[i]] = v.idx();
				fv_weights_new[i*3*max_face_count + counter*max_face_count + j + element_parts[i+1] - element_parts[i]] = mStats.meshAngle[fhIter.handle().idx()];
				counter++;
			}
		}


	}
}

double GPUrun(int n) {
	// TO-DO find better spot for this
	//begin
	int max_size_n = 0;
	int max_size_e = 0;
	for (int i = 0; i < n; i++) {
		int size_n = node_parts[i+1] - node_parts[i];
		int size_e = element_parts[i+1] - element_parts[i] + halo_faces_keys[i+1] - halo_faces_keys[i];

		max_size_n = std::max(max_size_n, size_n);
		max_size_e = std::max(max_size_e, size_e);
	}

	if(max_size_n > 1024) {
		printf("ERROR: Too many nodes per Block! (%d %d)\n", n, max_size_n);
		return -1.0;
	}
	if(max_size_e > 1024) {
		printf("ERROR: Too many elements per Block! (%d %d)\n", n, max_size_e);
		return -1.0;
	}

	int threads_n = ((max_size_n + 32 - 1) / 32) * 32;
	int threads_e = ((max_size_e + 32 - 1) / 32) * 32;
	int threads = std::max(threads_n, threads_e);
	uint smem_size = max_size_n*7*4;
	//end

	FN_TYPE *dev_nFn_one;
	FN_TYPE *dev_cFn_one;
	FN_TYPE *dev_nFn_two;
	FN_TYPE *dev_cFn_two;
	uint *dev_nbrTracker;
	uint *dev_nbr;
	FN_TYPE *dev_vtxW;
	FN_TYPE *dev_heWeights;
	uint *dev_parts_n;
	uint *dev_parts_e;

	uint *dev_halo_faces;
	uint *dev_halo_faces_keys;
	uint *dev_halo_vertices;
	uint *dev_halo_vertices_keys;

	uint *dev_faceVertices;
	float3 *dev_heGradients;
	float3 *dev_nFaceGradients;
	float3 *dev_cFaceGradients;
	float3 *dev_nVertexGradients;
	float3 *dev_cVertexGradients;
	FN_TYPE *dev_wg;
	uint *dev_faceTracker;
	uint *dev_vertexFaces;
	FN_TYPE *dev_faceWeights;
	FN_TYPE *dev_fv_weights;
	uint *dev_block_face_count;

	size_t vw_pitchInBytes;

	cudaPitchedPtr dev_nbr_v;
	cudaPitchedPtr dev_vertex_weights;

	cudaPitchedPtr dev_he_grads;
	cudaPitchedPtr dev_face_vertices;
	cudaPitchedPtr dev_fv_weights_new;

	checkCudaErrors(cudaMalloc(&dev_block_face_count, sizeof(uint)*n));
	checkCudaErrors(cudaMalloc(&dev_nFn_one, sizeof(FN_TYPE)*numVtx));
	checkCudaErrors(cudaMalloc(&dev_cFn_one, sizeof(FN_TYPE)*numVtx));
	checkCudaErrors(cudaMalloc(&dev_nFn_two, sizeof(FN_TYPE)*numVtx));
	checkCudaErrors(cudaMalloc(&dev_cFn_two, sizeof(FN_TYPE)*numVtx));
	//checkCudaErrors(cudaMalloc(&dev_nbrTracker, sizeof(uint)*(numVtx+1)));
	//checkCudaErrors(cudaMalloc(&dev_nbr, sizeof(uint)*numAngles));
	//checkCudaErrors(cudaMalloc(&dev_vtxW, sizeof(FN_TYPE)*numVtx));
	//checkCudaErrors(cudaMalloc(&dev_heWeights, sizeof(FN_TYPE)*numAngles));
	checkCudaErrors(cudaMalloc(&dev_parts_n, sizeof(uint)*(n+1)));
	checkCudaErrors(cudaMalloc(&dev_parts_e, sizeof(uint)*(n+1)));

	checkCudaErrors(cudaMallocPitch(&dev_vtxW, &vw_pitchInBytes, max_vertex_count*sizeof(FN_TYPE), n));
	cudaExtent nbr_v_e = make_cudaExtent(max_vertex_count*sizeof(uint), vv_max_neighbors+1, n);
	checkCudaErrors(cudaMalloc3D(&dev_nbr_v, nbr_v_e));

	cudaExtent vertex_weights_e = make_cudaExtent(max_vertex_count*sizeof(FN_TYPE), vv_max_neighbors, n);
	checkCudaErrors(cudaMalloc3D(&dev_vertex_weights, vertex_weights_e));

	cudaExtent he_grads_e = make_cudaExtent(max_face_count*sizeof(float4), 2, n);
	checkCudaErrors(cudaMalloc3D(&dev_he_grads, he_grads_e));

	cudaExtent face_vertices_e = make_cudaExtent(max_face_count*sizeof(uint), 3, n);
	checkCudaErrors(cudaMalloc3D(&dev_face_vertices, face_vertices_e));

	cudaExtent fv_weights_e = make_cudaExtent(max_face_count*sizeof(FN_TYPE), 3, n);
	checkCudaErrors(cudaMalloc3D(&dev_fv_weights_new, fv_weights_e));


	checkCudaErrors(cudaMalloc(&dev_halo_faces, sizeof(uint)*halo_faces.size()));
	checkCudaErrors(cudaMalloc(&dev_halo_vertices, sizeof(uint)*halo_vertices.size()));
	checkCudaErrors(cudaMalloc(&dev_halo_faces_keys, sizeof(uint)*(n+1)));
	checkCudaErrors(cudaMalloc(&dev_halo_vertices_keys, sizeof(uint)*(n+1)));

	checkCudaErrors(cudaMalloc(&dev_faceVertices, sizeof(uint)*numFaces*3));
	checkCudaErrors(cudaMalloc(&dev_fv_weights, sizeof(FN_TYPE)*numFaces*3));
	checkCudaErrors(cudaMalloc(&dev_heGradients, sizeof(float3)*numFaces*2));

	checkCudaErrors(cudaMalloc(&dev_faceTracker, sizeof(uint)*(numVtx+1)));
	checkCudaErrors(cudaMalloc(&dev_vertexFaces, sizeof(uint)*numFaces*3));
	checkCudaErrors(cudaMalloc(&dev_faceWeights, sizeof(FN_TYPE)*numFaces*3));

	checkCudaErrors(cudaMalloc(&dev_nFaceGradients, sizeof(float3)*numFaces));
	checkCudaErrors(cudaMalloc(&dev_cFaceGradients, sizeof(float3)*numFaces));

	checkCudaErrors(cudaMalloc(&dev_nVertexGradients, sizeof(float3)*numVtx));
	checkCudaErrors(cudaMalloc(&dev_cVertexGradients, sizeof(float3)*numVtx));
	checkCudaErrors(cudaMalloc(&dev_wg, sizeof(FN_TYPE)*numVtx));
	checkCudaErrors(cudaMemset(dev_nVertexGradients, 0.0, sizeof(float3)*numVtx));
	checkCudaErrors(cudaMemset(dev_cVertexGradients, 0.0, sizeof(float3)*numVtx));
	checkCudaErrors(cudaMemset(dev_wg, 0.0, sizeof(FN_TYPE)*numVtx));

	checkCudaErrors(cudaMemcpy(dev_block_face_count, block_face_count, sizeof(uint)*n, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy2D(dev_vtxW, vw_pitchInBytes, vertex_weights, max_vertex_count*sizeof(FN_TYPE), max_vertex_count*sizeof(FN_TYPE), n, cudaMemcpyHostToDevice));

	cudaMemcpy3DParms nbr_v_p = {0};
	nbr_v_p.dstPtr = dev_nbr_v;
	nbr_v_p.srcPtr = make_cudaPitchedPtr(nbr_v, max_vertex_count*sizeof(uint), max_vertex_count, vv_max_neighbors+1);
	nbr_v_p.extent = nbr_v_e;
	nbr_v_p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&nbr_v_p));

	cudaMemcpy3DParms vertex_weights_p = {0};
	vertex_weights_p.dstPtr = dev_vertex_weights;
	vertex_weights_p.srcPtr = make_cudaPitchedPtr(vv_weights, max_vertex_count*sizeof(FN_TYPE), max_vertex_count, vv_max_neighbors);
	vertex_weights_p.extent = vertex_weights_e;
	vertex_weights_p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&vertex_weights_p));

	cudaMemcpy3DParms he_grads_p = {0};
	he_grads_p.dstPtr = dev_he_grads;
	he_grads_p.srcPtr = make_cudaPitchedPtr(he_grads, max_face_count*sizeof(float4), max_face_count, 2);
	he_grads_p.extent = he_grads_e;
	he_grads_p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&he_grads_p));

	cudaMemcpy3DParms face_vertices_p = {0};
	face_vertices_p.dstPtr = dev_face_vertices;
	face_vertices_p.srcPtr = make_cudaPitchedPtr(face_vertices, max_face_count*sizeof(uint), max_face_count, 3);
	face_vertices_p.extent = face_vertices_e;
	face_vertices_p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&face_vertices_p));

	cudaMemcpy3DParms fv_weights_p = {0};
	fv_weights_p.dstPtr = dev_fv_weights_new;
	fv_weights_p.srcPtr = make_cudaPitchedPtr(fv_weights_new, max_face_count*sizeof(FN_TYPE), max_face_count, 3);
	fv_weights_p.extent = fv_weights_e;
	fv_weights_p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&fv_weights_p));


	checkCudaErrors(cudaMemcpy(dev_nFn_one, nFn, sizeof(FN_TYPE)*numVtx, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_cFn_one, cFn, sizeof(FN_TYPE)*numVtx, cudaMemcpyHostToDevice));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<FN_TYPE>();
	bindN1Texture(dev_nFn_one, desc, numVtx*sizeof(FN_TYPE));
	bindC1Texture(dev_cFn_one, desc, numVtx*sizeof(FN_TYPE));
	bindN2Texture(dev_nFn_two, desc, numVtx*sizeof(FN_TYPE));
	bindC2Texture(dev_cFn_two, desc, numVtx*sizeof(FN_TYPE));
	//checkCudaErrors(cudaMemcpy(dev_nbrTracker, nbrTracker, sizeof(uint)*(numVtx+1), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(dev_nbr, neighbors, sizeof(uint)*numAngles, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(dev_vtxW, mStats.wVtx, sizeof(FN_TYPE)*numVtx, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(dev_heWeights, heWeights, sizeof(FN_TYPE)*numAngles, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_parts_n, node_parts, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_parts_e, element_parts, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(dev_halo_faces, &halo_faces[0], sizeof(uint)*halo_faces.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_halo_vertices, &halo_vertices[0], sizeof(uint)*halo_vertices.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_halo_faces_keys, halo_faces_keys, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_halo_vertices_keys, halo_vertices_keys, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(dev_faceVertices, faceVertices, sizeof(uint)*numFaces*3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_fv_weights, fv_weights, sizeof(FN_TYPE)*numFaces*3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_heGradients, gradients, sizeof(float3)*numFaces*2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_faceTracker, faceTracker, sizeof(uint)*(numVtx+1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_vertexFaces, vertexFaces, sizeof(uint)*numFaces*3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_faceWeights, faceWeights, sizeof(FN_TYPE)*numFaces*3, cudaMemcpyHostToDevice));


	cudaProfilerStart();
	step(dev_nFn_one, dev_cFn_one, dev_nFn_two, dev_cFn_two, (uint *)dev_face_vertices.ptr, (FN_TYPE *)dev_fv_weights_new.ptr, dev_face_vertices.pitch, (uint *)dev_nbr_v.ptr, dev_vtxW, vw_pitchInBytes, (FN_TYPE *)dev_vertex_weights.ptr, dev_nbr_v.pitch, vv_max_neighbors, (float4 *)dev_he_grads.ptr, dev_he_grads.pitch, dev_parts_n, dev_block_face_count, n, threads, dt, smem_size, true);
	cudaProfilerStop();
	step(dev_nFn_two, dev_cFn_two, dev_nFn_one, dev_cFn_one, (uint *)dev_face_vertices.ptr, (FN_TYPE *)dev_fv_weights_new.ptr, dev_face_vertices.pitch, (uint *)dev_nbr_v.ptr, dev_vtxW, vw_pitchInBytes, (FN_TYPE *)dev_vertex_weights.ptr, dev_nbr_v.pitch, vv_max_neighbors, (float4 *)dev_he_grads.ptr, dev_he_grads.pitch, dev_parts_n, dev_block_face_count, n, threads, dt, smem_size, false);



	// Error Test
	checkCudaErrors(cudaDeviceSynchronize());
	cudaError_t err = cudaGetLastError();
	if (err != 0)
		fprintf(stderr, "%s\n", cudaGetErrorString(err));

	// Correctness Test
	FN_TYPE *test_nFn = new FN_TYPE[numVtx];
	FN_TYPE *test_cFn = new FN_TYPE[numVtx];

	// Round 1
	CPUrun(test_nFn, test_cFn, 1);

	checkCudaErrors(cudaMemcpy(nFn, dev_nFn_two, sizeof(FN_TYPE)*numVtx, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cFn, dev_cFn_two, sizeof(FN_TYPE)*numVtx, cudaMemcpyDeviceToHost));

	double sum_error_n = 0.0, sum_error_c = 0.0;
	int error_counter = 0;
	for(int i = 0; i < numVtx; i++) {
		sum_error_n += fabs(nFn[i]-test_nFn[i]);
		sum_error_c += fabs(cFn[i]-test_cFn[i]);
		if (fabs(nFn[i]-test_nFn[i]) > 0 || fabs(cFn[i]-test_cFn[i]) > 0) {
			error_counter++;
			//printf("Error at %d: Should be %f but is %f!\n", i, test_nFn[i], nFn[i]);
		}
	}

	if (verbose || debug)
		printf("Round 1: Sum / Mean error for n: %f / %f and for c: %f / %f. Number of errors in total: %d\n", sum_error_n, sum_error_n/numVtx, sum_error_c, sum_error_c/numVtx, error_counter);

	// Round 2
	CPUrun(test_nFn, test_cFn, 1);

	checkCudaErrors(cudaMemcpy(nFn, dev_nFn_one, sizeof(FN_TYPE)*numVtx, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cFn, dev_cFn_one, sizeof(FN_TYPE)*numVtx, cudaMemcpyDeviceToHost));

	sum_error_n = 0.0; sum_error_c = 0.0;
	error_counter = 0;

	for(int i = 0; i < numVtx; i++) {
		sum_error_n += fabs(nFn[i]-test_nFn[i]);
		sum_error_c += fabs(cFn[i]-test_cFn[i]);
		if (fabs(nFn[i]-test_nFn[i]) > 0 || fabs(cFn[i]-test_cFn[i]) > 0) {
			error_counter++;
		}
	}

	if (verbose || debug)
		printf("Round 2: Sum / Mean error for n: %f / %f and for c: %f / %f. Number of errors in total: %d\n", sum_error_n, sum_error_n/numVtx, sum_error_c, sum_error_c/numVtx, error_counter);


	if(debug)
		return 0;

	// Speed Test
	int maxIt = 1000;


	float elapsedTime;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int i = 0; i < maxIt; i++) {
		step(dev_nFn_one, dev_cFn_one, dev_nFn_two, dev_cFn_two, (uint *)dev_face_vertices.ptr, (FN_TYPE *)dev_fv_weights_new.ptr, dev_face_vertices.pitch, (uint *)dev_nbr_v.ptr, dev_vtxW, vw_pitchInBytes, (FN_TYPE *)dev_vertex_weights.ptr, dev_nbr_v.pitch, vv_max_neighbors, (float4 *)dev_he_grads.ptr, dev_he_grads.pitch, dev_parts_n, dev_block_face_count, n, threads, dt, smem_size, true);
		step(dev_nFn_two, dev_cFn_two, dev_nFn_one, dev_cFn_one, (uint *)dev_face_vertices.ptr, (FN_TYPE *)dev_fv_weights_new.ptr, dev_face_vertices.pitch, (uint *)dev_nbr_v.ptr, dev_vtxW, vw_pitchInBytes, (FN_TYPE *)dev_vertex_weights.ptr, dev_nbr_v.pitch, vv_max_neighbors, (float4 *)dev_he_grads.ptr, dev_he_grads.pitch, dev_parts_n, dev_block_face_count, n, threads, dt, smem_size, false);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &elapsedTime, start, stop);
	elapsedTime /= 2;


	if (verbose) {
		printf("Time needed for %d iterations: %f ms with %d threads each and %d partitions\n", maxIt, elapsedTime, threads, n);
		printf("Average time needed per Iteration: %f us\n", 1000*elapsedTime/maxIt);
	} else {
		printf("Errors: %d, Partitions: %d, Threads: %d, Time: %f\n", error_counter, n, threads, 1000*elapsedTime/maxIt);
	}

	// Free Data
	checkCudaErrors(cudaFree(dev_nFn_one));
	checkCudaErrors(cudaFree(dev_cFn_one));
	checkCudaErrors(cudaFree(dev_nFn_two));
	checkCudaErrors(cudaFree(dev_cFn_two));
	checkCudaErrors(cudaFree(dev_nbrTracker));
	checkCudaErrors(cudaFree(dev_nbr));
	checkCudaErrors(cudaFree(dev_vtxW));
	checkCudaErrors(cudaFree(dev_heWeights));
	checkCudaErrors(cudaFree(dev_parts_n));
	checkCudaErrors(cudaFree(dev_parts_e));

	checkCudaErrors(cudaFree(dev_halo_faces));
	checkCudaErrors(cudaFree(dev_halo_faces_keys));
	checkCudaErrors(cudaFree(dev_halo_vertices));
	checkCudaErrors(cudaFree(dev_halo_vertices_keys));

	checkCudaErrors(cudaFree(dev_faceVertices));
	checkCudaErrors(cudaFree(dev_heGradients));
	checkCudaErrors(cudaFree(dev_nFaceGradients));
	checkCudaErrors(cudaFree(dev_cFaceGradients));
	checkCudaErrors(cudaFree(dev_nVertexGradients));
	checkCudaErrors(cudaFree(dev_cVertexGradients));
	checkCudaErrors(cudaFree(dev_wg));
	checkCudaErrors(cudaFree(dev_faceTracker));
	checkCudaErrors(cudaFree(dev_vertexFaces));
	checkCudaErrors(cudaFree(dev_faceWeights));
	checkCudaErrors(cudaFree(dev_fv_weights));

	unbindN1Texture();
	unbindC1Texture();
	unbindN2Texture();
	unbindC2Texture();

	return 1000*elapsedTime/maxIt;
}

void CPUrun(FN_TYPE *test_nFn, FN_TYPE *test_cFn, int num_steps) {
	std::copy(nFn, nFn + numVtx, test_nFn);
	std::copy(cFn, cFn + numVtx, test_cFn);

	for (int i = 0; i < num_steps; i++) {
		FN_TYPE *nLap = new FN_TYPE[numVtx];
		FN_TYPE *cLap = new FN_TYPE[numVtx];
		OpenMesh::Vec3f *facGradN = new OpenMesh::Vec3f[numFaces];
		OpenMesh::Vec3f *facGradC = new OpenMesh::Vec3f[numFaces];
		OpenMesh::Vec3f *vtxGradN = new OpenMesh::Vec3f[numVtx];
		OpenMesh::Vec3f *vtxGradC = new OpenMesh::Vec3f[numVtx];
		FN_TYPE *dauN = new FN_TYPE[numVtx];
		FN_TYPE *dauC = new FN_TYPE[numVtx];

		computeLaplacianCPU<FN_TYPE>(orderedMesh, test_nFn, nLap, mStats.wHej,
				mStats.wVtx);
		computeLaplacianCPU<FN_TYPE>(orderedMesh, test_cFn, cLap, mStats.wHej,
				mStats.wVtx);

		computeFaceGradientsCPU<OpenMesh::Vec3f>(orderedMesh, mStats.gradVec12,
				mStats.gradVec13, facGradN, test_nFn);
		computeVertexGradientsCPU<OpenMesh::Vec3f>(orderedMesh, facGradN,
				vtxGradN, mStats.meshAngle);

		computeFaceGradientsCPU<OpenMesh::Vec3f>(orderedMesh, mStats.gradVec12,
				mStats.gradVec13, facGradC, test_cFn);
		computeVertexGradientsCPU<OpenMesh::Vec3f>(orderedMesh, facGradC,
				vtxGradC, mStats.meshAngle);

		for (unsigned int i = 0; i < numVtx; ++i) {
			dauN[i] = D * nLap[i] - alpha * test_nFn[i] * cLap[i]
					- alpha * (vtxGradC[i] | vtxGradN[i])
					+ S * r * test_nFn[i] * (nMax - test_nFn[i]);
			dauC[i] = cLap[i]
					+ S * (-test_cFn[i] + test_nFn[i] / (1 + test_nFn[i]));
		}

		// update functions
		for (unsigned int i = 0; i < numVtx; ++i) {
			test_nFn[i] = test_nFn[i] + (dt * (dauN[i]));
			test_cFn[i] = test_cFn[i] + (dt * dauC[i]);

			if (test_nFn[i] < 0)
				test_nFn[i] = 0;

			if (test_cFn[i] < 0)
				test_cFn[i] = 0;

		}
	}
}
