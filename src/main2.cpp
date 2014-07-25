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
const FN_TYPE D = 0.25;
const FN_TYPE r = 1.52;
const FN_TYPE alpha = 12.02;
const FN_TYPE S = 1;
double dt;

// General
string file_name;
bool verbose = false;

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
float3 *gradients;

uint *node_parts;
uint *element_parts;

std::vector<uint> halo_faces;
std::vector<uint> halo_vertices;
uint *halo_faces_keys;
uint *halo_vertices_keys;

//batch configuration
int start_n = 39; // number of partitions to start with
int end_n = 39; // number of partitions to end with

using namespace OpenMesh;

int main(int argc, char **argv);

void initializeCUDA();
void loadMesh();
void initializeData(int n);
void initializeGPUData(int n);

double GPUrun(int n);
void CPUrun(FN_TYPE *test_nFn, FN_TYPE *test_cFn);


int main(int argc, char **argv) {
file_name = argv[1];

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
printf("Best time with %d partitions is %f!\n", best_n, best_time);

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
//checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
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
SimpleTriMesh::FaceIter fIter, fEnd = orderedMesh.faces_end();
for (fIter = orderedMesh.faces_begin(); fIter != fEnd; ++fIter) {
uint fIdx = fIter.handle().idx();
SimpleTriMesh::FaceVertexIter fvIter = orderedMesh.fv_begin(fIter.handle());
for (int i = 0; i < 3; i++) {
uint vIdx = fvIter.handle().idx();
faceVertices[fIdx * 3 + i] = vIdx;
++fvIter;
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

}

double GPUrun(int n) {
// TO-DO find better spot for this
//begin
int max_size_n = 0;
int max_size_e = 0;

for (int i = 0; i < n; i++) {
int size_n = node_parts[i+1] - node_parts[i] + halo_vertices_keys[i+1] - halo_vertices_keys[i];
int size_e = element_parts[i+1] - element_parts[i] + halo_faces_keys[i+1] - halo_faces_keys[i];

max_size_n = std::max(max_size_n, size_n);
max_size_e = std::max(max_size_e, size_e);
}

if(max_size_n > 1024) {
fprintf(stderr, "ERROR: Too many nodes per Block! (%d %d)\n", n, max_size_n);
return -1.0;
}
if(max_size_e > 1024) {
fprintf(stderr, "ERROR: Too many elements per Block! (%d %d)\n", n, max_size_e);
return -1.0;
}
int max_size = std::max(max_size_n, max_size_e);

int threads_n = ((max_size_n + 32 - 1) / 32) * 32;
int threads_e = ((max_size_e + 32 - 1) / 32) * 32;
int threads = ((max_size + 32 - 1) / 32) * 32;
//end

FN_TYPE *dev_nFn;
FN_TYPE *dev_cFn;
FN_TYPE *dev_nLap;
FN_TYPE *dev_cLap;
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
uint *dev_faceTracker;
uint *dev_vertexFaces;
FN_TYPE *dev_faceWeights;

checkCudaErrors(cudaMalloc(&dev_nFn, sizeof(FN_TYPE)*numVtx));
checkCudaErrors(cudaMalloc(&dev_cFn, sizeof(FN_TYPE)*numVtx));
checkCudaErrors(cudaMalloc(&dev_nbrTracker, sizeof(uint)*(numVtx+1)));
checkCudaErrors(cudaMalloc(&dev_nbr, sizeof(uint)*numAngles));
checkCudaErrors(cudaMalloc(&dev_vtxW, sizeof(FN_TYPE)*numVtx));
checkCudaErrors(cudaMalloc(&dev_heWeights, sizeof(FN_TYPE)*numAngles));
checkCudaErrors(cudaMalloc(&dev_parts_n, sizeof(uint)*(n+1)));
checkCudaErrors(cudaMalloc(&dev_parts_e, sizeof(uint)*(n+1)));

checkCudaErrors(cudaMalloc(&dev_halo_faces, sizeof(uint)*halo_faces.size()));
checkCudaErrors(cudaMalloc(&dev_halo_vertices, sizeof(uint)*halo_vertices.size()));
checkCudaErrors(cudaMalloc(&dev_halo_faces_keys, sizeof(uint)*(n+1)));
checkCudaErrors(cudaMalloc(&dev_halo_vertices_keys, sizeof(uint)*(n+1)));

checkCudaErrors(cudaMalloc(&dev_faceVertices, sizeof(uint)*numFaces*3));
checkCudaErrors(cudaMalloc(&dev_heGradients, sizeof(float3)*numFaces*2));

checkCudaErrors(cudaMalloc(&dev_faceTracker, sizeof(uint)*(numVtx+1)));
checkCudaErrors(cudaMalloc(&dev_vertexFaces, sizeof(uint)*numFaces*3));
checkCudaErrors(cudaMalloc(&dev_faceWeights, sizeof(FN_TYPE)*numFaces*3));

checkCudaErrors(cudaMalloc(&dev_nLap, sizeof(FN_TYPE)*numVtx));
checkCudaErrors(cudaMalloc(&dev_cLap, sizeof(FN_TYPE)*numVtx));
checkCudaErrors(cudaMalloc(&dev_nFaceGradients, sizeof(float3)*numFaces));
checkCudaErrors(cudaMalloc(&dev_cFaceGradients, sizeof(float3)*numFaces));
checkCudaErrors(cudaMalloc(&dev_nVertexGradients, sizeof(float3)*numVtx));
checkCudaErrors(cudaMalloc(&dev_cVertexGradients, sizeof(float3)*numVtx));


checkCudaErrors(cudaMemcpy(dev_nFn, nFn, sizeof(FN_TYPE)*numVtx, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_cFn, cFn, sizeof(FN_TYPE)*numVtx, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_nbrTracker, nbrTracker, sizeof(uint)*(numVtx+1), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_nbr, neighbors, sizeof(uint)*numAngles, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_vtxW, mStats.wVtx, sizeof(FN_TYPE)*numVtx, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_heWeights, heWeights, sizeof(FN_TYPE)*numAngles, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_parts_n, node_parts, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_parts_e, element_parts, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));

checkCudaErrors(cudaMemcpy(dev_halo_faces, &halo_faces[0], sizeof(uint)*halo_faces.size(), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_halo_vertices, &halo_vertices[0], sizeof(uint)*halo_vertices.size(), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_halo_faces_keys, halo_faces_keys, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_halo_vertices_keys, halo_vertices_keys, sizeof(uint)*(n+1), cudaMemcpyHostToDevice));

checkCudaErrors(cudaMemcpy(dev_faceVertices, faceVertices, sizeof(uint)*numFaces*3, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_heGradients, gradients, sizeof(float3)*numFaces*2, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_faceTracker, faceTracker, sizeof(uint)*(numVtx+1), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_vertexFaces, vertexFaces, sizeof(uint)*numFaces*3, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(dev_faceWeights, faceWeights, sizeof(FN_TYPE)*numFaces*3, cudaMemcpyHostToDevice));

cudaProfilerStart();
compute(dev_nFn, dev_cFn, dev_nLap, dev_cLap, dev_nbrTracker, dev_nbr, dev_vtxW, dev_heWeights, dev_halo_vertices, dev_halo_vertices_keys, dev_parts_n, dev_faceVertices, dev_heGradients, dev_nFaceGradients, dev_cFaceGradients, dev_halo_faces, dev_halo_faces_keys, dev_parts_e, dev_nVertexGradients, dev_cVertexGradients, dev_faceTracker, dev_vertexFaces, dev_faceWeights, n, threads);
cudaProfilerStop();
// computeLaplacian(dev_nFn, dev_cFn, dev_nLap, dev_cLap, dev_nbrTracker, dev_nbr, dev_vtxW, dev_heWeights, dev_halo_vertices, dev_halo_vertices_keys, dev_parts_n, n, threads_n);
// computeFaceGradients(dev_faceVertices, dev_nFn, dev_cFn, dev_heGradients, dev_nFaceGradients, dev_cFaceGradients, dev_halo_faces, dev_halo_faces_keys, dev_parts_e, n, threads_e);
// computeVertexGradients(dev_nFaceGradients, dev_cFaceGradients, dev_nVertexGradients, dev_cVertexGradients, dev_faceTracker, dev_vertexFaces, dev_faceWeights, dev_parts_n, n, threads_n);

FN_TYPE *test_nFn = new FN_TYPE[numVtx];
FN_TYPE *test_cFn = new FN_TYPE[numVtx];
CPUrun(test_nFn, test_cFn);

update(dev_nFn, dev_cFn, dev_nLap, dev_cLap, dev_nVertexGradients, dev_cVertexGradients, dt, numVtx, 64);

// Error Test
checkCudaErrors(cudaDeviceSynchronize());
cudaError_t err = cudaGetLastError();
if (err != 0)
fprintf(stderr, "%s\n", cudaGetErrorString(err));

checkCudaErrors(cudaMemcpy(nFn, dev_nFn, sizeof(FN_TYPE)*numVtx, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(cFn, dev_cFn, sizeof(FN_TYPE)*numVtx, cudaMemcpyDeviceToHost));

// Correctness Test
double sum_error_n = 0.0, sum_error_c = 0.0;
int error_counter = 0;
for(int i = 0; i < numVtx; i++) {
sum_error_n += fabs(nFn[i]-test_nFn[i]);
sum_error_c += fabs(cFn[i]-test_cFn[i]);
if (fabs(nFn[i]-test_nFn[i]) > 0 || fabs(cFn[i]-test_cFn[i]) > 0) {
error_counter++;
//printf("Vertex %d should be %f but is %f\n", i, test_nFn[i], nFn[i]);
}
}

if (verbose)
printf("Sum / Mean error for n: %f / %f and for c: %f / %f. Number of errors in total: %d\n", sum_error_n, sum_error_n/numVtx, sum_error_c, sum_error_c/numVtx, error_counter);


// Speed Test
int maxIt = 1000;
int best_configuartion;
float best_time = -1;
for (int th = 32; th <= 1024; th+=32) {
float elapsedTime;
cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

for (int i = 0; i < maxIt; i++) {
compute(dev_nFn, dev_cFn, dev_nLap, dev_cLap, dev_nbrTracker, dev_nbr, dev_vtxW, dev_heWeights, dev_halo_vertices, dev_halo_vertices_keys, dev_parts_n, dev_faceVertices, dev_heGradients, dev_nFaceGradients, dev_cFaceGradients, dev_halo_faces, dev_halo_faces_keys, dev_parts_e, dev_nVertexGradients, dev_cVertexGradients, dev_faceTracker, dev_vertexFaces, dev_faceWeights, n, threads);

update(dev_nFn, dev_cFn, dev_nLap, dev_cLap, dev_nVertexGradients, dev_cVertexGradients, dt, numVtx, th);
}

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime( &elapsedTime, start, stop);
if (best_time < 0 || best_time > elapsedTime) {
best_time = elapsedTime;
best_configuartion = th;
}
}
if (verbose) {
printf("Time needed for %d iterations: %f ms with %d threads each and %d partitions\n", maxIt, best_time, best_configuartion, n);
printf("Average time needed per Iteration: %f us\n", 1000*best_time/maxIt);
} else {
printf("Errors: %d, Partitions: %d, Threads: %d, Time: %f\n", error_counter, n, threads, 1000*best_time/maxIt);
}

// Free Data
checkCudaErrors(cudaFree(dev_nFn));
checkCudaErrors(cudaFree(dev_cFn));
checkCudaErrors(cudaFree(dev_nLap));
checkCudaErrors(cudaFree(dev_cLap));
checkCudaErrors(cudaFree(dev_nbrTracker));
checkCudaErrors(cudaFree(dev_nbr));
checkCudaErrors(cudaFree(dev_vtxW));
checkCudaErrors(cudaFree(dev_heWeights));
checkCudaErrors(cudaFree(dev_parts_n));
checkCudaErrors(cudaFree(dev_parts_e));

checkCudaErrors(cudaFree(dev_halo_faces));
checkCudaErrors(cudaFree(dev_halo_faces_keys));

checkCudaErrors(cudaFree(dev_faceVertices));
checkCudaErrors(cudaFree(dev_heGradients));
checkCudaErrors(cudaFree(dev_nFaceGradients));
checkCudaErrors(cudaFree(dev_cFaceGradients));
checkCudaErrors(cudaFree(dev_nVertexGradients));
checkCudaErrors(cudaFree(dev_cVertexGradients));
checkCudaErrors(cudaFree(dev_faceTracker));
checkCudaErrors(cudaFree(dev_vertexFaces));
checkCudaErrors(cudaFree(dev_faceWeights));

return 1000*best_time/maxIt;
}

void CPUrun(FN_TYPE *test_nFn, FN_TYPE *test_cFn) {
FN_TYPE *nLap = new FN_TYPE[numVtx];
FN_TYPE *cLap = new FN_TYPE[numVtx];
OpenMesh::Vec3f *facGradN = new OpenMesh::Vec3f[numFaces];
OpenMesh::Vec3f *facGradC = new OpenMesh::Vec3f[numFaces];
OpenMesh::Vec3f *vtxGradN = new OpenMesh::Vec3f[numVtx];
OpenMesh::Vec3f *vtxGradC = new OpenMesh::Vec3f[numVtx];
FN_TYPE *dauN = new FN_TYPE[numVtx];
FN_TYPE *dauC = new FN_TYPE[numVtx];

computeLaplacianCPU<FN_TYPE>(orderedMesh, nFn, nLap, mStats.wHej, mStats.wVtx);
computeLaplacianCPU<FN_TYPE>(orderedMesh, cFn, cLap, mStats.wHej, mStats.wVtx);

computeFaceGradientsCPU<OpenMesh::Vec3f>(orderedMesh, mStats.gradVec12, mStats.gradVec13, facGradN, nFn);
computeVertexGradientsCPU<OpenMesh::Vec3f>(orderedMesh, facGradN, vtxGradN, mStats.meshAngle);

computeFaceGradientsCPU<OpenMesh::Vec3f>(orderedMesh, mStats.gradVec12, mStats.gradVec13, facGradC, cFn);
computeVertexGradientsCPU<OpenMesh::Vec3f>(orderedMesh, facGradC, vtxGradC, mStats.meshAngle);

for(unsigned int i=0; i < numVtx; ++i)
{
dauN[i] = D*nLap[i] - alpha*nFn[i]*cLap[i] - alpha* (vtxGradC[i] | vtxGradN[i]) + S*r*nFn[i]*(nMax - nFn[i]);
dauC[i] = cLap[i] + S * (-cFn[i] + nFn[i]/(1+nFn[i]) );
}

// update functions
for(unsigned int i=0; i < numVtx; ++i)
{
test_nFn[i] = nFn[i] + (dt*(dauN[i]));
test_cFn[i] = cFn[i] + (dt*dauC[i]);

if (test_nFn[i] <0)
test_nFn[i] = 0;

if (test_cFn[i] <0)
test_cFn[i] = 0;

}
}
