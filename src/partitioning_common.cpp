#include <stdio.h>
#include <metis.h>
#include "partitioning_common.h"

using namespace OpenMesh;

// OpenMesh Includes


using namespace std;

extern "C" int partition(SimpleTriMesh *ipMesh, uint *parts, int n);
int generatePartitions(SimpleTriMesh ipMesh, long int *npart, uint *parts, int n);
void reorderMesh(SimpleTriMesh *ipMesh, long int *npart);




extern "C" int partition(SimpleTriMesh *ipMesh, uint *parts, int n) {
	long int *npart = new long int[(*ipMesh).n_vertices()];
	int communication = generatePartitions(*ipMesh, npart, parts, n);
	reorderMesh(ipMesh, npart);
	return communication;
}

int generatePartitions(SimpleTriMesh ipMesh, long int *npart, uint *parts, int n) {
	idx_t numVtx = ipMesh.n_vertices();
	idx_t numFaces = ipMesh.n_faces();
	idx_t numAngles = ipMesh.n_halfedges();
	idx_t *eptr = new idx_t[numFaces+1], *eind = new idx_t[numAngles];
	eptr[0] = 0;
	idx_t *epart = new idx_t[numFaces];

	int counter = 0;
	for (int i = 0; i < numFaces; i++) {
		SimpleTriMesh::FaceHandle face = ipMesh.face_handle(i);

		SimpleTriMesh::FaceEdgeIter vIter, vEnd(ipMesh.fe_end(face));
		for (vIter = ipMesh.fe_begin(face); vIter != vEnd; ++vIter) {
			eind[counter] = ipMesh.from_vertex_handle(vIter.current_halfedge_handle()).idx();
			counter++;
		}

		eptr[i+1] = counter;
	}

	idx_t objval;
	idx_t nparts = n;
	idx_t ncommon = 2;
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_NUMBERING] = 0;

	int success = METIS_PartMeshDual(&numFaces, &numVtx, eptr, eind, NULL, NULL, &ncommon, &nparts, NULL, options, &objval, epart, (idx_t*)npart);

	if (METIS_OK != success) {
		switch(success) {
		case METIS_ERROR_INPUT:
			printf("ERROR: Metis yield and input error!\n");
			break;
		case METIS_ERROR_MEMORY:
			printf("ERROR: Metis could not allocate the required memory!\n");
			break;
		case METIS_ERROR:
			printf("ERROR: Metis yields some undefined error!\n");
			break;
		default:
			printf("WARNING: Metis returned some strange value %d!\n", success);

		}

	}

	int *part_freq = new int[nparts]; std::fill_n(part_freq, nparts, 0);
	//long *part_sum ...
	long *npart_tmp = new long[numVtx];
	for (long i = 0; i < numVtx; i++) {
		long part_idx = npart[i];
		npart_tmp[i] = part_freq[part_idx];
		part_freq[part_idx]++;
	}

	for (long i = 0; i < nparts; i++) {
		part_freq[i] += i == 0 ? 0 : part_freq[i-1];
	}

	for (long i = 0; i < numVtx; i++) {
		long part_idx = npart[i];
		long offset = part_idx == 0 ? 0 : part_freq[part_idx - 1];
		npart[i] = npart_tmp[i] + offset;
	}

	// Add partition start indices
	for (int i = 0; i < n+1; i++) {
		parts[i] = i == 0 ? 0 : part_freq[i-1];

	}

	return objval;
}

void reorderMesh(SimpleTriMesh *orderedMesh, long int *npart) {
	SimpleTriMesh originalMesh = *orderedMesh;
	(*orderedMesh).clean();

	long int *npart_inv = new long int[originalMesh.n_vertices()];
	for (int i = 0; i < originalMesh.n_vertices(); i++) {
		int index = npart[i];
		npart_inv[index] = i;
	}

	for (int i = 0; i < originalMesh.n_vertices(); i++) {
		VertexHandle v = originalMesh.vertex_handle(npart_inv[i]);
		(*orderedMesh).add_vertex(originalMesh.point(v));
	}

	for (SimpleTriMesh::FaceIter fIter = originalMesh.faces_begin(); fIter != originalMesh.faces_end(); ++fIter) {
		SimpleTriMesh::FaceVertexIter fvIter = originalMesh.fv_begin(fIter.handle());
		VertexHandle v1 = (*orderedMesh).vertex_handle(npart[fvIter.handle().idx()]); ++fvIter;
		VertexHandle v2 = (*orderedMesh).vertex_handle(npart[fvIter.handle().idx()]); ++fvIter;
		VertexHandle v3 = (*orderedMesh).vertex_handle(npart[fvIter.handle().idx()]);

		(*orderedMesh).add_face(v1, v2, v3);
	}
}


