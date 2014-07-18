#include <stdio.h>
#include <metis.h>
#include "partitioning_common.h"

using namespace OpenMesh;

// OpenMesh Includes


using namespace std;

extern "C" int partition(SimpleTriMesh originalMesh, SimpleTriMesh *orderedMesh, uint **node_parts, uint **element_parts, int n);
int generatePartitions(SimpleTriMesh ipMesh, long int *npart, long int *epart, uint *node_parts, uint *element_parts, int n);
void createOrderArray(uint *parts, long int *part, int numVtx, int n);
void reorderMesh(SimpleTriMesh *orderedMesh, long int *npart, long int *epart);
void invertArray(long int *part, long int *part_inv, int n);




extern "C" int partition(SimpleTriMesh originalMesh, SimpleTriMesh *orderedMesh, uint **node_parts, uint **element_parts, int n) {
	*orderedMesh = originalMesh;
	*node_parts = new uint[n+1];
	*element_parts = new uint[n+2];
	long int *npart = new long int[(*orderedMesh).n_vertices()];
	long int *epart = new long int[(*orderedMesh).n_faces()];
	int communication = generatePartitions(*orderedMesh, npart, epart, *node_parts, *element_parts, n);
	reorderMesh(orderedMesh, npart, epart);
	return communication;
}

int generatePartitions(SimpleTriMesh ipMesh, long int *npart, long int *epart, uint *node_parts, uint *element_parts, int n) {
	idx_t numVtx = ipMesh.n_vertices();
	idx_t numFaces = ipMesh.n_faces();
	idx_t numAngles = ipMesh.n_halfedges();
	idx_t *eptr = new idx_t[numFaces+1], *eind = new idx_t[numAngles];
	eptr[0] = 0;

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

	// create face array, index i = j means face i belongs to partition j
	for (int i = 0; i < numFaces; i++) {
		FaceHandle f = ipMesh.face_handle(i);
		SimpleTriMesh::FaceVertexIter fvIter = ipMesh.fv_begin(f);
		int v1 = fvIter.handle().idx(); ++fvIter;
		int v2 = fvIter.handle().idx(); ++fvIter;
		int v3 = fvIter.handle().idx();
		long int v1_part = npart[v1];
		long int v2_part = npart[v2];
		long int v3_part = npart[v3];

		if (v1_part == v2_part && v2_part == v3_part) {
			epart[i] = v1_part;
		} else {
			epart[i] = n;
		}

	}



	createOrderArray(node_parts, npart, numVtx, nparts);
	createOrderArray(element_parts, epart, numFaces, nparts+1);

	return objval;
}

/*
 * Creates parts array and part array.
 * parts array tells the first index of every partition
 * part array tells the new position of the old index
 */
void createOrderArray(uint *parts, long int *part, int numVtx, int n) {
	int *part_freq = new int[n]; std::fill_n(part_freq, n, 0);

	long *npart_tmp = new long[numVtx];
	for (long i = 0; i < numVtx; i++) {
		long part_idx = part[i];
		npart_tmp[i] = part_freq[part_idx];
		part_freq[part_idx]++;
	}

	for (long i = 0; i < n; i++) {
		part_freq[i] += i == 0 ? 0 : part_freq[i-1];
	}

	for (long i = 0; i < numVtx; i++) {
		long part_idx = part[i];
		long offset = part_idx == 0 ? 0 : part_freq[part_idx - 1];
		part[i] = npart_tmp[i] + offset;
	}

	// Add partition start indices
	for (int i = 0; i < n+1; i++) {
		parts[i] = i == 0 ? 0 : part_freq[i-1];

	}
}

void reorderMesh(SimpleTriMesh *orderedMesh, long int *npart, long int *epart) {
	SimpleTriMesh originalMesh = *orderedMesh;
	(*orderedMesh).clean();

	long int *npart_inv = new long int[originalMesh.n_vertices()];
	long int *epart_inv = new long int[originalMesh.n_faces()];
	invertArray(npart, npart_inv, originalMesh.n_vertices());
	invertArray(epart, epart_inv, originalMesh.n_faces());

	for (int i = 0; i < originalMesh.n_vertices(); i++) {
		VertexHandle v = originalMesh.vertex_handle(npart_inv[i]);
		(*orderedMesh).add_vertex(originalMesh.point(v));
	}

	for (int i = 0; i < originalMesh.n_faces(); i++) {
		FaceHandle f = originalMesh.face_handle(epart_inv[i]);
		SimpleTriMesh::FaceVertexIter fvIter = originalMesh.fv_begin(f);
		VertexHandle v1 = (*orderedMesh).vertex_handle(npart[fvIter.handle().idx()]); ++fvIter;
		VertexHandle v2 = (*orderedMesh).vertex_handle(npart[fvIter.handle().idx()]); ++fvIter;
		VertexHandle v3 = (*orderedMesh).vertex_handle(npart[fvIter.handle().idx()]);

		(*orderedMesh).add_face(v1, v2, v3);
	}
}

void invertArray(long int *part, long int *part_inv, int n) {
	for (int i = 0; i < n; i++) {
		int index = part[i];
		part_inv[index] = i;
	}
}



