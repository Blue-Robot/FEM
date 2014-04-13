#include <stdio.h>
#include <metis.h>
#include "partitioning_common.h"

using namespace std;


extern "C" int partition(SimpleTriMesh ipMesh, int parts) {
	idx_t numVtx = ipMesh.n_vertices();
	idx_t numFaces = ipMesh.n_faces();
	idx_t numAngles = ipMesh.n_halfedges();
	idx_t *eptr = new idx_t[numFaces+1], *eind = new idx_t[numAngles];
	eptr[0] = 0;
	idx_t *epart = new idx_t[numFaces], *npart = new idx_t[numVtx];

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
	idx_t nparts = parts;
	idx_t ncommon = 2;
	idx_t *options=NULL;
	int success = METIS_PartMeshDual(&numFaces, &numVtx, eptr, eind, NULL, NULL, &ncommon, &nparts, NULL, options, &objval, epart, npart);
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

	return objval;
}
