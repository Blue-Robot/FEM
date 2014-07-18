#include "common.h"

extern "C" int partition(
		SimpleTriMesh originalMesh,
		SimpleTriMesh *orderedMesh,
		uint **node_parts,
		uint **element_parts,
		int n
		);
