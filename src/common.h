#ifndef _COMMON_DEFS_H_
#define _COMMON_DEFS_H_

//#include <Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

typedef double PDE_DOUBLE;
typedef float  PDE_FLOAT;

typedef PDE_FLOAT FN_TYPE;
typedef PDE_DOUBLE GRID_PT_TYPE;


typedef struct pt_t {
  GRID_PT_TYPE x,y,z;
} PtVec;


typedef struct {
  FN_TYPE *meshAngle;   // list of angles of the triangles at the terminal vtx of the indexing HE
  FN_TYPE *meshCot;     // list of cots of the angles of the triangles at the terminal vtx of the indexing HE
  FN_TYPE *meshWij;     // list of weights for averaging vector fields at the terminal vtx of the indexing HE
  FN_TYPE *meshLamdaIj; // list of weights for <vecVtxG (dot) vecHij> averaging vector fields at the
                        // origninating vtx of the indexing HE

  FN_TYPE *meshArea;
  FN_TYPE *wHej;
  FN_TYPE *wVtx;
  FN_TYPE *areaPerVtx;
  double  avgEdgeLen;
  double  maxEdgeLen;

  OpenMesh::Vec3f *gradVec12; // vect v31 for a facea rotated by 90 deg in the face plane
  OpenMesh::Vec3f *gradVec13; // vect v12 for a face rotated by 90 in the face plane
  OpenMesh::Vec3f *gradVecHE; // edge vector for a HEdge rotated by 90 deg in the corresponding face's plane
  OpenMesh::Vec3f *vecHij;    // Hij is used to compute the Jacobian Natrix
  OpenMesh::Vec3f *vecVtxG;   // VtxG is used to compute the Jacobian Natrix
} MeshStats;


struct SimpleMeshTraits : public OpenMesh::DefaultTraits
{
  //typedef OpenMesh::Vec3d Color;
  typedef OpenMesh::Vec3uc Color;
  typedef OpenMesh::Vec3d Point;
  typedef OpenMesh::Vec3d Normal;

  VertexAttributes (OpenMesh::Attributes::Color | OpenMesh::Attributes::Normal);

};

typedef OpenMesh::TriMesh_ArrayKernelT<SimpleMeshTraits>  SimpleTriMesh;


#endif //_COMMON_H_
