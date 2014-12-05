#ifndef _MESH_FN_H_
#define _MESH_FN_H_

#include "common.h"

template<class Data>
bool computeLaplacianCPU(SimpleTriMesh &m, Data *fn, Data *lap, FN_TYPE *wHej, FN_TYPE *wVtx)
{

  SimpleTriMesh::VertexIter vI, vEnd(m.vertices_end());

  for(vI = m.vertices_begin(); vI != vEnd; ++vI)
  {
    uint hIdx;
    uint vIdx = (*vI).idx();
    uint vNbr = (*vI).idx();

    lap[vIdx] = fn[vIdx] * wVtx[vIdx];

    SimpleTriMesh::VertexOHalfedgeIter hIter;
    for (hIter = m.voh_iter(*vI); hIter.is_valid(); ++hIter)
    {

      vNbr = m.to_vertex_handle(*hIter).idx();

      hIdx = (*hIter).idx();

      lap[vIdx] += fn[vNbr] * wHej[hIdx];
    }
    //lap[vIdx] /= wVtx[vIdx];
  }
  return true;
}

template <class Data>
bool computeFaceGradientsCPU(SimpleTriMesh &m, Data *gradV12,Data *gradV13, Data *opGrad, FN_TYPE *fn)
{

  SimpleTriMesh::FaceIter fIter, fEnd(m.faces_end());
  for (fIter = m.faces_begin(); fIter != fEnd; ++fIter)
  {
    OpenMesh::VertexHandle v1,v2,v3;

    OpenMesh::FaceHandle fCurr;
    OpenMesh::HalfedgeHandle hCurr,hNext;

    fCurr = *fIter;
    hCurr = m.halfedge_handle(fCurr);
    hNext = m.next_halfedge_handle(hCurr);

    v1 = m.from_vertex_handle(hCurr);
    v2 = m.to_vertex_handle(hCurr);
    v3 = m.to_vertex_handle(hNext);

    opGrad[fCurr.idx()] = gradV12[fCurr.idx()] * (fn[v2.idx()] - fn[v1.idx()])  +
                          gradV13[fCurr.idx()] * (fn[v3.idx()] - fn[v1.idx()]);
  }
  return true;
}

template <class Data>
bool computeVertexGradientsCPU(SimpleTriMesh &m, Data *faceGrad, Data *opVtxGrad, FN_TYPE *meshAngles)
{

  SimpleTriMesh::VertexIter vI, vEnd(m.vertices_end());
  for(vI = m.vertices_begin(); vI != vEnd; ++vI)
  {
    //uint hIdx;
    uint vIdx = (*vI).idx();
    uint fIdx;


    opVtxGrad[vIdx] *= 0;

    SimpleTriMesh::VertexOHalfedgeIter hIter;
    FN_TYPE wg = 0;

    Data tempD;

    for (hIter = m.voh_iter(*vI); hIter.is_valid(); ++hIter)
    {

      OpenMesh::HalfedgeHandle hCurr, hPrev;

      hCurr = *hIter;


      if(!m.is_boundary(hCurr))
      {
        fIdx = m.face_handle(hCurr).idx();
        hPrev = m.prev_halfedge_handle(hCurr);

        tempD = faceGrad[fIdx] * meshAngles[hPrev.idx()];

        opVtxGrad[vIdx] += tempD;

        wg += meshAngles[hPrev.idx()];
      } //end if
    } // end for

    if (wg > 0)
    {
      opVtxGrad[vIdx] /= wg;
    } else {
      opVtxGrad[vIdx] *= 0;
    }
  } // end vI

  return true;
}


void computeLaplacianWeights(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeAngles(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeNormalizedAngleWeights(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeAverageEdgeLength(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeMaxEdgeLength(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeAreas(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeGradVectors(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeHEGradVectors(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeHijVecForJacobian(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeVtxGVecForJacobian(SimpleTriMesh &ipMesh, MeshStats &mStats);
void computeHijLambdaForJacobian(SimpleTriMesh &ipMesh, MeshStats &mStats);
void initMeshStats(SimpleTriMesh &ipMesh, MeshStats &mStats);
void deleteMeshStats(MeshStats &mStats);

#endif //_MESH_FN_H_
