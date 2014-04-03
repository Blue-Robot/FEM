#include "mesh_function.h"

//#include <OpenMesh/Tools/Utils/Timer.hh>

float mini = 10.0f;

void deleteMeshStats(MeshStats &mStats)
{
  delete [] mStats.meshAngle;
  delete [] mStats.meshCot;
  delete [] mStats.meshArea;

  delete[] mStats.wHej;
  delete[] mStats.wVtx;
  delete[] mStats.areaPerVtx;
  delete[] mStats.meshWij;
  delete[] mStats.meshLamdaIj;


  delete[] mStats.gradVec12;
  delete[] mStats.gradVec13;
  delete[] mStats.gradVecHE;
  delete[] mStats.vecHij;
  delete[] mStats.vecVtxG;
}

void initMeshStats(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  uint numAngles = ipMesh.n_halfedges();
  uint numFaces  = ipMesh.n_faces();
  uint numVtx    = ipMesh.n_vertices();

  mStats.meshAngle    = new FN_TYPE[numAngles];
  mStats.meshCot      = new FN_TYPE[numAngles];
  mStats.meshArea     = new FN_TYPE[numFaces];
  mStats.wHej         = new FN_TYPE[numAngles];
  mStats.wVtx         = new FN_TYPE[numVtx];
  mStats.areaPerVtx   = new FN_TYPE[numVtx];
  mStats.meshWij      = new FN_TYPE[numAngles];
  mStats.meshLamdaIj  = new FN_TYPE[numAngles];


  mStats.gradVec12     = new OpenMesh::Vec3f[numFaces];
  mStats.gradVec13     = new OpenMesh::Vec3f[numFaces];
  mStats.gradVecHE     = new OpenMesh::Vec3f[ipMesh.n_halfedges()];
  mStats.vecHij        = new OpenMesh::Vec3f[ipMesh.n_halfedges()];
  mStats.vecVtxG       = new OpenMesh::Vec3f[numVtx];

  for(unsigned int i = 0; i < numVtx; ++i)
  {
    mStats.wVtx[i] = mStats.areaPerVtx[i] = 0;
  }

  computeAngles(ipMesh, mStats);
  computeNormalizedAngleWeights(ipMesh, mStats);
  computeAreas(ipMesh, mStats);
  computeLaplacianWeights(ipMesh, mStats);
  computeAverageEdgeLength(ipMesh, mStats);
  computeMaxEdgeLength(ipMesh, mStats);
  computeGradVectors(ipMesh, mStats);
  computeHEGradVectors(ipMesh, mStats);

  computeHijVecForJacobian(ipMesh, mStats);
  computeVtxGVecForJacobian(ipMesh, mStats);
  computeHijLambdaForJacobian(ipMesh, mStats);

}


void computeAngles(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  FN_TYPE *meshAngle    = mStats.meshAngle;
  FN_TYPE *meshCot      = mStats.meshCot;

  OpenMesh::Vec3f pt1, pt2, pt3, pt4;
  OpenMesh::Vec3f v21, v23, v34;

  //OpenMesh::Utils::Timer t;
  //double minAng,maxAng;

#ifdef _DEBUG_ANGLE_CALC_
  double minSum,maxSum;


  maxSum = maxAng = 0;
  minSum = minAng = 44.0/7;
#endif

  //maxAng = 0;
  //minAng = 45.0/7; //more than 2 pi
  std::cout << "Compute Angles..." << std::flush;
  //t.start();

  SimpleTriMesh::HalfedgeIter heIter, heEnd(ipMesh.halfedges_end());
  for (heIter = ipMesh.halfedges_begin(); heIter != heEnd; ++heIter)
  {
    OpenMesh::VertexHandle v1,v2,v3,v4;
    OpenMesh::HalfedgeHandle hCurr, hNext, hNext2;

    hCurr = heIter.handle();


    if (ipMesh.is_boundary(hCurr)) {

      meshAngle[hCurr.idx()] = 0;
      meshCot[hCurr.idx()] = 0;
      v1 = ipMesh.from_vertex_handle(hCurr);
      v2 = ipMesh.to_vertex_handle(hCurr);

    } else {

      hNext = ipMesh.next_halfedge_handle(hCurr);
      //hNext2 = ipMesh.next_halfedge_handle(hNext);

      v1 = ipMesh.from_vertex_handle(hCurr);
      v2 = ipMesh.to_vertex_handle(hCurr);
      v3 = ipMesh.to_vertex_handle(hNext);

      pt1 = ipMesh.point(v1);
      pt2 = ipMesh.point(v2);
      pt3 = ipMesh.point(v3);

      v21 = pt1 - pt2;
      v23 = pt3 - pt2;

      meshAngle[hCurr.idx()] = acos( v21 | v23 / (v21.norm()*v23.norm() ));
      meshCot[hCurr.idx()] = 1./tan(meshAngle[hCurr.idx()]);
    }

    //std::cout << "hCurr = " << hCurr << " v <" << v1<< "," << v2<< " > \n" << std::flush;
    //std::cout << "Angle = " << meshAngle[hCurr.idx()] << "\n"<< std::flush;
#ifdef _DEBUG_ANGLE_CALC_
    v4 = ipMesh.to_vertex_handle(hNext2);
    pt4 = ipMesh.point(v4);
    v34 = pt4 - pt3;
    double sum = acos( v21 | v23 / (v21.norm()*v23.norm() )) + acos( -v23 | v34 / (v23.norm()*v34.norm() )) + acos( v34 | v21 / (v21.norm()*v34.norm() ));
    if (meshAngle[hCurr.idx()] < minAng)
      minAng = meshAngle[hCurr.idx()];

    if (meshAngle[hCurr.idx()] > maxAng)
      maxAng = meshAngle[hCurr.idx()];

    if (sum < minSum)
      minSum = sum;

    if (sum > maxSum)
      maxSum = sum;

#endif
  }
#ifdef _DEBUG_ANGLE_CALC_
  std::cout << "Min Ang = " << minAng*180*7/22 << "\n" << std::flush;
  std::cout << "Max Ang = " << maxAng*180*7/22 << "\n" << std::flush;

  std::cout << "Min Sum = " << minSum*180*7/22 << "\n" << std::flush;
  std::cout << "Max Sum = " << maxSum*180*7/22 << "\n" << std::flush;
#endif
  //t.stop();
  //std::cout << "done (" << t.as_string() << ")\n";
}

void computeAreas(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  FN_TYPE *meshArea    = mStats.meshArea;
  FN_TYPE *areaPerVtx  = mStats.areaPerVtx;

  //OpenMesh::Utils::Timer t;
  OpenMesh::Vec3f pt1, pt2, pt3;
  OpenMesh::Vec3f v21, v23;

  std::cout << "Computing Area..." << std::flush;
  //t.start();

  double totalArea = 0;
  SimpleTriMesh::FaceIter fIter, fEnd(ipMesh.faces_end());
  for (fIter = ipMesh.faces_begin(); fIter != fEnd; ++fIter)
  {
    OpenMesh::VertexHandle v1,v2,v3;

    OpenMesh::FaceHandle fCurr;
    OpenMesh::HalfedgeHandle hCurr,hNext;

    fCurr = fIter.handle();
    hCurr = ipMesh.halfedge_handle(fCurr);
    hNext = ipMesh.next_halfedge_handle(hCurr);

    v1 = ipMesh.from_vertex_handle(hCurr);
    v2 = ipMesh.to_vertex_handle(hCurr);
    v3 = ipMesh.to_vertex_handle(hNext);

    pt1 = ipMesh.point(v1);
    pt2 = ipMesh.point(v2);
    pt3 = ipMesh.point(v3);

    v21 = pt1 - pt2;
    v23 = pt3 - pt2;

    v23 = v21 % v23;
    //std::cout << "hCurr = " << hCurr << "\n" << std::flush;
    meshArea[fCurr.idx()] =  v23.norm()/2;

    areaPerVtx[v1.idx()] += (meshArea[fCurr.idx()] /3);
    areaPerVtx[v2.idx()] += (meshArea[fCurr.idx()] /3);
    areaPerVtx[v3.idx()] += (meshArea[fCurr.idx()] /3);

    totalArea +=  meshArea[fCurr.idx()] ;
#ifdef _DEBUG_AREA_CALC_
    std::cout << "Area[ " << fCurr << "] = " <<meshArea[fCurr.idx()] << "\n" << std::flush;
#endif
  }

  //t.stop();
  //std::cout << "done (" << t.as_string() << ")\n";

  std::cout << "Total Area = " << totalArea <<  "\n" << std::flush;
}


void computeLaplacianWeights(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  FN_TYPE *meshCot    = mStats.meshCot;
  FN_TYPE *wHej       = mStats.wHej;
  FN_TYPE *areaPerVtx = mStats.areaPerVtx;
  FN_TYPE *wVtx       = mStats.wVtx;

  //OpenMesh::Utils::Timer t;

  std::cout << "compute laplacian weights..." << std::flush;
  //t.start();

  SimpleTriMesh::HalfedgeIter heIter, heEnd(ipMesh.halfedges_end());

  for (heIter = ipMesh.halfedges_begin(); heIter != heEnd; ++heIter)
  {
    OpenMesh::VertexHandle v1;
    OpenMesh::HalfedgeHandle hCurr, hNext, hOpp, hOppNext;


    hCurr = heIter.handle();
    hNext = ipMesh.next_halfedge_handle(hCurr);

    v1 = ipMesh.from_vertex_handle(hCurr);

    hOpp = ipMesh.opposite_halfedge_handle(hCurr);
    hOppNext = ipMesh.next_halfedge_handle(hOpp);

    //double divF = 0;
    double wg   = 0;
    if (!ipMesh.is_boundary(hCurr))
    {
      wg += meshCot[hNext.idx()];

      //divF += 1;
    }

    if (!ipMesh.is_boundary(hOpp))
    {
      wg += meshCot[hOppNext.idx()];

      //divF += 1;
    }

    wHej[hCurr.idx()] =  wg/ (2 * areaPerVtx[v1.idx()]);

    wVtx[v1.idx()] -= wHej[hCurr.idx()] ;
  }

  //t.stop();
  //std::cout << "done (" << t.as_string() << ")\n";

}
void computeAverageEdgeLength(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  //OpenMesh::Utils::Timer t;

  mStats.avgEdgeLen = 0;
  // compute average edge length
  std::cout << "compute average edge length..." << std::flush;
  //t.start();

  OpenMesh::Vec3f pt1, pt2;
  OpenMesh::Vec3f v21;

  SimpleTriMesh::HalfedgeIter heIter, heEnd(ipMesh.halfedges_end());
  for (heIter = ipMesh.halfedges_begin(); heIter != heEnd; ++heIter)
  {
    OpenMesh::VertexHandle v1, v2;
    OpenMesh::HalfedgeHandle hCurr;


    hCurr = heIter.handle();

    v1 = ipMesh.from_vertex_handle(hCurr);
    v2 = ipMesh.to_vertex_handle(hCurr);

    pt1 = ipMesh.point(v1);
    pt2 = ipMesh.point(v2);

    v21 = pt1 - pt2;

    mStats.avgEdgeLen += v21.norm();
  }

  mStats.avgEdgeLen /= ipMesh.n_halfedges();
  //t.stop();
 // std::cout << "done (" << t.as_string() << ")\n";
  std::cout << "Average Edge Length = " <<  mStats.avgEdgeLen << "\n";
}

void computeMaxEdgeLength(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  //OpenMesh::Utils::Timer t;

  mStats.maxEdgeLen = 0;
  // compute average edge length
  std::cout << "compute average edge length..." << std::flush;
  //t.start();

  OpenMesh::Vec3f pt1, pt2;
  OpenMesh::Vec3f v21;

  SimpleTriMesh::HalfedgeIter heIter, heEnd(ipMesh.halfedges_end());
  for (heIter = ipMesh.halfedges_begin(); heIter != heEnd; ++heIter)
  {
    OpenMesh::VertexHandle v1, v2;
    OpenMesh::HalfedgeHandle hCurr;


    hCurr = heIter.handle();

    v1 = ipMesh.from_vertex_handle(hCurr);
    v2 = ipMesh.to_vertex_handle(hCurr);

    pt1 = ipMesh.point(v1);
    pt2 = ipMesh.point(v2);

    v21 = pt1 - pt2;

    double cLen = v21.norm();

    if(cLen > mStats.maxEdgeLen)
    {
      mStats.maxEdgeLen = cLen;
    }
  }

  //t.stop();
  //std::cout << "done (" << t.as_string() << ")\n";
  std::cout << "Average Edge Length = " <<   mStats.maxEdgeLen << "\n";
}
void computeGradVectors(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
 // OpenMesh::Utils::Timer t;

  std::cout << "compute gradient vectors..." << std::flush;
 // t.start();

  OpenMesh::Vec3f pt1, pt2, pt3;
  OpenMesh::Vec3f v31, v12;

  OpenMesh::Vec3f fNorm;

  SimpleTriMesh::FaceIter fIter, fEnd(ipMesh.faces_end());
  for (fIter = ipMesh.faces_begin(); fIter != fEnd; ++fIter)
  {
    OpenMesh::VertexHandle v1,v2,v3;

    OpenMesh::FaceHandle fCurr;
    OpenMesh::HalfedgeHandle hCurr,hNext;

    fCurr = fIter.handle();
    hCurr = ipMesh.halfedge_handle(fCurr);
    hNext = ipMesh.next_halfedge_handle(hCurr);

    v1 = ipMesh.from_vertex_handle(hCurr);
    v2 = ipMesh.to_vertex_handle(hCurr);
    v3 = ipMesh.to_vertex_handle(hNext);

    pt1 = ipMesh.point(v1);
    pt2 = ipMesh.point(v2);
    pt3 = ipMesh.point(v3);

    v31 = pt1 - pt3;
    v12 = pt2 - pt1;

    fNorm = ipMesh.calc_face_normal(fCurr);
    fNorm = fNorm.normalize();

    mStats.gradVec12[fCurr.idx()] = fNorm % v31;
    mStats.gradVec12[fCurr.idx()] /= (2*mStats.meshArea[fCurr.idx()]);

    mStats.gradVec13[fCurr.idx()] = fNorm % v12;
    mStats.gradVec13[fCurr.idx()] /= (2*mStats.meshArea[fCurr.idx()]);
  }
 // std::cout << "done (" << t.as_string() << ")\n";

}

void computeHEGradVectors(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
 // OpenMesh::Utils::Timer t;

  std::cout << "compute half edge gradient vectors..." << std::flush;
//  t.start();

  OpenMesh::Vec3f pt1, pt2, pt3;
  OpenMesh::Vec3f v12;

  OpenMesh::Vec3f fNorm;

  SimpleTriMesh::HalfedgeIter heIter, heEnd(ipMesh.halfedges_end());

  for (heIter = ipMesh.halfedges_end(); heIter != heIter; ++heIter)
  {
    OpenMesh::VertexHandle v1,v2,v3;

    OpenMesh::FaceHandle fCurr;
    OpenMesh::HalfedgeHandle hCurr;

    hCurr = heIter.handle();
    fCurr = ipMesh.face_handle(hCurr);

    v1 = ipMesh.from_vertex_handle(hCurr);
    v2 = ipMesh.to_vertex_handle(hCurr);

    pt1 = ipMesh.point(v1);
    pt2 = ipMesh.point(v2);

    v12 = pt2 - pt1;

    fNorm = ipMesh.calc_face_normal(fCurr);
    fNorm = fNorm.normalize();
    mStats.gradVecHE[hCurr.idx()] = fNorm % v12;
    mStats.gradVecHE[hCurr.idx()] /= (2*mStats.meshArea[fCurr.idx()]);
  }

//  std::cout << "done (" << t.as_string() << ")\n";
}

void computeNormalizedAngleWeights(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  SimpleTriMesh::VertexIter vI, vEnd(ipMesh.vertices_end());
  for(vI = ipMesh.vertices_begin(); vI != vEnd; ++vI)
  {
    //uint hIdx;
    //uint vIdx = vI.handle().idx();
    //uint fIdx;

    SimpleTriMesh::VertexOHalfedgeIter hIter;
    FN_TYPE wg = 0;

    for (hIter = ipMesh.voh_iter(vI.handle()); hIter; ++hIter)
    {

      OpenMesh::HalfedgeHandle hCurr, hPrev;

      hCurr = hIter.handle();

      if(!ipMesh.is_boundary(hCurr))
      {
        hPrev = ipMesh.prev_halfedge_handle(hCurr);
        wg += mStats.meshAngle[hPrev.idx()];
      } //end if
    } // end for

    for (hIter = ipMesh.voh_iter(vI.handle()); hIter; ++hIter)
    {

      OpenMesh::HalfedgeHandle hCurr, hPrev;

      hCurr = hIter.handle();

      hPrev = ipMesh.prev_halfedge_handle(hCurr);
      if(!ipMesh.is_boundary(hCurr))
      {
        if (wg != 0)
        {
          mStats.meshWij[hPrev.idx()] =  mStats.meshAngle[hPrev.idx()] / wg;
        } else {
          mStats.meshWij[hPrev.idx()] =  0;
        }
      } else
      {
          mStats.meshWij[hPrev.idx()] =  0;
      }
    } // end for
  } // end vI
}

void computeHijVecForJacobian(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
 //OpenMesh::Utils::Timer t;

  std::cout << "compute Hij vectors for the Jacobian..." << std::flush;
 // t.start();

  OpenMesh::Vec3f pt1, pt2, pt3;
  OpenMesh::Vec3f v12;

  OpenMesh::Vec3f fNorm;

  SimpleTriMesh::HalfedgeIter heIter, heEnd(ipMesh.halfedges_end());

  for (heIter = ipMesh.halfedges_end(); heIter != heIter; ++heIter)
  {

    OpenMesh::HalfedgeHandle hCurr, hOpp, hPrev, hNextOfOpp;
    //long fLeftIdx, fRightIdx;

    hCurr = heIter.handle();
    hOpp  = ipMesh.opposite_halfedge_handle(hCurr);

    mStats.vecHij[hCurr.idx()][0] = 0;
    mStats.vecHij[hCurr.idx()][1] = 0;
    mStats.vecHij[hCurr.idx()][2] = 0;

    if(! ipMesh.is_boundary(hCurr))
    {
      hPrev = ipMesh.prev_halfedge_handle(hCurr);
      mStats.vecHij[hCurr.idx()] +=  mStats.gradVecHE[ hPrev.idx()] * mStats.meshWij[hPrev.idx()];
    }

    if(! ipMesh.is_boundary(hOpp))
    {
      hNextOfOpp = ipMesh.next_halfedge_handle(hOpp);
      mStats.vecHij[hCurr.idx()] +=  mStats.gradVecHE[ hNextOfOpp.idx()] * mStats.meshWij[hOpp.idx()];
    }
  }
 // std::cout << "done (" << t.as_string() << ")\n";
}

void computeVtxGVecForJacobian(SimpleTriMesh &ipMesh, MeshStats &mStats)
{
  SimpleTriMesh::VertexIter vI, vEnd(ipMesh.vertices_end());

  for(vI = ipMesh.vertices_begin(); vI != vEnd; ++vI)
  {
    //uint hIdx;
    uint vIdx = vI.handle().idx();

    SimpleTriMesh::VertexOHalfedgeIter hIter;


    mStats.vecVtxG[vIdx][0] = 0;
    mStats.vecVtxG[vIdx][1] = 0;
    mStats.vecVtxG[vIdx][2] = 0;

    for (hIter = ipMesh.voh_iter(vI.handle()); hIter; ++hIter)
    {
      OpenMesh::HalfedgeHandle hCurr, hNext, hPrev;

      hCurr = hIter.handle();

      if(!ipMesh.is_boundary(hCurr))
      {
        hNext = ipMesh.next_halfedge_handle(hCurr);
        hPrev = ipMesh.prev_halfedge_handle(hCurr);

        mStats.vecVtxG[vIdx] +=  mStats.gradVecHE[hNext.idx()] * mStats.meshWij[hPrev.idx()];
      } //end if
    } // end for
  } // end vI
}

void computeHijLambdaForJacobian(SimpleTriMesh &ipMesh, MeshStats &mStats)
{

  SimpleTriMesh::VertexIter vI, vEnd(ipMesh.vertices_end());

  for(vI = ipMesh.vertices_begin(); vI != vEnd; ++vI)
  {
    uint hIdx;
    uint vIdx = vI.handle().idx();

    SimpleTriMesh::VertexOHalfedgeIter hIter;


    for (hIter = ipMesh.voh_iter(vI.handle()); hIter; ++hIter)
    {

      hIdx = hIter.handle().idx();

      mStats.meshLamdaIj[vIdx] = mStats.vecVtxG[vIdx] | mStats.vecHij[hIdx];
    } // end for
  } // end vI
}




