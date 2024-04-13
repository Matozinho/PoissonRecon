// ##########################################################################
// #                                                                        #
// #               CLOUDCOMPARE WRAPPER: PoissonReconLib                    #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #               COPYRIGHT: Daniel Girardeau-Montaut                      #
// #                                                                        #
// ##########################################################################

#include "PoissonReconLib.h"

// PoissonRecon
#include "../Src/DataStream.h"
#include "../Src/FEMTree.h"
// #include "../Src/Image.h"
// #include "../Src/MyMiscellany.h"
// #include "../Src/PPolynomial.h"
// #include "../Src/RegularGrid.h"
#include "../Src/VertexFactory.h"

// Local
#include "BSplineData.h"
#include "PlyFile.h"
#include "PointData.h"
#include "PointDataStream.imp.h"
#include "confia.h"

// System
#include <cassert>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {
// The order of the B-Spline used to splat in data for color interpolation
constexpr int DATA_DEGREE = 0;
// The order of the B-Spline used to splat in the weights for density estimation
constexpr int WEIGHT_DEGREE = 2;
// The order of the B-Spline used to splat in the normals for constructing the
// Laplacian constraints
constexpr int NORMAL_DEGREE = 2;
// The default finite-element degree
constexpr int DEFAULT_FEM_DEGREE = 1;
// The dimension of the system
constexpr int DIMENSION = 3;

inline float ComputeNorm(const float vec[3]) {
  return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

inline double ComputeNorm(const double vec[3]) {
  return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}
} // namespace

int PoissonReconLib::Parameters::GetMaxThreadCount() {
#ifdef WITH_OPENMP
  return omp_get_num_procs();
#else
  return std::thread::hardware_concurrency();
#endif
}

PoissonReconLib::Parameters::Parameters() : threads(GetMaxThreadCount()) {}

template <unsigned int Dim, class Real> struct FEMTreeProfiler {
  FEMTree<Dim, Real> &tree;
  double t;

  FEMTreeProfiler(FEMTree<Dim, Real> &t) : tree(t) {}
  void start(void) { t = Time(), FEMTree<Dim, Real>::ResetLocalMemoryUsage(); }
  void dumpOutput(const char *header) const {
    FEMTree<Dim, Real>::MemoryUsage();
    // if (header) {
    //	utility::LogDebug("{} {} (s), {} (MB) / {} (MB) / {} (MB)", header,
    //		Time() - t,
    //		FEMTree<Dim, Real>::LocalMemoryUsage(),
    //		FEMTree<Dim, Real>::MaxMemoryUsage(),
    //		MemoryInfo::PeakMemoryUsageMB());
    // }
    // else {
    //	utility::LogDebug("{} (s), {} (MB) / {} (MB) / {} (MB)", Time() - t,
    //		FEMTree<Dim, Real>::LocalMemoryUsage(),
    //		FEMTree<Dim, Real>::MaxMemoryUsage(),
    //		MemoryInfo::PeakMemoryUsageMB());
    // }
  }
};

template <unsigned int Dim, typename Real> struct ConstraintDual {
  Real target, weight;
  ConstraintDual(Real t, Real w) : target(t), weight(w) {}
  CumulativeDerivativeValues<Real, Dim, 0>
  operator()(const Point<Real, Dim> &p) const {
    return CumulativeDerivativeValues<Real, Dim, 0>(target * weight);
  };
};

template <unsigned int Dim, typename Real> struct SystemDual {
  SystemDual(Real w) : weight(w) {}
  CumulativeDerivativeValues<Real, Dim, 0>
  operator()(const Point<Real, Dim> &p,
             const CumulativeDerivativeValues<Real, Dim, 0> &dValues) const {
    return dValues * weight;
  };

  CumulativeDerivativeValues<double, Dim, 0>
  operator()(const Point<Real, Dim> &p,
             const CumulativeDerivativeValues<double, Dim, 0> &dValues) const {
    return dValues * weight;
  };

  Real weight;
};

template <unsigned int Dim> struct SystemDual<Dim, double> {
  typedef double Real;

  SystemDual(Real w) : weight(w) {}
  CumulativeDerivativeValues<Real, Dim, 0>
  operator()(const Point<Real, Dim> &p,
             const CumulativeDerivativeValues<Real, Dim, 0> &dValues) const {
    return dValues * weight;
  };

  Real weight;
};

template <typename Real>
class MeshCollector : public PoissonReconLib::IMesh<Real> {
public:
  std::vector<Real> vertices;
  std::vector<Real> normals;
  std::vector<Real> colors;
  std::vector<double> densities;
  std::vector<size_t> triangles;

  void addVertex(const Real *coords) override {
    vertices.insert(vertices.end(), coords, coords + 3);
  }

  void addNormal(const Real *coords) override {
    normals.insert(normals.end(), coords, coords + 3);
  }

  void addColor(const Real *rgb) override {
    colors.insert(colors.end(), rgb, rgb + 3);
  }

  void addDensity(double d) override { densities.push_back(d); }

  void addTriangle(size_t i1, size_t i2, size_t i3) override {
    triangles.push_back(i1);
    triangles.push_back(i2);
    triangles.push_back(i3);
  }
};

template <typename VertexFactory, typename Index, class Real, int Dim>
std::shared_ptr<PoissonReconLib::IMesh<Real>>
GetMesh(const VertexFactory &vFactory, size_t vertexNum,
        InputDataStream<typename VertexFactory::VertexType> &vertexStream) {
  auto out_mesh = std::make_shared<MeshCollector<Real>>();
  vertexStream.reset();

  // Process vertices
  if (vFactory.isStaticallyAllocated()) {
    for (size_t i = 0; i < vertexNum; ++i) {
      typename VertexFactory::VertexType vertex = vFactory();
      if (!vertexStream.read(vertex)) {
        throw std::runtime_error("Failed to read vertex " + std::to_string(i) +
                                 " / " + std::to_string(vertexNum));
      }
      out_mesh->addVertex(vertex.coords);
      out_mesh->addNormal(vertex.normal.coords);
      out_mesh->addColor(vertex.color);
    }
  } else {
    Pointer(char) buffer = NewPointer<char>(vFactory.bufferSize());
    for (size_t i = 0; i < vertexNum; ++i) {
      typename VertexFactory::VertexType vertex = vFactory();
      if (!vertexStream.read(vertex)) {
        throw std::runtime_error("Failed to read vertex " + std::to_string(i) +
                                 " / " + std::to_string(vertexNum));
      }
      vFactory.toBuffer(vertex, buffer);
      // Process buffer to extract vertex data and add to out_mesh
    }
    DeletePointer(buffer);
  }

  return out_mesh;
}

template <typename Real, unsigned int Dim, unsigned int FEMSig,
          bool HasGradients, bool HasDensity>
void WriteMesh(Reconstructor::Implicit<Real, Dim, FEMSig> &implicit,
               const Reconstructor::LevelSetExtractionParameters &meParams) {
  // A description of the output vertex information
  using VInfo =
      Reconstructor::OutputVertexInfo<Real, Dim, HasGradients, HasDensity>;

  // A factory generating the output vertices
  using Factory = typename VInfo::Factory;
  Factory factory = VInfo::GetFactory();

  // A backing stream for the vertices
  Reconstructor::OutputInputFactoryTypeStream<Factory> vertexStream(
      factory, false, false, std::string("v_"));
  Reconstructor::OutputInputFaceStream<Dim - 1> faceStream(false, true,
                                                           std::string("f_"));

  // The wrapper converting native to output types
  typename VInfo::StreamWrapper _vertexStream(vertexStream, factory());

  // Extract the level set
  implicit.extractLevelSet(_vertexStream, faceStream, meParams);

  // Write the mesh to a .ply file
  std::vector<std::string> noComments;
  PLY::Write<Factory, node_index_type, Real, Dim>(
      "confia.ply", factory, vertexStream.size(), faceStream.size(),
      vertexStream, faceStream, PLY_BINARY_NATIVE, noComments);
}

template <typename Real, unsigned int Dim, unsigned int FEMSig,
          typename AuxDataFactory, bool HasGradients, bool HasDensity>
void WriteMeshWithData(
    const AuxDataFactory &auxDataFactory,
    Reconstructor::Implicit<Real, Dim, FEMSig,
                            typename AuxDataFactory::VertexType> &implicit,
    const Reconstructor::LevelSetExtractionParameters &meParams, bool hasColors,
    const PoissonReconLib::IMesh<Real> &out_mesh) {
  // A description of the output vertex information
  using VInfo =
      Reconstructor::OutputVertexWithDataInfo<Real, Dim, AuxDataFactory,
                                              HasGradients, HasDensity>;

  // A factory generating the output vertices
  using Factory = typename VInfo::Factory;
  Factory factory = VInfo::GetFactory(auxDataFactory);

  // A backing stream for the vertices
  Reconstructor::OutputInputFactoryTypeStream<Factory> vertexStream(
      factory, false, false, std::string("v_"));
  Reconstructor::OutputInputFaceStream<Dim - 1> faceStream(false, true,
                                                           std::string("f_"));

  {
    // The wrapper converting native to output types
    typename VInfo::StreamWrapper _vertexStream(vertexStream, factory());

    // Extract the level set
    implicit.extractLevelSet(_vertexStream, faceStream, meParams);
  }

  std::vector<std::string> noComments;

  Confia::Write_Confia<Factory, node_index_type, Real, Dim>(
      factory, vertexStream.size(), faceStream.size(), vertexStream, faceStream,
      HasDensity, hasColors, out_mesh);
}

template <class Real, unsigned int Dim, unsigned int FEMSig,
          typename AuxDataFactory>
bool Execute(const PoissonReconLib::ICloud<Real> &inCloud,
             const PoissonReconLib::IMesh<Real> &out_mesh,
             const PoissonReconLib::Parameters &params,
             const AuxDataFactory &auxDataFactory) {
  static const bool HasAuxData =
      !std::is_same<AuxDataFactory, VertexFactory::EmptyFactory<Real>>::value;
  ///////////////
  // Types --> //
  typedef IsotropicUIntPack<Dim, FEMSig> Sigs;
  using namespace VertexFactory;

  // The factory for constructing an input sample's data
  typedef typename std::conditional<
      HasAuxData, Factory<Real, NormalFactory<Real, Dim>, AuxDataFactory>,
      NormalFactory<Real, Dim>>::type InputSampleDataFactory;

  // The factory for constructing an input sample
  typedef Factory<Real, PositionFactory<Real, Dim>, InputSampleDataFactory>
      InputSampleFactory;

  typedef InputDataStream<typename InputSampleFactory::VertexType>
      InputPointStream;

  // The type storing the reconstruction solution (depending on whether
  // auxiliary data is provided or not)
  using Implicit = typename std::conditional<
      HasAuxData,
      Reconstructor::Poisson::Implicit<Real, Dim, FEMSig,
                                       typename AuxDataFactory::VertexType>,
      Reconstructor::Poisson::Implicit<Real, Dim, FEMSig>>::type;
  // <-- Types //
  ///////////////

  double startTime = Time();

  InputSampleFactory *_inputSampleFactory;
  if constexpr (HasAuxData)
    _inputSampleFactory = new InputSampleFactory(
        VertexFactory::PositionFactory<Real, Dim>(),
        InputSampleDataFactory(VertexFactory::NormalFactory<Real, Dim>(),
                               auxDataFactory));
  else
    _inputSampleFactory =
        new InputSampleFactory(VertexFactory::PositionFactory<Real, Dim>(),
                               VertexFactory::NormalFactory<Real, Dim>());
  InputSampleFactory &inputSampleFactory = *_inputSampleFactory;
  XForm<Real, Dim + 1> toModel = XForm<Real, Dim + 1>::Identity();
  InputPointStream *pointStream;

  // Get the point stream
  pointStream = new PointInputDataStream<Real, InputSampleFactory>(
      inCloud, inputSampleFactory);

  // define the envelope
  typename Reconstructor::Poisson::EnvelopeMesh<Real, Dim> *envelopeMesh = NULL;
  // {
  //   envelopeMesh =
  //       new typename Reconstructor::Poisson::EnvelopeMesh<Real, Dim>();

  //   Confia::generateMesh<PositionFactory<Real, Dim>, Real, Dim>(
  //       PositionFactory<Real, Dim>(), inCloud, envelopeMesh);

  //   // print all the vertices of the envelopeMesh
  //   for (size_t i = 0; i < envelopeMesh->vertices.size(); i++) {
  //     std::cout << "Vertex[" << i << "]: " << envelopeMesh->vertices[i]
  //               << std::endl;
  //   }
  //   std::cout << "Simplicies[0] " << envelopeMesh->simplices[0][0] << " "
  //             << envelopeMesh->simplices[0][1] << " "
  //             << envelopeMesh->simplices[0][2] << std::endl;
  // }

  // A wrapper class to realize InputDataStream< SampleType > as an
  // InputSampleStream
  struct _InputSampleStream
      : public Reconstructor::InputSampleStream<Real, Dim> {
    typedef Reconstructor::Normal<Real, Dim> DataType;
    typedef VectorTypeUnion<Real, Reconstructor::Position<Real, Dim>, DataType>
        SampleType;
    typedef InputDataStream<SampleType> _InputPointStream;
    _InputPointStream &pointStream;
    SampleType scratch;
    _InputSampleStream(_InputPointStream &pointStream)
        : pointStream(pointStream) {
      scratch = SampleType(Reconstructor::Position<Real, Dim>(),
                           Reconstructor::Normal<Real, Dim>());
    }
    void reset(void) { pointStream.reset(); }
    bool base_read(Reconstructor::Position<Real, Dim> &p,
                   Reconstructor::Normal<Real, Dim> &n) {
      bool ret = pointStream.read(scratch);
      if (ret)
        p = scratch.template get<0>(), n = scratch.template get<1>();
      return ret;
    }
  };

  // A wrapper class to realize InputDataStream< SampleType > as an
  // InputSampleWithDataStream
  struct _InputSampleWithDataStream
      : public Reconstructor::InputSampleWithDataStream<
            Real, Dim, typename AuxDataFactory::VertexType> {
    typedef VectorTypeUnion<Real, Reconstructor::Normal<Real, Dim>,
                            typename AuxDataFactory::VertexType>
        DataType;
    typedef VectorTypeUnion<Real, Reconstructor::Position<Real, Dim>, DataType>
        SampleType;
    typedef InputDataStream<SampleType> _InputPointStream;
    _InputPointStream &pointStream;
    SampleType scratch;
    _InputSampleWithDataStream(_InputPointStream &pointStream,
                               typename AuxDataFactory::VertexType zero)
        : Reconstructor::InputSampleWithDataStream<
              Real, Dim, typename AuxDataFactory::VertexType>(zero),
          pointStream(pointStream) {
      scratch = SampleType(Reconstructor::Position<Real, Dim>(),
                           DataType(Reconstructor::Normal<Real, Dim>(), zero));
    }
    void reset(void) { pointStream.reset(); }
    bool base_read(Reconstructor::Position<Real, Dim> &p,
                   Reconstructor::Normal<Real, Dim> &n,
                   typename AuxDataFactory::VertexType &d) {
      bool ret = pointStream.read(scratch);
      if (ret)
        p = scratch.template get<0>(),
        n = scratch.template get<1>().template get<0>(),
        d = scratch.template get<1>().template get<1>();
      return ret;
    }
  };

  Reconstructor::Poisson::SolutionParameters<Real> solParams;

  // Map the fields from params to solParams
  solParams.scale = static_cast<float>(params.scale);
  solParams.confidence = static_cast<float>(params.normalConfidence);
  solParams.confidenceBias = static_cast<float>(params.normalConfidenceBias);
  solParams.samplesPerNode = static_cast<float>(params.samplesPerNode);
  solParams.cgSolverAccuracy = static_cast<float>(params.cgAccuracy);
  solParams.depth = static_cast<unsigned int>(params.depth);
  solParams.baseDepth = static_cast<unsigned int>(params.baseDepth);
  solParams.fullDepth = static_cast<unsigned int>(params.fullDepth);
  solParams.iters = static_cast<unsigned int>(params.iters);
  solParams.baseVCycles = static_cast<unsigned int>(params.baseVCycles);
  solParams.pointWeight = static_cast<float>(params.pointWeight);
  solParams.outputDensity = static_cast<float>(params.density);

  Implicit *implicit = NULL;

  if constexpr (HasAuxData) {
    _InputSampleWithDataStream sampleStream(*pointStream, auxDataFactory());

    implicit = new typename Reconstructor::Poisson::Implicit<
        Real, Dim, FEMSig, typename AuxDataFactory::VertexType>(
        sampleStream, solParams, envelopeMesh);
  } else {
    _InputSampleStream sampleStream(*pointStream);
    implicit = new typename Reconstructor::Poisson::Implicit<Real, Dim, FEMSig>(
        sampleStream, solParams);
  }

  delete pointStream;
  delete _inputSampleFactory;
  delete envelopeMesh;

  if constexpr (HasAuxData)
    if (implicit->auxData)
      implicit->weightAuxDataByDepth((Real)32.f);

  Reconstructor::LevelSetExtractionParameters meParams;

  if constexpr (HasAuxData) {
    std::cout << "Writing mesh with data" << std::endl;
    if (solParams.outputDensity) {
      std::cout << "Writing mesh WITH density" << std::endl;
      WriteMeshWithData<Real, Dim, FEMSig, AuxDataFactory, false, true>(
          auxDataFactory, *implicit, meParams, params.withColors, out_mesh);
    } else {
      std::cout << "Writing mesh WITHOUT density" << std::endl;
      WriteMeshWithData<Real, Dim, FEMSig, AuxDataFactory, false, false>(
          auxDataFactory, *implicit, meParams, params.withColors, out_mesh);
    }
  } else {
    std::cout << "Writing mesh" << std::endl;
    WriteMesh<Real, Dim, FEMSig, false, false>(*implicit, meParams);
  }

  delete implicit;
  return true;
}

bool PoissonReconLib::Reconstruct(const Parameters &params,
                                  const ICloud<float> &inCloud,
                                  IMesh<float> &outMesh) {
  if (!inCloud.hasNormals()) {
    // we need normals
    return false;
  }

#ifdef WITH_OPENMP
  ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP,
                   params.threads);
#else
  ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL,
                   params.threads);
#endif
  bool success = false;

  switch (params.boundary) {
  case Parameters::FREE:
    if (params.withColors)
      success =
          Execute<float, 3, FEMDegreeAndBType<1, BOUNDARY_FREE>::Signature>(
              inCloud, outMesh, params,
              VertexFactory::RGBColorFactory<float>());
    else
      success =
          Execute<float, 3, FEMDegreeAndBType<1, BOUNDARY_FREE>::Signature>(
              inCloud, outMesh, params, VertexFactory::EmptyFactory<float>());
    break;

  case Parameters::DIRICHLET:
    if (params.withColors)
      success = Execute<float, 3,
                        FEMDegreeAndBType<1, BOUNDARY_DIRICHLET>::Signature>(
          inCloud, outMesh, params, VertexFactory::RGBColorFactory<float>());
    else
      success = Execute<float, 3,
                        FEMDegreeAndBType<1, BOUNDARY_DIRICHLET>::Signature>(
          inCloud, outMesh, params, VertexFactory::EmptyFactory<float>());
    break;

  case Parameters::NEUMANN:
    if (params.withColors)
      success =
          Execute<float, 3, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature>(
              inCloud, outMesh, params,
              VertexFactory::RGBColorFactory<float>());
    else
      success =
          Execute<float, 3, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature>(
              inCloud, outMesh, params, VertexFactory::EmptyFactory<float>());
    break;
  default:
    assert(false);
    break;
  }

  ThreadPool::Terminate();

  return success;
}

bool PoissonReconLib::Reconstruct(const Parameters &params,
                                  const ICloud<double> &inCloud,
                                  IMesh<double> &outMesh) {
  if (!inCloud.hasNormals()) {
    // we need normals
    return false;
  }

#ifdef WITH_OPENMP
  ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP,
                   std::thread::hardware_concurrency());
#else
  ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL,
                   std::thread::hardware_concurrency());
#endif

  // PointStream<double> pointStream(inCloud);

  bool success = false;

  switch (params.boundary) {
  case Parameters::FREE:
    if (params.withColors)
      success =
          Execute<double, 3, FEMDegreeAndBType<1, BOUNDARY_FREE>::Signature>(
              inCloud, outMesh, params,
              VertexFactory::RGBColorFactory<double>());
    else
      success =
          Execute<double, 3, FEMDegreeAndBType<1, BOUNDARY_FREE>::Signature>(
              inCloud, outMesh, params, VertexFactory::EmptyFactory<double>());
    break;
  case Parameters::DIRICHLET:
    if (params.withColors)
      success = Execute<double, 3,
                        FEMDegreeAndBType<1, BOUNDARY_DIRICHLET>::Signature>(
          inCloud, outMesh, params, VertexFactory::RGBColorFactory<double>());
    else
      success = Execute<double, 3,
                        FEMDegreeAndBType<1, BOUNDARY_DIRICHLET>::Signature>(
          inCloud, outMesh, params, VertexFactory::EmptyFactory<double>());
    break;
  case Parameters::NEUMANN:
    if (params.withColors)
      success =
          Execute<double, 3, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature>(
              inCloud, outMesh, params,
              VertexFactory::RGBColorFactory<double>());
    else
      success =
          Execute<double, 3, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature>(
              inCloud, outMesh, params, VertexFactory::EmptyFactory<double>());
    break;
  default:
    assert(false);
    break;
  }

  ThreadPool::Terminate();

  return success;
}
