
#ifndef CONFIA_INCLUDED
#define CONFIA_INCLUDED

#include "../Src/Ply.h"
#include "../Src/Reconstructors.h"
#include "PoissonReconLib.h"
#include <string>
#include <vector>

namespace Confia {
template <typename VertexFactory, class Real, int Dim>
void generateMesh(const VertexFactory &vFactory,
                  const PoissonReconLib::ICloud<Real> &inCloud,
                  Reconstructor::Poisson::EnvelopeMesh<Real, Dim> envelopeMesh);

// PLY write mesh functionality
template <typename VertexFactory, typename Index, class Real, int Dim,
          typename OutputIndex = int, bool UseCharIndex = false>
void Write_Confia(
    const VertexFactory &vFactory, size_t vertexNum, size_t polygonNum,
    InputDataStream<typename VertexFactory::VertexType> &vertexStream,
    InputDataStream<std::vector<Index>> &polygonStream, bool hasDensity,
    bool hasColors, PoissonReconLib::IMesh<Real> &out_mesh);
} // namespace Confia
#include "confia.inl"
#endif // PLY_INCLUDED