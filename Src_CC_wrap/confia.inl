#include "../Src/PlyFile.h"
#include "Array.h"
#include "PoissonReconLib.h"
#include "Reconstructors.h"
#include <iostream>
#include <vector>
namespace Confia {
template <typename Real>
void findBoundingBox(const PoissonReconLib::ICloud<Real> &inCloud,
                     Real bbox[8][3]) {
  // Initialize min and max values to the first point's coordinates
  Real minX, minY, minZ, maxX, maxY, maxZ;
  Real firstPoint[3];
  inCloud.getPoint(0, firstPoint);
  minX = maxX = firstPoint[0];
  minY = maxY = firstPoint[1];
  minZ = maxZ = firstPoint[2];

  // Iterate through all points to find min and max values
  for (size_t i = 1; i < inCloud.size(); ++i) {
    Real coords[3];
    inCloud.getPoint(i, coords);

    if (coords[0] < minX)
      minX = coords[0];
    if (coords[0] > maxX)
      maxX = coords[0];
    if (coords[1] < minY)
      minY = coords[1];
    if (coords[1] > maxY)
      maxY = coords[1];
    if (coords[2] < minZ)
      minZ = coords[2];
    if (coords[2] > maxZ)
      maxZ = coords[2];
  }

  // Assign the 8 bounding box points
  bbox[0][0] = minX;
  bbox[0][1] = minY;
  bbox[0][2] = minZ; // Min corner
  bbox[1][0] = maxX;
  bbox[1][1] = minY;
  bbox[1][2] = minZ;
  bbox[2][0] = minX;
  bbox[2][1] = maxY;
  bbox[2][2] = minZ;
  bbox[3][0] = maxX;
  bbox[3][1] = maxY;
  bbox[3][2] = minZ;
  bbox[4][0] = minX;
  bbox[4][1] = minY;
  bbox[4][2] = maxZ;
  bbox[5][0] = maxX;
  bbox[5][1] = minY;
  bbox[5][2] = maxZ;
  bbox[6][0] = minX;
  bbox[6][1] = maxY;
  bbox[6][2] = maxZ;
  bbox[7][0] = maxX;
  bbox[7][1] = maxY;
  bbox[7][2] = maxZ; // Max corner
}

template <typename VertexFactory, class Real, int Dim>
void generateMesh(
    const VertexFactory &vFactory, const PoissonReconLib::ICloud<Real> &inCloud,
    Reconstructor::Poisson::EnvelopeMesh<Real, Dim> *envelopeMesh) {

  Real bbox[8][3];

  findBoundingBox(inCloud, bbox);

  typename VertexFactory::VertexType vertex = vFactory();

  for (size_t i = 0; i < 8; i++) {
    char *buffer = reinterpret_cast<char *>(&bbox[i]);

    vFactory.fromBuffer(buffer, vertex);
    envelopeMesh->vertices.push_back(vertex);
  }

  std::vector<std::vector<int>> simplices = {
      {0, 1, 2}, {0, 2, 3}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
      {2, 3, 7}, {2, 7, 6}, {1, 2, 6}, {1, 6, 5}, {0, 3, 7}, {0, 7, 4}};

  for (size_t i = 0; i < simplices.size(); i++) {
    envelopeMesh->simplices.resize(simplices.size());
    for (size_t j = 0; j < simplices[i].size(); j++) {
      envelopeMesh->simplices[i][j] = simplices[i][j];
    }
  }
}

template <typename VertexFactory, typename Index, class Real, int Dim,
          typename OutputIndex, bool UseCharIndex>
void Write_Confia(
    const VertexFactory &vFactory, size_t vertexNum, size_t polygonNum,
    InputDataStream<typename VertexFactory::VertexType> &vertexStream,
    InputDataStream<std::vector<Index>> &polygonStream, bool hasDensity,
    bool hasColors, const PoissonReconLib::IMesh<Real> &out_mesh) {

  vertexStream.reset();
  polygonStream.reset();

  size_t numFloats = vFactory.bufferSize() / sizeof(Real);
  Pointer(char) buffer = NewPointer<char>(vFactory.bufferSize());

  for (size_t i = 0; i < vertexNum; i++) {
    typename VertexFactory::VertexType vertex = vFactory();

    if (!vertexStream.read(vertex))
      ERROR_OUT("Failed to read vertex ", i, " / ", vertexNum);

    vFactory.toBuffer(vertex, buffer);

    // Cast the buffer to a float pointer
    Real *floatArray = reinterpret_cast<Real *>(buffer);

    // the order of the buffer is x, y, z, density, r, g, b if it have density
    // and colors setted
    auto position = floatArray;
    double density = hasDensity ? static_cast<double>(*(position + Dim)) : 0.0;
    // if hasDensity, increment the position pointer
    auto color =
        hasColors ? hasDensity ? position + Dim + 1 : position + Dim : nullptr;

    out_mesh.addVertex(floatArray);
    if (hasColors)
      out_mesh.addColor(color);
    if (hasDensity)
      out_mesh.addDensity(density);
  }
  DeletePointer(buffer);

  // write faces
  std::vector<Index> polygon;
  for (size_t i = 0; i < polygonNum; i++) {
    if (!polygonStream.read(polygon))
      ERROR_OUT("Failed to read polygon ", i, " / ", polygonNum);

    if (polygon.size() != 3)
      ERROR_OUT("Face size not supported");

    out_mesh.addTriangle(polygon[0], polygon[1], polygon[2]);
  } // for, write faces
}
} // namespace Confia
