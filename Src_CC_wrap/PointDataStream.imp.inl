#include "PointDataStream.imp.h"

template <class Real, typename Factory>
PointInputDataStream<Real, Factory>::PointInputDataStream(
    const PoissonReconLib::ICloud<Real> &cloud, const Factory &factory)
    : _factory(factory), _cloud(cloud), _currentIndex(0) {}

template <class Real, typename Factory> void PointInputDataStream<Real, Factory>::reset() {
  this->_currentIndex = 0;
}

template <class Real, typename Factory>
bool PointInputDataStream<Real, Factory>::base_read(Data &d) {
  // std::cout << "================================" << std::endl;
  // std::cout << "PointInputDataStream<Factory>::base_read" << std::endl;
  // std::cout << "Cloud size: " << this->_cloud.size() << std::endl;

  if (_currentIndex >= _cloud.size())
    return false;

  Real position[3] = {0, 0, 0};
  Real normal[3] = {0, 0, 0};
  Real color[3] = {0, 0, 0};
  std::vector<char> buffer;

  // Add position
  _cloud.getPoint(_currentIndex, position);
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(position),
                reinterpret_cast<const char *>(position) + sizeof(position));

  if (_cloud.hasNormals()) {
    _cloud.getNormal(_currentIndex, normal);
    buffer.insert(buffer.end(), reinterpret_cast<const char *>(normal),
                  reinterpret_cast<const char *>(normal) + sizeof(Real) * 3);
  }

  if (_cloud.hasColors()) {
    _cloud.getColor(_currentIndex, color);
    buffer.insert(buffer.end(), reinterpret_cast<const char *>(color),
                  reinterpret_cast<const char *>(color) + sizeof(Real) * 3);
  }

  // Convert to const char*
  const char *concatenated = buffer.data();

  _factory.fromBuffer(concatenated, d);

  // // print the position
  // std::cout << "Position: ";
  // for (int i = 0; i < 3; i++) {
  //   std::cout << position[i] << " ";
  // }
  // std::cout << std::endl;

  // // print the normal
  // std::cout << "Normal: ";
  // for (int i = 0; i < 3; i++) {
  //   std::cout << normal[i] << " ";
  // }
  // std::cout << std::endl;

  // // print the color
  // std::cout << "Color: ";
  // for (int i = 0; i < 3; i++) {
  //   std::cout << color[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "d.template get<0>: " << d.template get<0>() << std::endl;
  // std::cout << "d.template get<1>: " << d.template get<1>() << std::endl;
  // std::cout << "================================" << std::endl;

  _currentIndex++;
  return true;
}
