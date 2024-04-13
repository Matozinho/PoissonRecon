#ifndef POINT_DATA_STREAM_IMP_INL
#define POINT_DATA_STREAM_IMP_INL

#include "../Src/DataStream.h"
#include "../Src/VertexFactory.h"
#include "PoissonReconLib.h"

template <class Real, typename Factory>
struct PointInputDataStream
    : public InputDataStream<typename Factory::VertexType> {
  typedef typename Factory::VertexType Data;

  PointInputDataStream(const PoissonReconLib::ICloud<Real> &cloud,
                       const Factory &factory);
  // ~PointInputDataStream(void);
  void reset(void);

protected:
  const Factory _factory;
  const PoissonReconLib::ICloud<Real> &_cloud;
  size_t _currentIndex;

  bool base_read(Data &d);
};

#include "PointDataStream.imp.inl"
#endif // POINT_DATA_STREAM_IMP_INL