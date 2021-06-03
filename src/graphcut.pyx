# distutils: language = c++
# cython: language_level=2
cimport numpy as np
from graphcut cimport GraphCutMaxFlow

def RunGraphCut(unsigned int _totalPixelsInROI,
                unsigned int[::1] dataCostPixels,
                unsigned int[::1] flat_dataCostSource,
                unsigned int[::1] flat_dataCostSink,
                unsigned int _totalNeighbors,
                unsigned int[::1] CentersPixels,
                unsigned int[::1] NeighborsPixels,
                unsigned int[::1] flat_smoothCostFromCenter,
                unsigned int[::1] flat_smoothCostToCenter):
  cdef unsigned int * result = GraphCutMaxFlow(_totalPixelsInROI,
                                               & dataCostPixels[0],
                                               & flat_dataCostSource[0],
                                               & flat_dataCostSink[0],
                                               _totalNeighbors,
                                               & CentersPixels[0],
                                               & NeighborsPixels[0],
                                               & flat_smoothCostFromCenter[0],
                                               & flat_smoothCostToCenter[0])
  return [int(result[i]) for i in range(int(_totalPixelsInROI))]
