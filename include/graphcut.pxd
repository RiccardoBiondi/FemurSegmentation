# distutils: language = c++
# cython: language_level=2

cdef extern from "3DGraphCut.cpp":
  cdef unsigned int * GraphCutMaxFlow ( unsigned int &_totalPixelsInROI,
                                        unsigned int * dataCostPixels,
                                        unsigned int * flat_dataCostSource,
                                        unsigned int * flat_dataCostSink,
                                        unsigned int &_totalNeighbors,
                                        unsigned int * CentersPixels,
                                        unsigned int * NeighborsPixels,
                                        unsigned int * flat_smoothCostFromCenter,
                                        unsigned int * flat_smoothCostToCenter
                                       )
