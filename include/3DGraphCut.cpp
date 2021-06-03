#include "graph.h"
#include <iostream>

unsigned int * GraphCutMaxFlow ( unsigned int &_totalPixelsInROI,
                                 unsigned int * dataCostPixels,
                                 unsigned int * flat_dataCostSource,
                                 unsigned int * flat_dataCostSink,
                                 unsigned int &_totalNeighbors,
                                 unsigned int * CentersPixels,
                                 unsigned int * NeighborsPixels,
                                 unsigned int * flat_smoothCostFromCenter,
                                 unsigned int * flat_smoothCostToCenter
                                )
{
  typedef Graph<short,short,long long> GraphType;
  GraphType *_gc;
  _gc = new GraphType(_totalPixelsInROI, 3 * _totalPixelsInROI);
  _gc -> add_node(_totalPixelsInROI);
  std :: cout << "Building graph, " << _totalPixelsInROI << " nodes" << std :: endl;
  // initializeDataCosts
  for (unsigned i=0; i < _totalPixelsInROI; ++i) {
    _gc -> add_tweights(dataCostPixels[i], flat_dataCostSource[i], flat_dataCostSink[i]);
  }
  // initializeNeighbours
  for (unsigned i=0; i < _totalNeighbors; ++i) {
    _gc -> add_edge(CentersPixels[i], NeighborsPixels[i], flat_smoothCostFromCenter[i], flat_smoothCostToCenter[i]);
  }
  std :: cout  << _totalNeighbors << " t-links added" << std :: endl;
  // compute maxflow
  std :: cout << "Graph built. Computing the max flow" << std :: endl;
  _gc -> maxflow();
  std :: cout << "Max flow computed" << std::endl;
  // updateLabelImageAccordingToGraph
  unsigned int * _labelIdImage = new unsigned int [_totalPixelsInROI];
  for (unsigned i=0; i < _totalPixelsInROI; ++i) {
    _labelIdImage[i] = (_gc->what_segment(i) == GraphType::SOURCE) ? 1 : 0;
  }
  // delete graph
  delete _gc;
  return _labelIdImage;
}
