/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Example breadth-first search application for demoing Galois system. For optimized
 * version, use SSSP application with BFS option instead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Graph.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

//kik 
#include "Galois/Atomic.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/Barrier.h"

#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for_each.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/concurrent_vector.h"
#include "tbb/task_scheduler_init.h"
#endif

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <cmath>
#include <functional>
#include <numeric>

#include <sys/time.h>

#define FINE_GRAIN_TIMING
//#define GALOIS_JUNE
//#define NO_SORT
//#define SERIAL_SWAP
//#define TOTAL_PREFIX

static const char* name = "Breadth-first Search Example";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = 0;

//****** Command Line Options ******
enum BFSAlgo {
	barrierCM
};

enum ExecPhase {
	INIT,
	RUN,
	CLEANUP,
	TOTAL,
};

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(DIST_INFINITY));
static cll::opt<unsigned int> reportNode("reportnode",
    cll::desc("Node to report distance to"),
    cll::init(1));
static cll::opt<bool> scaling("scaling", 
		llvm::cl::desc("Scale to the number of threads with a given step starting from"), 
		llvm::cl::init(false));
static cll::opt<unsigned int> scalingStep("step",
    cll::desc("Scaling step"),
    cll::init(2));
static cll::opt<unsigned int> niter("iter",
    cll::desc("Number of benchmarking iterations"),
    cll::init(5));
static cll::opt<unsigned int> qlen("qlen",
    cll::desc("Minimum queue length for parallel prefix sum"),
    cll::init(50));
static cll::opt<BFSAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(barrierCM, "Barrier-based Parallel Cuthill-McKee"),
      clEnumValEnd), cll::init(barrierCM));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
	unsigned int id;
	//unsigned int numChildren;
	unsigned int order;
	unsigned int sum;
#ifndef NO_SORT
	unsigned int startindex;
#endif
	Galois::GAtomic<unsigned int> numChildren;
	//bool rflag;
	//bool pflag;
	//bool have;
	Galois::Graph::LC_CSR_Graph<SNode, void>::GraphNode parent;
	//std::vector<Galois::Graph::LC_CSR_Graph<SNode, void>::GraphNode> bucket;
	//Galois::gdeque<Galois::Graph::LC_CSR_Graph<SNode, void>::GraphNode>* bucket;
	//GaloisRuntime::LL::SimpleLock<true> mutex;
};

struct Prefix {
  unsigned int id;
	unsigned int val;
	Prefix(unsigned int _id, unsigned _val) : id(_id), val(_val) {}
};

typedef Galois::Graph::LC_CSR_Graph<SNode, void> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

static size_t degree(const GNode& node) { 
  return std::distance(graph.edge_begin(node, Galois::NONE), graph.edge_end(node, Galois::NONE));
}

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out <<  "(dist: " << n.dist << ")";
  return out;
}

struct GNodeIndexer {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, Galois::NONE).dist;
  }
};

struct GNodeSort {
  bool operator()(const GNode& a, const GNode& b) const {
    return degree(a) < degree(b);
  }
};

std::vector<GNode> initial[2];
GNode source, report;

//std::map<GNode, unsigned int> order;
//std::vector< std::vector<GNode> > bucket;
//Galois::gdeque<GNode> bucket;
Galois::InsertBag<GNode> bucket;
GaloisRuntime::LL::SimpleLock<true> dbglock;

std::vector< std::map<GNode, unsigned int> > redbuck;

//debug
Galois::GAtomic<unsigned int> loops = Galois::GAtomic<unsigned int>(0);
Galois::GAtomic<unsigned int> sorts = Galois::GAtomic<unsigned int>(0);
Galois::GAtomic<unsigned int> maxbucket = Galois::GAtomic<unsigned int>(0);
Galois::GAtomic<unsigned int> minbucket = Galois::GAtomic<unsigned int>(DIST_INFINITY);
Galois::GAtomic<unsigned int> avgbucket = Galois::GAtomic<unsigned int>(0);
Galois::GAtomic<unsigned int> numbucket = Galois::GAtomic<unsigned int>(0);
Galois::GAtomic<unsigned int> smallbucket = Galois::GAtomic<unsigned int>(0);

struct PartialSum {
	GNode& operator()(const GNode& partial, GNode& item) {
		/*
		if(graph.getData(item).numChildren > 0)
			graph.getData(item).have = true;

		std::cerr << "[" << graph.getData(item).id << "] " << graph.getData(item).numChildren << " have?: " << graph.getData(item).have << "\n";
		*/

		//dbglock.lock();
#ifdef SERIAL_SWAP
		graph.getData(item, Galois::NONE).numChildren += graph.getData(partial, Galois::NONE).numChildren;
#else
		SNode& idata = graph.getData(item, Galois::NONE);
		idata.sum = idata.numChildren;
		idata.numChildren += graph.getData(partial, Galois::NONE).numChildren;
#endif
		//std::cerr << "[" << graph.getData(item, Galois::NONE).id << "] " << graph.getData(item, Galois::NONE).numChildren << "\n";
		//dbglock.unlock();
		return item;
	}
};

struct SegReduce {
	unsigned int sum;

	SegReduce(unsigned int _sum) : sum(_sum) {}

	void operator()(const GNode& item, Galois::UserContext<GNode>& ctx) {
		graph.getData(item, Galois::NONE).numChildren += sum;
	}
};

#ifndef SERIAL_SWAP
struct Swap {
	void operator()(const GNode& item, Galois::UserContext<GNode>& ctx) {
	  operator()(item);
	}
  void operator()(const GNode& item) {
		SNode& idata = graph.getData(item, Galois::NONE);
		idata.numChildren -= idata.sum; 
#ifndef NO_SORT
		idata.startindex = idata.numChildren; 
#endif
	}
};
#endif

#ifndef NO_SORT
struct SortChildren {
	unsigned int round; 
  Galois::GReduceMax<unsigned int>& maxlen;

	SortChildren(unsigned int r, Galois::GReduceMax<unsigned int>& m) : round(r), maxlen(m) {}

	void operator()(GNode& parent, Galois::UserContext<GNode>& ctx) {
	  operator()(parent);
	}
  void operator()(GNode& parent) {
		SNode& pdata = graph.getData(parent, Galois::NONE);

		if(pdata.sum > 1){

			maxlen.update(pdata.sum);

			//dbglock.lock();
			//std::cerr << "[" << pdata.id << "] sorting: " << pdata.sum << "\n";
			//dbglock.unlock();

			sort(initial[round].begin()+pdata.startindex, initial[round].begin()+(pdata.startindex + pdata.sum), GNodeSort());
		}
	}
};
#endif

#ifndef TOTAL_PREFIX
struct LocalPrefix {
	typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

	unsigned int round; 
	unsigned int chunk; 

	LocalPrefix(unsigned int r, unsigned int c) : round(r), chunk(c) {}

	void operator()(unsigned int me, unsigned int tot) {

		unsigned int len = initial[round].size();
		//unsigned int start = me * ceil((double) len / tot);
		//unsigned int end = (me+1) * ceil((double) len / tot);
		unsigned int start = me * chunk;
		unsigned int end = (me+1) * chunk;

		if(me != tot-1){
			//dbglock.lock();
			//std::cerr << "On_each thread: " << me << " step: " << ceil(len / tot) << " start: " << start << " end " << end+1 << "\n";
			//std::cerr << "On_each thread tot: " << tot << " len: " << len << " ceil: " << ceil((double) len / tot) << " floor: " << floor((double) len / tot) << "\n";

			//std::cerr << graph.getData(*(initial[round].begin()+start), Galois::NONE).id << " to " << graph.getData(*(initial[round].begin()+(end+1)), Galois::NONE).id << "\n";

#ifndef SERIAL_SWAP
			SNode& idata = graph.getData(initial[round][start], Galois::NONE);
			idata.sum = idata.numChildren;
#endif
			std::partial_sum(initial[round].begin()+start, initial[round].begin()+end, initial[round].begin()+start, PartialSum());
			//dbglock.unlock();
		}
		else {
			//dbglock.lock();
			//std::cerr << "On_each thread: " << me << " size: " << len << " start: " << start << " end " << initial[round].size() << "\n";
			//std::cerr << "On_each thread tot: " << tot << " len: " << len << " ceil: " << ceil((double) len / tot) << " floor: " << floor((double) len / tot) << "\n";
			//std::cerr << graph.getData(*(initial[round].begin()+start), Galois::NONE).id << " to " << graph.getData(*(initial[round].end()-1), Galois::NONE).id << "\n";

#ifndef SERIAL_SWAP
			SNode& idata = graph.getData(initial[round][start], Galois::NONE);
			idata.sum = idata.numChildren;
#endif
			std::partial_sum(initial[round].begin()+start, initial[round].end(), initial[round].begin()+start, PartialSum());
			//dbglock.unlock();
		}
	}
};

struct DistrPrefix {
	typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

	unsigned int round; 
	unsigned int chunk; 

	DistrPrefix(unsigned int r, unsigned int c) : round(r), chunk(c) {}

	void operator()(unsigned int me, unsigned int tot) {
		if(me > 0){
			if(me != tot-1){

				unsigned int len = initial[round].size();
				unsigned int start = me * chunk;
				unsigned int end = (me+1) * chunk - 1;
				unsigned int val = graph.getData(initial[round][start-1], Galois::NONE).numChildren;

				//dbglock.lock();
				//std::cerr << "On_each thread: " << me << " step: " << ceil(len / tot) << " start: " << start << " end " << end+1 << "\n";
				//std::cerr << "On_each thread tot: " << tot << " len: " << len << " ceil: " << ceil((double) len / tot) << " floor: " << floor((double) len / tot) << "\n";

				//std::cerr << graph.getData(*(initial[round].begin()+start), Galois::NONE).id << " to " << graph.getData(*(initial[round].begin()+(end+1)), Galois::NONE).id << "\n";
				for(unsigned int i = start; i < end; ++i){
					graph.getData(initial[round][i], Galois::NONE).numChildren += val;
					//std::cerr << "Loop: " << i << " size: " << seglen << " start: " << start << " end " << end << "\n";
				}
				//dbglock.unlock();
			}
			else {
				//dbglock.lock();
				//std::cerr << "On_each thread: " << me << " size: " << len << " start: " << start << " end " << initial[round].size() << "\n";
				//std::cerr << "On_each thread tot: " << tot << " len: " << len << " ceil: " << ceil((double) len / tot) << " floor: " << floor((double) len / tot) << "\n";
				//std::cerr << graph.getData(*(initial[round].begin()+start), Galois::NONE).id << " to " << graph.getData(*(initial[round].end()-1), Galois::NONE).id << "\n";

				unsigned int len = initial[round].size();
				unsigned int start = me * chunk;
				unsigned int val = graph.getData(initial[round][start-1], Galois::NONE).numChildren;

				for(unsigned int i = start; i < len; ++i){
					graph.getData(initial[round][i], Galois::NONE).numChildren += val;
					//std::cerr << "Loop: " << i << " size: " << seglen << " start: " << start << " end " << end << "\n";
				}
				//dbglock.unlock();
			}
		}
	}
};

#else

struct TotalPrefix {
	typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

	unsigned int round; 
	unsigned int chunk; 
	GaloisRuntime::PthreadBarrier barrier;

	TotalPrefix(unsigned int r, unsigned int c, GaloisRuntime::PthreadBarrier b) : round(r), chunk(c), barrier(b) {}

	void operator()(unsigned int me, unsigned int tot) {

		unsigned int len = initial[round].size();
		unsigned int start = me * chunk;
		unsigned int end = (me+1) * chunk;

		if(me != tot-1){
#ifndef SERIAL_SWAP
			SNode& idata = graph.getData(initial[round][start], Galois::NONE);
			idata.sum = idata.numChildren;
#endif
			std::partial_sum(initial[round].begin()+start, initial[round].begin()+end, initial[round].begin()+start, PartialSum());
		}
		else {
#ifndef SERIAL_SWAP
			SNode& idata = graph.getData(initial[round][start], Galois::NONE);
			idata.sum = idata.numChildren;
#endif
			std::partial_sum(initial[round].begin()+start, initial[round].end(), initial[round].begin()+start, PartialSum());
		}

		barrier.wait();

		if(me == 0){
			for(unsigned int i = 1; i < tot-1; ++i){
				start = i * chunk;
				end = (i+1) * chunk - 1;
				graph.getData(initial[round][end], Galois::NONE).numChildren += graph.getData(initial[round][start-1], Galois::NONE).numChildren;
			}
		}

		barrier.wait();

		if(me != 0){
			if(me != tot-1){
				--end;
				unsigned int val = graph.getData(initial[round][start-1], Galois::NONE).numChildren;
				for(unsigned int i = start; i < end; ++i){
					graph.getData(initial[round][i], Galois::NONE).numChildren += val;
				}
			}
			else {
				unsigned int val = graph.getData(initial[round][start-1], Galois::NONE).numChildren;
				for(unsigned int i = start; i < len; ++i){
					graph.getData(initial[round][i], Galois::NONE).numChildren += val;
				}
			}
		}
	}
};

#endif

// Find a good starting node for CM based on minimum degree
static void findStartingNode(GNode& starting) {
	unsigned int mindegree = DIST_INFINITY; 

	for (Graph::iterator src = graph.begin(), ei =
			graph.end(); src != ei; ++src) {
		unsigned int nodedegree = degree(*src);

		if(nodedegree < mindegree){
			mindegree = nodedegree;
			starting = *src;
		}
	}

	SNode& data = graph.getData(starting);
	std::cerr << "Starting Node: " << data.id << " degree: " << degree(starting) << "\n";
}

template<typename T>
class GReduceAverage {
  typedef std::pair<T, unsigned> TP;
  struct AVG {
    void operator() (TP& lhs, const TP& rhs) const {
      lhs.first += rhs.first;
      lhs.second += rhs.second;
    }
  };
  Galois::GReducible<std::pair<T, unsigned>, AVG> data;

public:
  void update(const T& _newVal) {
    data.update(std::make_pair(_newVal, 1));
  }

  /**
   * returns the thread local value if in a parallel loop or
   * the final reduction if in serial mode
   */
  const T reduce() {
#ifdef GALOIS_JUNE
    const TP& d = data.get();
#else
    const TP& d = data.reduce();
#endif
    return d.first / d.second;
  }

  void reset(const T& d) {
    data.reset(std::make_pair(d, 0));
  }

  GReduceAverage& insert(const T& rhs) {
#ifdef GALOIS_JUNE
    TP& d = data.reduce();
#else
    TP& d = data.reduce();
#endif
    d.first += rhs;
    d.second++;
    return *this;
  }
};

//Compute mean distance from the source
struct avg_dist {
  GReduceAverage<unsigned int>& m;
  avg_dist(GReduceAverage<unsigned int>& _m): m(_m) { }

  void operator()(const GNode& n) const {
    m.update(graph.getData(n).dist);
  }
};

//Compute variance around mean distance from the source
static void variance(unsigned int mean) {
	unsigned int n = 0;
	double M2 = 0.0;
	double var = 0.0;

	for (Graph::iterator src = graph.begin(), ei =
			graph.end(); src != ei; ++src) {
		SNode& data = graph.getData(*src);
		M2 += (data.dist - mean)*(data.dist - mean);
	}

	var = M2/(n-1);
	std::cout << "var: " << var << " mean: " << mean << "\n";
}

struct not_consistent {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value for " << graph.getData(dst).id << ": " << ddist << " > " << (dist + 1) << "\n";
	return true;
      }
    }
    return false;
  }
};

struct not_visited {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "unvisited node " << graph.getData(n).id << ": " << dist << " >= INFINITY\n";
      return true;
    }
		//std::cerr << "visited node " << graph.getData(n).id << ": " << dist << "\n";
    return false;
  }
};

struct max_dist {
  Galois::GReduceMax<unsigned int>& m;
  max_dist(Galois::GReduceMax<unsigned int>& _m): m(_m) { }

  void operator()(const GNode& n) const {
    m.update(graph.getData(n).dist);
  }
};

//! Simple verifier
static bool verify(GNode& source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  size_t id = 0;
  
#ifdef GALOIS_JUNE
  bool okay = Galois::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && Galois::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#else
  bool okay = Galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && Galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();
#endif

  if (okay) {
    Galois::GReduceMax<unsigned int> m;
    GReduceAverage<unsigned int> mean;
    Galois::do_all(graph.begin(), graph.end(), max_dist(m));
#ifdef GALOIS_JUNE
    std::cout << "max dist: " << m.get() << "\n";
#else
    std::cout << "max dist: " << m.reduce() << "\n";
#endif
    Galois::do_all(graph.begin(), graph.end(), avg_dist(mean));
    Galois::do_all(graph.begin(), graph.end(), avg_dist(mean));
    std::cout << "avg dist: " << mean.reduce() << "\n";

		variance(mean.reduce());
  }
  
  return okay;
}

// Compute maximum bandwidth for a given graph
struct banddiff {

	Galois::GAtomic<unsigned int>& maxband;
  banddiff(Galois::GAtomic<unsigned int>& _mb): maxband(_mb) { }

  void operator()(const GNode& source) const {

		SNode& sdata = graph.getData(source, Galois::NONE);
		for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
				 ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {

      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

			unsigned int diff = abs(sdata.id - ddata.id);
			unsigned int max = maxband;

			if(diff > max){
				while(!maxband.cas(max, diff)){
					max = maxband;
					if(!diff > max)
						break;
				}
			}
    }
  }
};

// Parallel loop for maximum bandwidth computation
static void bandwidth(std::string msg) {
		Galois::GAtomic<unsigned int> maxband = Galois::GAtomic<unsigned int>(0);
    Galois::do_all(graph.begin(), graph.end(), banddiff(maxband));
    std::cout << msg << " Bandwidth: " << maxband << "\n";
}

//Clear node data to re-execute on specific graph
struct resetNode {
	void operator()(const GNode& n) const {
    SNode& node = graph.getData(n, Galois::NONE);
		node.dist = DIST_INFINITY;
		node.numChildren = 0;
		//node.numChildren = 0;
		//node.rflag = false;
		//node.pflag = false;
		//node.have = false;
		node.parent = n;
		//order[n] = DIST_INFINITY;
		node.order = DIST_INFINITY;
		//node.bucket->clear();
	}
};

static void resetGraph() {
	initial[0].clear();
	initial[1].clear();
	bucket.clear();
	Galois::do_all(graph.begin(), graph.end(), resetNode());
}

// Read graph from a binary .gr as dirived from a Matrix Market .mtx using graph-convert
static void readGraph(GNode& source, GNode& report) {
  graph.structureFromFile(filename);

  source = *graph.begin();
  report = *graph.begin();

  size_t nnodes = graph.size();
  std::cout << "Read " << nnodes << " nodes\n";
  
  size_t id = 0;
  bool foundReport = false;
  bool foundSource = false;

	//bucket.reserve(nnodes);
	//order.reserve(nnodes);

  for (Graph::iterator src = graph.begin(), ei =
      graph.end(); src != ei; ++src) {

    SNode& node = graph.getData(*src, Galois::NONE);
    node.dist = DIST_INFINITY;
    node.id = id;
    node.parent = id;
    //node.bucket = new Galois::gdeque<GNode>();
    //node.numChildren = 0;
    node.numChildren = Galois::GAtomic<unsigned int>(0);
    //node.rflag = false;
    //node.pflag = false;
    //node.have = false;
		//order[*src] = DIST_INFINITY;
		node.order = DIST_INFINITY;

    //std::cout << "Report node: " << reportNode << " (dist: " << distances[reportNode] << ")\n";

    if (id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (id == reportNode) {
      foundReport = true;
      report = *src;
    }
    ++id;
  }

	if(startNode == DIST_INFINITY){
		findStartingNode(source);
		foundSource = true;
	}

  if (!foundReport || !foundSource) {
    std::cerr 
      << "failed to set report: " << reportNode 
      << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}

struct BarrierNoDup {
	std::string name() const { return "Cuthill (Inline) Barrier)"; }

  void operator()(const GNode& source) const {

		#ifdef FINE_GRAIN_TIMING
		Galois::TimeAccumulator vTmain[6]; 
		vTmain[0] = Galois::TimeAccumulator();
		vTmain[1] = Galois::TimeAccumulator();
		vTmain[2] = Galois::TimeAccumulator();
		vTmain[3] = Galois::TimeAccumulator();
		vTmain[4] = Galois::TimeAccumulator();
		vTmain[5] = Galois::TimeAccumulator();

		vTmain[0].start();
		#endif

		unsigned int round = 0;

		initial[0].reserve(100);
		initial[1].reserve(100);

		SNode& sdata = graph.getData(source);
		sdata.dist = 0;
		//order[source] = 0;
		sdata.order = 0;

		//round = (round + 1) & 1;

    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      ddata.numChildren = 0;
      ddata.parent = sdata.id;
      initial[round].push_back(dst);
      //sdata.numChildren++;
    }

		sort(initial[round].begin(), initial[round].end(), GNodeSort());

		for(unsigned int i = 0; i < initial[round].size(); ++i) {
			//order[initial[round][i]] = i+1;
      graph.getData(initial[round][i]).order = i+1;
		}

		#ifdef FINE_GRAIN_TIMING
		vTmain[0].stop();
		#endif

		//unsigned int added = 0;
		Galois::GAtomic<unsigned int> added = Galois::GAtomic<unsigned int>(0);;
		Galois::GAtomic<unsigned int> temp = Galois::GAtomic<unsigned int>(0);;

		unsigned int depth = 0;
		unsigned int thr = Galois::getActiveThreads();
		GaloisRuntime::PthreadBarrier barrier(thr);
		//GaloisRuntime::GBarrier& barrier = GaloisRuntime::getSystemBarrier();

		while (true) {
			unsigned next = (round + 1) & 1;

			#ifdef FINE_GRAIN_TIMING
			vTmain[1].start();
			#endif

			//std::cerr << "Depth: " << ++depth << " "; 
			//std::cerr << "Parents: " << initial[round].size() << "\n"; 
			Galois::do_all<>(initial[round].begin(), initial[round].end(), Expand(next), "expand");
			#ifdef FINE_GRAIN_TIMING
			vTmain[1].stop();
			vTmain[2].start();
			#endif

			//std::cerr << "Children: " << bucket.size() << "\n"; 
			Galois::do_all<>(bucket.begin(), bucket.end(), Children(), "reduction");
/*
			for(Galois::InsertBag<GNode>::iterator ii = bucket.begin(), ei = bucket.end(); ii != ei; ++ii){
				SNode& cdata = graph.getData(*ii, Galois::NONE);
				if(!cdata.rflag) {
					graph.getData(cdata.parent, Galois::NONE).numChildren++;
					graph.getData(cdata.parent, Galois::NONE).have = true;
					cdata.rflag = true;
				}
			}
			*/

			#ifdef FINE_GRAIN_TIMING
			vTmain[2].stop();
			vTmain[3].start();
			#endif

			added = 0;
			temp = 0;

			/*
			std::cerr << "Size: " << initial[round].size() << "\n";
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.id << " "; 
			}
			std::cerr << "\n"; 

			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.numChildren << " "; 
			}
			std::cerr << "\n"; 
			*/

			unsigned int seglen = initial[round].size();
			unsigned int chunk = (seglen + (thr-1)) / thr;
			unsigned int start;
			unsigned int end;

			//std::cerr << "Segment : " << seglen / thr << "\n";

			//if(seglen / thr > 2) 
			//if(seglen > qlen) {
			if(seglen > 1000) {
#ifdef TOTAL_PREFIX
				Galois::on_each(TotalPrefix(round, chunk, barrier), "totalprefix");
#else

				Galois::on_each(LocalPrefix(round, chunk), "localprefix");

				for(unsigned int i = 1; i < thr-1; ++i){
					start = i * chunk;
					end = (i+1) * chunk - 1;
					graph.getData(initial[round][end], Galois::NONE).numChildren += graph.getData(initial[round][start-1], Galois::NONE).numChildren;
				}

				Galois::on_each(DistrPrefix(round, chunk), "distrprefix");
#endif
			}
			else {
#ifndef SERIAL_SWAP
				SNode& idata = graph.getData(initial[round][0], Galois::NONE);
				idata.sum = idata.numChildren;
#endif
				std::partial_sum(initial[round].begin(), initial[round].end(), initial[round].begin(), PartialSum());
			}

			//std::partial_sum(initial[round].begin(), initial[round].end(), initial[round].begin(), PartialSum());

			/*
			std::cerr << "Size for prefix sum: " << initial[round].size() << "\n";
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.numChildren << " "; 
			}
				std::cerr << "\n"; 

			std::cerr << "Size for sum: " << initial[round].size() << "\n";
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.sum << " "; 
			}
				std::cerr << "\n"; 
				*/

			/*
			#ifdef FINE_GRAIN_TIMING
			vTmain[3].stop();
			vTmain[5].start();
			#endif
			*/

#ifdef SERIAL_SWAP
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				std::swap(graph.getData(*ii, Galois::NONE).numChildren, added);
			}
#else
			added = graph.getData(initial[round][seglen-1], Galois::NONE).numChildren;
			Galois::do_all<>(initial[round].begin(), initial[round].end(), Swap(), "swap");
#endif

/*
			std::cerr << "After swap Size for prefix sum: " << initial[round].size() << "\n";
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.numChildren << " "; 
			}
				std::cerr << "\n"; 

			std::cerr << "Size for startindex: " << initial[round].size() << "\n";
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.startindex << " "; 
			}
				std::cerr << "\n"; 

			std::cerr << "After swap Size for sum: " << initial[round].size() << "\n";
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				std::cerr << data.sum << " "; 
			}
				std::cerr << "\n"; 
				*/

			//std::cerr << "total: " << added << "\n"; 

			/*
			for(std::vector<GNode>::iterator ii = initial[round].begin(), ei = initial[round].end(); ii != ei; ++ii){
				SNode& data = graph.getData(*ii, Galois::NONE);
				temp = data.numChildren;
				data.numChildren = added;
				added += temp; 
			}
			*/

			initial[next].resize(added);

			//std::cerr << "After partial sum: " << added << "\n"; 

			#ifdef FINE_GRAIN_TIMING
			vTmain[3].stop();
			#endif

			if(added == 0) {
				#ifdef FINE_GRAIN_TIMING
				std::cerr << "Init: " << vTmain[0].get() << "\n";
				std::cerr << "Expand(par): " << vTmain[1].get() << "\n";
				std::cerr << "Reduction(par): " << vTmain[2].get() << "\n";
				std::cerr << "PartialSum(par): " << vTmain[3].get() << "\n";
				//std::cerr << "Swap(ser): " << vTmain[5].get() << "\n";
				std::cerr << "Placement(par): " << vTmain[4].get() << "\n";
				std::cout << "& " << vTmain[1].get() << " & " << vTmain[2].get() << " & " << vTmain[3].get() << " & " << vTmain[4].get() << " & " << vTmain[1].get() + vTmain[2].get()  + vTmain[3].get() + vTmain[4].get() << "\n";
				#endif
				break;
			}

			#ifdef FINE_GRAIN_TIMING
			vTmain[4].start();
			#endif
			//Galois::for_each<WL>(initial[round].begin(), initial[round].end(), Place(next));
			Galois::do_all<>(bucket.begin(), bucket.end(), Place(next), "placement");
			#ifndef NO_SORT
			Galois::GReduceMax<unsigned int> maxlen;
			Galois::do_all<>(initial[round].begin(), initial[round].end(), SortChildren(next, maxlen), "sort");
			//std::cout << "max sorting len: " << maxlen.get() << "\n";
			#endif

			initial[round].clear();
			bucket.clear();
			round = next;

			#ifdef FINE_GRAIN_TIMING
			vTmain[4].stop();
			#endif
		}

/*
		std::cerr << "Order: \n";
		for(int i=0; i<order.size(); ++i){
			std::cerr << i << " at: " << order[i] << "\n";
		}
		*/
		//std::cerr << "\n";
  }

	struct Expand {
		unsigned int round; 

		Expand(unsigned int r) : round(r) {}

		// For fine-grain timing inside foreach threads
		// Millis
		unsigned long tget(unsigned int start_hi, unsigned int start_low, unsigned int stop_hi, unsigned int stop_low) const {
			unsigned long msec = stop_hi - start_hi;
			msec *= 1000;
			if (stop_low > start_low)
				msec += (stop_low - start_low) / 1000;
			else {
				msec -= 1000; //borrow
				msec += (stop_low + 1000000 - start_low) / 1000;
			}
			return msec;
		}

		// Micros
		unsigned long utget(unsigned int start_hi, unsigned int start_low, unsigned int stop_hi, unsigned int stop_low) const {
			unsigned long usec = stop_hi - start_hi;
			usec *= 1000000;
			if (stop_low > start_low)
				usec += (stop_low - start_low);
			else {
				usec -= 1000000; //borrow
				usec += (stop_low + 1000000 - start_low);
			}
			return usec;
		}
	  
		void operator()(GNode& n, Galois::UserContext<GNode>& ctx) {
		  operator()(n);
		}
	  void operator()(GNode& n) {
			SNode& sdata = graph.getData(n, Galois::NONE);
			unsigned int newDist = sdata.dist + 1;

			for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
					ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
				GNode dst = graph.getEdgeDst(ii);
				SNode& ddata = graph.getData(dst, Galois::NONE);

				if(ddata.dist < newDist)
					continue; 

				unsigned int oldDist;

				GNode parent;

				while (true) {
					oldDist = ddata.dist;
					//It actually enters only with equality
					if (oldDist <= newDist){
						break;
					}
					if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
						bucket.push_back(dst);
						break;
					}
				}

/*
				if(ddata.dist > newDist){
					ddata.dist = newDist;
					bucket.push_back(dst);
				}
				*/

					//parent = ddata.parent;
					//__sync_bool_compare_and_swap(&ddata.parent, parent, n);

/*
				if(graph.getData(ddata.parent).order > sdata.order)
					ddata.parent = n;
				if(order[ddata.parent] > order[n])
					ddata.parent = n;
				GNode parent;
				*/
/*
			dbglock.lock();
			std::cerr << "[" << sdata.id << "] checking: " << ddata.id << " current parent: " << graph.getData(ddata.parent).id << " nc: " << graph.getData(ddata.parent).numChildren << " me: " << sdata.numChildren << "\n";
			dbglock.unlock();

*/
				while (true) {
					parent = ddata.parent;
					//if(order[parent] > order[n]){
					if(graph.getData(parent, Galois::NONE).order > sdata.order){
					//if(graph.getData(ddata.parent).numChildren > sdata.numChildren){
						if(__sync_bool_compare_and_swap(&ddata.parent, parent, n)){
							break;
						}
						continue;
					}
					break;
				}
			}
		}
	};

/*
	struct Place {
		unsigned int round; 

		Place(unsigned int r) : round(r) {}

		void operator()(GNode& parent, Galois::UserContext<GNode>& ctx) {
			SNode& pdata = graph.getData(parent, Galois::NONE);

			if(!pdata.have)
				return;

			unsigned int index = pdata.numChildren; 

			//unsigned int count = 0;
			//unsigned int actual = 0;
			for(Galois::InsertBag<GNode>::iterator ii = bucket.begin(), ei = bucket.end(); ii != ei; ++ii){
				SNode& cdata = graph.getData(*ii, Galois::NONE);
				//count++;
				if(!cdata.pflag && cdata.parent == parent){
					//order[*ii] = index;
					cdata.order = index;
					initial[round][index++] = *ii;
					cdata.pflag = true;
					//actual++;
				}
			}
		}
	};
	*/

	struct Place {
		unsigned int round; 

		Place(unsigned int r) : round(r) {}

		void operator()(GNode& child, Galois::UserContext<GNode>& ctx) {
		  operator()(child);
		}
	  void operator()(GNode& child) {
			SNode& cdata = graph.getData(child, Galois::NONE);
			SNode& pdata = graph.getData(cdata.parent, Galois::NONE);

			unsigned int index = pdata.numChildren++; 
			cdata.order = index;
			initial[round][index] = child;

			/*
			dbglock.lock();
			std::cerr << "[" << pdata.id << "] scanned: " << count << " added: " << actual << "\n";
			dbglock.unlock();
			*/

			/*
				 if(sdata.sum > 1) {
				 sort(sdata.bucket.begin(), sdata.bucket.end(), GNodeSort());
				 }
				 */
		}
	};

	struct Children {
		void operator()(GNode& child, Galois::UserContext<GNode>& ctx) {
		  operator()(child);
		}
	  void operator()(GNode& child) {
			SNode& cdata = graph.getData(child, Galois::NONE);
			//graph.getData(cdata.parent, Galois::NONE).mutex.lock();
			graph.getData(cdata.parent, Galois::NONE).numChildren++;
			//graph.getData(cdata.parent, Galois::NONE).have = true;
			//graph.getData(cdata.parent, Galois::NONE).mutex.unlock();
		}
	};

/*
	struct Children {
		void operator()(GNode& owner, Galois::UserContext<GNode>& ctx) {
			SNode& odata = graph.getData(owner, Galois::NONE);
			//for(std::vector<GNode>::iterator ii = odata.bucket.begin(), ei = odata.bucket.end(); ii != ei; ++ii){
			for(Galois::gdeque<GNode>::iterator ii = odata.bucket->begin(), ei = odata.bucket->end(); ii != ei; ++ii){
				SNode& cdata = graph.getData(*ii, Galois::NONE);
				//I'll make it GAtomic
				graph.getData(cdata.parent, Galois::NONE).mutex.lock();
				graph.getData(cdata.parent, Galois::NONE).numChildren++;
				graph.getData(cdata.parent, Galois::NONE).mutex.unlock();
			}
		}
	};
	*/
};

template<typename AlgoTy>
void run(const AlgoTy& algo) {

	int maxThreads = numThreads; 
	Galois::TimeAccumulator vT[maxThreads+20]; 

	//Measure time to read graph
	vT[INIT] = Galois::TimeAccumulator();
	vT[INIT].start();

  readGraph(source, report);
	bandwidth("Initial Bandwidth");

	std::cout << "Size of: " << sizeof(SNode) << "\n";

	vT[INIT].stop();
	std::cout << "Init: " << vT[INIT].get() << " ( " << (double) vT[INIT].get() / 1000 << " seconds )\n";

	//Measure total computation time to read graph
	vT[TOTAL].start();

	//Galois::setActiveThreads(1);

	if(scaling) {
		for(int thr = 2; thr <= maxThreads; thr+=scalingStep){
			numThreads = Galois::setActiveThreads(thr);

			vT[TOTAL+thr] = Galois::TimeAccumulator();
			std::cout << "Running " << algo.name() << " version with " << numThreads << " threads for " << niter << " iterations\n";

			for(int i = 0; i < niter; i++){
				vT[TOTAL+thr].start();
				algo(source);
				vT[TOTAL+thr].stop();

				//permute();
				//bandwidth("Permuted");

				std::cout << "Iteration " << i << " numthreads: " << numThreads << " " << vT[TOTAL+thr].get() << "\n";
				if(i != niter-1)
					resetGraph();
			}

			if(thr+scalingStep <= maxThreads){
				std::cout << "Total time numthreads: " << numThreads << " " << vT[TOTAL+thr].get() << "\n";
				std::cout << "Avg time numthreads: " << numThreads << " " << vT[TOTAL+thr].get() / niter << "\n";
				resetGraph();
			}
			else {
				std::cout << "Final time numthreads: " << numThreads << " " << vT[TOTAL+thr].get() << "\n";
				std::cout << "Avg time numthreads: " << numThreads << " " << vT[TOTAL+thr].get() / niter << "\n";
			}
		}
	}
	else {

		// Execution with the specified number of threads
		vT[RUN] = Galois::TimeAccumulator();
		vT[CLEANUP] = Galois::TimeAccumulator();

		std::cout << "Running " << algo.name() << " version with " << numThreads << " threads for " << niter << " iterations\n";

		// I've observed cold start. First run takes a few millis more. 
		//algo(source);
		//resetGraph();

		Galois::StatTimer T;
		T.start();
		for(int i = 0; i < niter; i++){
			vT[RUN].start();
			
			algo(source);

			vT[RUN].stop();

			//permute();
			//bandwidth("Permuted");

			std::cout << "Iteration " << i << " numthreads: " << numThreads << " " << vT[RUN].get() << "\n";

			if(i < niter-1){
				vT[CLEANUP].start();
				resetGraph();
				vT[CLEANUP].stop();
			}
		}
		T.stop();

		std::cout << "Final time numthreads: " << numThreads << " " << vT[RUN].get() << "\n";
		std::cout << "Avg time numthreads: " << numThreads << " " << vT[RUN].get() / niter << "\n";
		if(niter > 1)
			std::cout << "Cleanup time numthreads: " << numThreads << " " << vT[CLEANUP].get() / (niter-1) << "\n";
		else
			std::cout << "Cleanup time numthreads: " << numThreads << " " << vT[CLEANUP].get() << "\n";
	}

	vT[TOTAL].stop();

	std::cout << "Total with threads: " << numThreads << " " << vT[TOTAL].get() << " ( " << (double) vT[TOTAL].get() / 1000 << " seconds )\n";
  std::cout << "Report node: " << reportNode << " " << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify(source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace GaloisRuntime::WorkList;
  typedef dChunkedLIFO<8> BSWL_LIFO;
  typedef dChunkedFIFO<8> BSWL_FIFO;

/*
#ifdef GALOIS_USE_EXP
  typedef BulkSynchronousInline<> BSInline;
#else
	*/
  typedef BulkSynchronousInline<> BSInline1;
  typedef BulkSynchronousInline<> BSInline2;
  //typedef BSWL_LIFO BSInline1;
  //typedef BSWL_FIFO BSInline2;
//#endif

  switch (algo) {
		case barrierCM: run(BarrierNoDup()); break;
    default: std::cerr << "Unknown algorithm" << algo << "\n"; abort();
  }

  return 0;
}