#define ARRAY_SIZE 4
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/graphs/LCGraph.h"

struct NodeData {
  float some_value[ARRAY_SIZE];
};

typedef galois::graphs::LC_CSR_Graph<NodeData, void> Graph; // <node data, edge data>
typedef Graph::GraphNode GNode;

struct Work {
    Graph* graph;
    galois::GAccumulator<float>& local_sum;
    
    Work(Graph* graph, galois::GAccumulator<float>& sum) : graph(graph), local_sum(sum) {}
    
    void static go(Graph& _graph) {
        galois::GAccumulator<float> sum;
        sum.reset();
        
        #ifdef __GALOIS_HET_CUDA__
        	if (personality == GPU_CUDA) {
        		std::string impl_str("CUDA_DO_ALL_IMPL_Work_" + (_graph.get_run_identifier()));
        		galois::StatTimer StatTimer_cuda(impl_str.c_str());
        		StatTimer_cuda.start();
        		Work_allNodes_cuda(sum, cuda_ctx);
        		StatTimer_cuda.stop();
        	} else if (personality == CPU)
        #endif
        galois::do_all(galois::iterate(_graph), Work(&_graph, sum), galois::loopname("struct"));
        
        float sumResult = sum.reduce();
        galois::gInfo("sum: ", sumResult);
    }
    
    void operator()(GNode src) {
        NodeData& data = graph->getData(src);
        galois::add(data.some_value[2], 1.0f);
        local_sum += data.some_value[2];
    }
};

int main(int argc, char** argv) {
    Graph graph;
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = graph.getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      // galois::opencl::cl_env.init(cldevice.Value);
    }
#endif
    galois::graphs::readGraph(graph, argv[1]); // argv[1] is the file name for graph
    Work::go(graph);

    return 0;
}
