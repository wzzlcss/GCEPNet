# GCEPNet: Graph Convolution-Enhanced Expectation Propagation for Massive MIMO Detection
![](https://github.com/wzzlcss/GCEPNet/blob/main/plots/GCEPNet_overall.png)

We show that the real-valued system can be modeled as spectral signal convolution on graph, and propose graph convolution-enhanced expectation propagation (GCEPNet), a graph convolution-enhanced EP detector.

## Repository
```python
|-- EP_time # code for measuring CUDA enabled EP
|-- GEPNet #code for GEPNet
|-- SpectralGNN_k #code for GCEPNet
|-- classic_main.py #test traditional methods
|-- classic_solver.py #code for traditional methods
|-- data.py #generate MIMO data
|-- helper.py #utils functions
```

## Performance comparison with existing SOTA (GEPNet)

GCEPNet incorporates data-dependent attention scores into Chebyshev polynomial for powerful graph convolution with better generalization capacity.

![](https://github.com/wzzlcss/GCEPNet/blob/main/plots/performance.png)

## Running time

Both GCEPNet and GEPNet introduce additional computations to EP, which originate from their GNN modules. Comparing with EP, GEPNet does not scale with the problem size as a result of the inefficient GNN aggregation, while GCEPNet effectively resolves the bottleneck with the newly proposed graph convolution.

![](https://github.com/wzzlcss/GCEPNet/blob/main/plots/compare_inference_time_b10_iter5000.png)


## Attribution
Parts of the code are based on
- [GEPNet](https://github.com/GNN-based-MIMO-Detection/GNN-based-MIMO-Detection)
- [REMIMO](https://github.com/krpratik/RE-MIMO)


## License
MIT

