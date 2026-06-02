# Vector Intelligence

**Your embeddings have hidden clusters. Fiedler analysis reveals them.**

Every embedding space — whether from CLIP, BERT, or custom encoders — contains latent structure that simple distance metrics miss. `vectorintelligence` applies spectral graph theory to discover these hidden clusters, measure their separation, and quantify the quality of the partition.

Built on the insight that the **Fiedler vector** (second eigenvector of the graph Laplacian) gives the optimal spectral cut of any similarity graph, this package provides:

- **k-NN graph construction** from cosine distances (custom implementation, no external deps)
- **Spectral analysis** via Laplacian eigenvalue decomposition using `gonum/mat`
- **Cheeger constant** approximation — the gold standard for cluster quality
- **Recursive spectral bisection** for multi-cluster discovery
- **Jensen-Shannon divergence** between cluster distance distributions

## Quick Start

```go
import "github.com/weaviate/weaviate/vectorintelligence"

vectors := [][]float64{
    {1.0, 0.0}, {0.95, 0.1}, {1.05, -0.1},    // cluster A
    {-1.0, 0.0}, {-0.95, 0.1}, {-1.05, -0.1},  // cluster B
    {0.0, 1.0}, {0.1, 0.95}, {-0.1, 1.05},     // cluster C
}

report, err := vectorintelligence.RunAnalysis(vectors, 3, 3, 0.5, 5)
if err != nil {
    log.Fatal(err)
}

json, _ := report.JSON()
fmt.Println(string(json))
```

## Core Concepts

| Concept | What it tells you |
|---------|------------------|
| **Fiedler vector** | Which side of the optimal cut each point belongs to |
| **Cheeger constant** | How "bottlenecked" the graph is (lower = better clusters) |
| **JSD matrix** | How different clusters' distance distributions are |
| **Spectral bisection** | Recursively finds hierarchical cluster structure |

## Why Spectral Clustering?

Unlike k-means (spherical clusters) or DBSCAN (density-based), spectral clustering works on arbitrary geometries — concentric rings, intertwined spirals, or the manifold structures common in real embeddings. The Cheeger constant provides a principled stopping criterion.
