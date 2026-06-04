# Spectral Vector Analysis for Weaviate

> Your 1.2M vectors have 23 geometric clusters. You labeled 8.

Spectral analysis for HNSW graphs — built on `gonum`. Understand the *geometry* of your vector index, not just its recall.

## What It Does

You've got vectors. You've built an HNSW index. But **do you know what shape your data is?**

This package answers questions like:
- **How many natural clusters exist in my vector space?** (Spectral clustering)
- **How well-connected is my proximity graph?** (Cheeger constant, algebraic connectivity)
- **Are there bottleneck regions that hurt recall?** (Fiedler vector analysis)
- **Is my embedding quality any good?** (Full spectral quality report)

## Quick Start

```go
import "github.com/weaviate/weaviate/spectral"
```

### Build a Graph, Get Answers

```go
// Wrap your HNSW graph (or use AdjacencyGraph for testing)
g := spectral.NewAdjacencyGraph()
// Add your nodes and edges...
g.AddNode(0)
g.AddEdge(0, 1)

// Full analysis — one call
report, err := spectral.AnalyzeEmbeddingQuality(g)

fmt.Println(report.Summary())
```

**Output:**
```
Spectral Analysis Report
========================
Vectors: 12 | Edges: 31

Graph Connectivity
  Algebraic Connectivity (λ₂): 0.142857
  Spectral Gap:                0.142857
  Cheeger Constant:            0.166667
  Cheeger Bounds:              [0.071429, 0.534522]
  Eff. Resistance (est.):      84.0000

Community Structure
  Communities detected: 2
  Modularity:          0.5825
  Community sizes:     [6 6]

Interpretation
  ⚠️  Moderately connected — some cluster separation
  ✅ Strong community structure — natural clusters exist
```

### Fiedler Vector — Find the Cut

The Fiedler vector reveals the *natural bipartition* of your graph:

```go
result, _ := spectral.Fiedler(g)

// The Fiedler value (algebraic connectivity) tells you:
//   → High = well-connected graph, no easy cuts
//   → Low  = graph has a bottleneck / natural split
fmt.Printf("Algebraic connectivity: %.4f\n", result.FiedlerValue)

// Node assignments: which side of the cut is each node on?
for i, assignment := range result.NodeAssignments {
    fmt.Printf("Node %d → Cluster %d\n", i, assignment)
}
```

**Real example** — two 6-cliques joined by a single bridge edge:
```
Algebraic connectivity: 0.1429

Node  0 → Cluster 0    |  Node  6 → Cluster 1
Node  1 → Cluster 0    |  Node  7 → Cluster 1
Node  2 → Cluster 0    |  Node  8 → Cluster 1
Node  3 → Cluster 0    |  Node  9 → Cluster 1
Node  4 → Cluster 0    |  Node 10 → Cluster 1
Node  5 → Cluster 0    |  Node 11 → Cluster 1
```

### Cheeger Constant — Measure the Bottleneck

```go
h, _ := spectral.CheegerConstant(g)
lower, upper, _ := spectral.CheegerBound(g)

fmt.Printf("Cheeger constant: %.4f (bounds: [%.4f, %.4f])\n", h, lower, upper)
```

| Graph | Cheeger Constant | Meaning |
|-------|-----------------|---------|
| Complete K₆ | 3.0 | Perfectly connected — no bottlenecks |
| Path P₁₀ | 0.222 | Long chain — severe bottleneck |
| Two cliques + bridge | 0.167 | Clear bottleneck at bridge |
| Star S₅ | 0.714 | Hub is the only path |

### Community Detection — Find Your Clusters

```go
// Auto-detect number of communities (eigengap heuristic)
result, _ := spectral.DetectCommunities(g, 0)

// Or specify k explicitly
result, _ := spectral.DetectCommunities(g, 5)

fmt.Printf("Found %d communities, modularity %.4f\n",
    result.NumCommunities, result.Modularity)

for i, community := range result.Communities {
    fmt.Printf("  Cluster %d: %d nodes\n", i, len(community))
}
```

**Three clusters, auto-detected:**
```
Found 3 communities, modularity 0.5610
  Cluster 0: 4 nodes
  Cluster 1: 4 nodes
  Cluster 2: 4 nodes
```

### The Human-Readable Summary

```go
desc := spectral.DescribeCommunityStructure(1_200_000, 23, 8)
// "Your 1.2M vectors have 23 geometric clusters. You labeled 8."

desc := spectral.DescribeCommunityStructure(500, 5, 5)
// "Your 500 vectors have 5 geometric clusters. You labeled them all."

desc := spectral.DescribeCommunityStructure(10_000, 12, 0)
// "Your 10.0K vectors have 12 geometric clusters. None are labeled yet — spectral analysis found them."
```

## API Reference

### Core Types

| Type | Purpose |
|------|---------|
| `Graph` | Interface — implement this to wrap your HNSW index |
| `AdjacencyGraph` | In-memory graph for testing and standalone use |
| `FiedlerResult` | Fiedler vector, eigenvalues, bipartition assignments |
| `CommunityResult` | Communities, assignments, modularity |
| `EmbeddingQualityReport` | Full analysis with human-readable summary |

### Functions

```go
// Laplacian matrices
func Laplacian(g Graph) *mat.Dense
func NormalizedLaplacian(g Graph) *mat.Dense

// Fiedler analysis
func Fiedler(g Graph) (*FiedlerResult, error)

// Cheeger constant
func CheegerConstant(g Graph) (float64, error)
func CheegerBound(g Graph) (lower, upper float64, err error)

// Community detection
func DetectCommunities(g Graph, k int) (*CommunityResult, error)

// Full analysis
func AnalyzeEmbeddingQuality(g Graph) (*EmbeddingQualityReport, error)

// Human-readable descriptions
func DescribeCommunityStructure(numVectors, numCommunities, numLabels int) string

// Utilities
func SummaryStats(data []float64) (mean, std, min, max float64)
func IndexMap(g Graph) map[uint64]int
```

## How It Works

### Small graphs (≤ 256 nodes)
Full eigendecomposition of the normalized Laplacian via `gonum/mat`. Exact eigenvalues and eigenvectors.

### Large graphs (> 256 nodes)
Power iteration with orthogonalization against the all-ones vector (trivial eigenvector) and previously computed eigenvectors. 200 iterations for convergence.

### Cheeger constant
Sweep cut along the Fiedler vector — the classic approximation algorithm. O(n·d) where d is average degree.

### Community detection
1. Compute first *k* eigenvectors of the normalized Laplacian
2. Normalize rows of the embedding matrix
3. K-means++ with 50 iterations
4. Modularity scoring of the resulting partition

### Eigengap heuristic for auto-k
Find k that maximizes λ_{k+1} − λ_k among the first 10 eigenvalues.

## Performance

```
BenchmarkFiedlerSmall-8               4622    257019 ns/op
BenchmarkCommunityDetectionSmall-8    6540    181637 ns/op
BenchmarkAnalyzeEmbeddingQuality-8    2726    430840 ns/op
```

Small graphs (≤ 50 nodes) run in microseconds. For production HNSW indices with millions of vectors, you'd typically subsample or run on a per-shard basis.

## Integration with HNSW

Implement the `Graph` interface to wrap your Weaviate HNSW index:

```go
type HNSWSpectralGraph struct {
    index *hnsw.Index  // your Weaviate HNSW index
}

func (h *HNSWSpectralGraph) Nodes() []uint64 {
    // Iterate over index nodes
}

func (h *HNSWSpectralGraph) Neighbors(node uint64, level int) []uint64 {
    // Get connections at the given level
}

func (h *HNSWSpectralGraph) MaxLevel() int {
    // Return the current maximum level
}
```

Then pass it to any analysis function:

```go
report, _ := spectral.AnalyzeEmbeddingQuality(myGraph)
fmt.Println(report.Summary())
```

## Running Tests

```bash
go test ./spectral/... -v
```

22 tests covering:
- Laplacian construction (path, complete, star graphs)
- Fiedler vector correctness (cluster separation, eigenvalue bounds)
- Cheeger constant (bottleneck detection)
- Community detection (2-cluster, 3-cluster, auto-k, barbell)
- Full quality reports
- Edge cases (1 node, disconnected)

## License

Same as Weaviate — BSD-3-Clause. See LICENSE.
