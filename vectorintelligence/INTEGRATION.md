# INTEGRATION.md — Using Vector Intelligence in Weaviate

This package lives at `vectorintelligence/` as a standalone Go module built into the Weaviate tree. It has no internal Weaviate dependencies — only `gonum.org/v1/gonum` (already a transitive dep).

## Integration Patterns

### 1. Standalone Analysis

```go
import vi "github.com/weaviate/weaviate/vectorintelligence"

vectors := loadYourVectors() // [][]float64 from any source

report, err := vi.RunAnalysis(vectors, 5, 0, 0.4, 10)
// k=5, auto-detect cluster count, cheeger threshold 0.4, 10 bins for JSD
```

### 2. Batch from File

```go
report, err := vi.RunAnalysis(vectors, 8, 5, 0.3, 0)
jsonBytes, _ := report.JSON()
os.WriteFile("analysis.json", jsonBytes, 0644)
```

### 3. Pipeline Integration (Calling from adapters)

```go
// In an adapter that processes query results:
func analyzeResults(vectors [][]float64) (*vi.AnalysisReport, error) {
    // Use sqrt(n) neighbors for large n, min 3
    k := int(math.Sqrt(float64(len(vectors))))
    if k < 3 { k = 3 }
    return vi.RunAnalysis(vectors, k, 0, 0.5, 0)
}
```

### 4. Custom k-NN Graph + Spectral Analysis

```go
// Step-by-step
adj := vi.KNNGraph(vectors, 5)
adjMat := vi.DistMatrixFromAdjacency(adj)
L, _ := vi.Laplacian(adjMat)
fiedler, _ := vi.FiedlerVector(L)
h, _ := vi.CheegerConstantApprox(adj, fiedler)
fmt.Printf("Cheeger constant: %f\n", h)

left, right := vi.SpectralCut(fiedler)
fmt.Printf("Cluster sizes: %d, %d\n", len(left), len(right))
```

### 5. Using JSD for Cluster Comparison

```go
clusters := [][]int{{0,1,2}, {3,4,5}, {6,7,8}}
dists, _ := vi.ClusterDistanceDistribution(vectors, clusters, 10)
jsdMat, _ := vi.JSDMatrix(dists)
for i := range jsdMat {
    for j := range jsdMat[i] {
        if i < j {
            fmt.Printf("JSD(cluster %d, cluster %d) = %f\n", i, j, jsdMat[i][j])
        }
    }
}
```

## Performance Notes

- **Small (<1000 vectors):** Full Jacobi eigenvalue decomposition — fast enough.
- **Large (1000+ vectors):** Consider dimension reduction before spectral analysis, or run on subsampled data.
- **k-NN is O(n²·d):** Currently a brute-force implementation. For production-scale embeddings, pre-filter or use approximate nearest neighbors.
- **Jacobi iteration** converges quadratically for symmetric matrices. Max iterations scale with matrix size (500 for n≤128, 1500 for larger).

## Adding to Weaviate Modules

If you want to expose this from a Weaviate module:

```go
// modules/my-module/analysis.go
package mymodule

import (
    "github.com/weaviate/weaviate/entities/vector"
    vi "github.com/weaviate/weaviate/vectorintelligence"
)

func AnalyzeVectors(vectors []vector.Vector) (*vi.AnalysisReport, error) {
    vv := make([][]float64, len(vectors))
    for i, v := range vectors {
        vv[i] = v
    }
    return vi.RunAnalysis(vv, 5, 0, 0.4, 0)
}
```

## Testing

```bash
cd /path/to/weaviate
go test -v ./vectorintelligence/
```
