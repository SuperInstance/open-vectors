package vectorintelligence

import (
	"encoding/json"
	"fmt"

	"gonum.org/v1/gonum/mat"
	"time"
)

// AnalysisReport contains the full results of a spectral cluster analysis
// on an embedding space. It is serializable to JSON for downstream use.
type AnalysisReport struct {
	// Timestamp of analysis
	Timestamp string `json:"timestamp"`

	// Number of vectors analyzed
	NumVectors int `json:"num_vectors"`

	// Dimensionality of the embedding space
	Dimension int `json:"dimension"`

	// Number of nearest neighbors used in the k-NN graph
	K int `json:"k"`

	// Eigenvalues of the graph Laplacian (sorted ascending)
	Eigenvalues []float64 `json:"eigenvalues,omitempty"`

	// Fiedler vector (second eigenvector) values per data point
	FiedlerVector []float64 `json:"fiedler_vector,omitempty"`

	// Cheeger constant approximation
	CheegerConstant float64 `json:"cheeger_constant"`

	// Number of clusters found
	NumClusters int `json:"num_clusters"`

	// Cluster assignments: for each cluster, the indices of vectors in it
	Clusters [][]int `json:"clusters"`

	// Cluster sizes
	ClusterSizes []int `json:"cluster_sizes"`

	// Pairwise JSD between cluster distance distributions
	JSDMatrix [][]float64 `json:"jsd_matrix,omitempty"`

	// Summary of JSD: mean, min, max across off-diagonal entries
	JSDSummary *JSDSummary `json:"jsd_summary,omitempty"`
}

// JSDSummary summarizes the pairwise JSD values between clusters.
type JSDSummary struct {
	Mean float64 `json:"mean"`
	Min  float64 `json:"min"`
	Max  float64 `json:"max"`
}

// JSON serializes the AnalysisReport to indented JSON.
func (r *AnalysisReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RunAnalysis performs the full spectral analysis pipeline on a set of
// embedding vectors: builds a k-NN graph, computes the Laplacian,
// finds the Fiedler vector, approximates the Cheeger constant,
// performs recursive spectral bisection, computes cluster distance
// distributions, and calculates the JSD matrix between clusters.
//
// Parameters:
//   - vectors: slice of embedding vectors (each a []float64).
//   - k: number of nearest neighbors for the graph (0 or negative means auto).
//   - numClusters: target number of clusters for recursive bisection.
//   - cheegerThreshold: stop splitting clusters when Cheeger > this value.
//   - jsdBins: number of bins for distance distributions (0 = auto).
func RunAnalysis(vectors [][]float64, k, numClusters int, cheegerThreshold float64, jsdBins int) (*AnalysisReport, error) {
	n := len(vectors)
	if n == 0 {
		return nil, fmt.Errorf("no vectors provided")
	}
	dim := len(vectors[0])
	if dim == 0 {
		return nil, fmt.Errorf("vectors have zero dimension")
	}

	if k <= 0 || k >= n {
		k = min(8, n-1)
	}
	if numClusters <= 0 {
		numClusters = 2
	}
	if cheegerThreshold <= 0 {
		cheegerThreshold = 1.0
	}
	if jsdBins <= 0 {
		jsdBins = 0 // auto
	}

	// 1. Build k-NN graph
	adj := kNNGraph(vectors, k)
	adjMat := DistMatrixFromAdjacency(adj)

	// 2. Laplacian (handle n < 2 gracefully)
	var evals []float64
	var fiedler []float64
	var cheeger float64

	if n < 2 {
		evals = []float64{0}
		fiedler = []float64{0}
		cheeger = 1.0
	} else {
		L, err := Laplacian(adjMat)
		if err != nil {
			return nil, fmt.Errorf("laplacian construction: %w", err)
		}

		// 3. Eigenvalues and Fiedler vector
		var evecs *mat.Dense
		evals, evecs, err = EigenDecomposition(L)
		if err != nil {
			return nil, fmt.Errorf("eigen decomposition: %w", err)
		}

		fiedler = make([]float64, n)
		for i := 0; i < n; i++ {
			fiedler[i] = evecs.At(i, 1)
		}

		// 4. Cheeger constant
		cheeger, err = CheegerConstantApprox(adj, fiedler)
		if err != nil {
			return nil, fmt.Errorf("cheeger constant: %w", err)
		}
	}

	// 5. Recursive spectral clustering
	clusters, err := RecursiveSpectral(vectors, numClusters, cheegerThreshold)
	if err != nil {
		return nil, fmt.Errorf("recursive spectral clustering: %w", err)
	}

	clusterSizes := make([]int, len(clusters))
	for i, cl := range clusters {
		clusterSizes[i] = len(cl)
	}

	// 6. Cluster distance distributions and JSD matrix
	distributions, err := ClusterDistanceDistribution(vectors, clusters, jsdBins)
	if err != nil {
		return nil, fmt.Errorf("distance distributions: %w", err)
	}

	jsdMat, err := JSDMatrix(distributions)
	if err != nil {
		return nil, fmt.Errorf("jsd matrix: %w", err)
	}

	// 7. JSD summary
	var jsdMean, jsdMin, jsdMax float64
	var jsdCount int
	if len(jsdMat) > 1 {
		jsdMin = 1e10
		for i := 0; i < len(jsdMat); i++ {
			for j := 0; j < len(jsdMat); j++ {
				if i != j {
					v := jsdMat[i][j]
					jsdMean += v
					jsdCount++
					if v < jsdMin {
						jsdMin = v
					}
					if v > jsdMax {
						jsdMax = v
					}
				}
			}
		}
		if jsdCount > 0 {
			jsdMean /= float64(jsdCount)
		}
	}

	report := &AnalysisReport{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		NumVectors:      n,
		Dimension:       dim,
		K:               k,
		Eigenvalues:     evals,
		FiedlerVector:   fiedler,
		CheegerConstant: cheeger,
		NumClusters:     len(clusters),
		Clusters:        clusters,
		ClusterSizes:    clusterSizes,
		JSDMatrix:       jsdMat,
		JSDSummary: &JSDSummary{
			Mean: jsdMean,
			Min:  jsdMin,
			Max:  jsdMax,
		},
	}

	return report, nil
}
