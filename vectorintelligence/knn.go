// Package vectorintelligence provides spectral graph analysis of embedding
// spaces to discover hidden cluster structure via spectral clustering,
// Cheeger constant estimation, and Jensen-Shannon divergence comparisons.
package vectorintelligence

import (
	"math"
	"sort"
)

// CosineDist computes cosine distance between two vectors of equal length.
// Returns 1 - dot(a,b)/(|a|*|b|). A zero vector pair yields distance 1.0.
func CosineDist(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	na := math.Sqrt(normA)
	nb := math.Sqrt(normB)
	if na == 0 || nb == 0 {
		return 1.0
	}
	return 1.0 - dot/(na*nb)
}

// kNNGraph builds a k-nearest neighbor graph from a set of vectors.
// Returns an adjacency list where nodes[i] is a sorted slice of neighbor
// indices (excluding self), ordered by distance ascending.
func kNNGraph(vectors [][]float64, k int) [][]int {
	n := len(vectors)
	if k <= 0 || k >= n {
		k = n - 1
	}
	adj := make([][]int, n)
	for i := 0; i < n; i++ {
		type neighbor struct {
			idx int
			dist float64
		}
		neighbors := make([]neighbor, 0, n-1)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			d := CosineDist(vectors[i], vectors[j])
			neighbors = append(neighbors, neighbor{idx: j, dist: d})
		}
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].dist < neighbors[b].dist
		})
		if k < len(neighbors) {
			neighbors = neighbors[:k]
		}
		adj[i] = make([]int, len(neighbors))
		for idx := range neighbors {
			adj[i][idx] = neighbors[idx].idx
		}
	}
	return adj
}

// CosineDistMatrix builds the full pairwise cosine distance matrix.
// Returns an n×n matrix where D[i][j] = cosine distance between vectors i and j.
func CosineDistMatrix(vectors [][]float64) [][]float64 {
	n := len(vectors)
	D := make([][]float64, n)
	for i := 0; i < n; i++ {
		D[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			D[i][j] = CosineDist(vectors[i], vectors[j])
		}
	}
	return D
}

// DistMatrixFromAdjacency builds a distance matrix from an adjacency list.
// Missing edges get distance 0 (used for Laplacian construction).
func DistMatrixFromAdjacency(adj [][]int) [][]float64 {
	n := len(adj)
	D := make([][]float64, n)
	for i := 0; i < n; i++ {
		D[i] = make([]float64, n)
		for _, j := range adj[i] {
			D[i][j] = 1.0
		}
	}
	return D
}
