package vectorintelligence

import (
	"fmt"
	"math"
)

// JensenShannonDivergence computes the Jensen-Shannon divergence between two
// probability distributions P and Q of equal length.
// JSD(P||Q) = 1/2*KL(P||M) + 1/2*KL(Q||M) where M = (P+Q)/2.
// Returns a value in [0, 1] for base-e logarithms.
func JensenShannonDivergence(p, q []float64) (float64, error) {
	if len(p) != len(q) {
		return 0, fmt.Errorf("distribution length mismatch: %d vs %d", len(p), len(q))
	}
	if len(p) == 0 {
		return 0, fmt.Errorf("empty distribution")
	}

	// Build M = (P+Q)/2
	m := make([]float64, len(p))
	for i := range p {
		m[i] = (p[i] + q[i]) / 2.0
	}

	klpm := klDiv(p, m)
	klqm := klDiv(q, m)

	jsd := (klpm + klqm) / 2.0
	return jsd, nil
}

func klDiv(p, q []float64) float64 {
	var kl float64
	for i := range p {
		if p[i] <= 0 {
			continue
		}
		if q[i] <= 0 {
			// If q[i] == 0 and p[i] > 0, divergence is +inf; cap at 100
			kl += 100.0 * p[i]
			continue
		}
		kl += p[i] * math.Log(p[i]/q[i])
	}
	return kl
}

// DistributionFromDistances builds a binned probability distribution from
// a set of distance values. Uses sqrt(n) bins with equal-width binning.
func DistributionFromDistances(distances []float64, numBins int) []float64 {
	if len(distances) == 0 {
		return []float64{1.0}
	}
	if numBins <= 0 {
		numBins = int(math.Sqrt(float64(len(distances)))) + 1
		if numBins < 3 {
			numBins = 3
		}
	}

	// Find min and max
	minD, maxD := distances[0], distances[0]
	for _, d := range distances {
		if d < minD {
			minD = d
		}
		if d > maxD {
			maxD = d
		}
	}

	if maxD-minD < 1e-15 {
		// All distances identical
		dist := make([]float64, numBins)
		dist[0] = 1.0
		return dist
	}

	binWidth := (maxD - minD) / float64(numBins)
	bins := make([]float64, numBins)
	for _, d := range distances {
		idx := int((d - minD) / binWidth)
		if idx >= numBins {
			idx = numBins - 1
		}
		bins[idx]++
	}

	// Normalize
	total := 0.0
	for _, v := range bins {
		total += v
	}
	for i := range bins {
		bins[i] /= total
	}
	return bins
}

// ClusterDistanceDistribution computes the distance distribution for each
// cluster: for each point in cluster X, the distances to all other points
// in the same cluster. Returns a slice of binned probability distributions.
func ClusterDistanceDistribution(vectors [][]float64, clusters [][]int, numBins int) ([][]float64, error) {
	distributions := make([][]float64, len(clusters))
	for ci, cluster := range clusters {
		if len(cluster) < 2 {
			distributions[ci] = []float64{1.0}
			continue
		}
		var dists []float64
		for _, i := range cluster {
			for _, j := range cluster {
				if i < j {
					dists = append(dists, CosineDist(vectors[i], vectors[j]))
				}
			}
		}
		distributions[ci] = DistributionFromDistances(dists, numBins)
	}
	return distributions, nil
}

// JSDMatrix computes the pairwise JSD between all cluster distributions.
// Returns a symmetric matrix where M[i][j] = JSD(cluster_i || cluster_j).
func JSDMatrix(distributions [][]float64) ([][]float64, error) {
	k := len(distributions)
	jsdMat := make([][]float64, k)
	for i := 0; i < k; i++ {
		jsdMat[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			if i == j {
				continue
			}
			jsd, err := JensenShannonDivergence(distributions[i], distributions[j])
			if err != nil {
				return nil, fmt.Errorf("jsd(%d,%d): %w", i, j, err)
			}
			jsdMat[i][j] = jsd
		}
	}
	return jsdMat, nil
}
