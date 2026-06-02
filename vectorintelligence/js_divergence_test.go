package vectorintelligence

import (
	"math"
	"testing"
)

func TestJensenShannonDivergence_SameDist(t *testing.T) {
	p := []float64{0.5, 0.5}
	q := []float64{0.5, 0.5}
	jsd, err := JensenShannonDivergence(p, q)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if jsd > 1e-12 {
		t.Errorf("same distributions should have JSD=0, got %f", jsd)
	}
}

func TestJensenShannonDivergence_Different(t *testing.T) {
	p := []float64{0.9, 0.1}
	q := []float64{0.1, 0.9}
	jsd, err := JensenShannonDivergence(p, q)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if jsd <= 0 {
		t.Errorf("different distributions should have JSD>0, got %f", jsd)
	}
}

func TestJensenShannonDivergence_Symmetric(t *testing.T) {
	p := []float64{0.7, 0.3}
	q := []float64{0.2, 0.8}
	jsd1, _ := JensenShannonDivergence(p, q)
	jsd2, _ := JensenShannonDivergence(q, p)
	if math.Abs(jsd1-jsd2) > 1e-12 {
		t.Errorf("JSD should be symmetric: %f vs %f", jsd1, jsd2)
	}
}

func TestJensenShannonDivergence_Bounded(t *testing.T) {
	p := []float64{1, 0}
	q := []float64{0, 1}
	jsd, err := JensenShannonDivergence(p, q)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if jsd > math.Log(2) {
		t.Errorf("JSD should be bounded by ln(2)=%f, got %f", math.Log(2), jsd)
	}
}

func TestJensenShannonDivergence_LengthMismatch(t *testing.T) {
	_, err := JensenShannonDivergence([]float64{0.5, 0.5}, []float64{1.0})
	if err == nil {
		t.Error("expected error for length mismatch")
	}
}

func TestJensenShannonDivergence_Empty(t *testing.T) {
	_, err := JensenShannonDivergence(nil, nil)
	if err == nil {
		t.Error("expected error for empty distributions")
	}
}

func TestDistributionFromDistances_Uniform(t *testing.T) {
	dists := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	dist := DistributionFromDistances(dists, 5)
	if len(dist) != 5 {
		t.Fatalf("expected 5 bins, got %d", len(dist))
	}
	sum := 0.0
	for _, v := range dist {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("distribution should sum to 1, got %f", sum)
	}
}

func TestDistributionFromDistances_AllSame(t *testing.T) {
	dists := []float64{0.5, 0.5, 0.5, 0.5}
	dist := DistributionFromDistances(dists, 4)
	if dist[0] != 1.0 {
		t.Errorf("all same distances should put everything in bin 0")
	}
	for i := 1; i < len(dist); i++ {
		if dist[i] != 0 {
			t.Errorf("bin %d should be 0, got %f", i, dist[i])
		}
	}
}

func TestDistributionFromDistances_Empty(t *testing.T) {
	dist := DistributionFromDistances(nil, 5)
	if len(dist) != 1 || dist[0] != 1.0 {
		t.Errorf("empty input should return [1], got %v", dist)
	}
}

func TestDistributionFromDistances_AutoBins(t *testing.T) {
	dists := make([]float64, 100)
	for i := 0; i < 100; i++ {
		dists[i] = float64(i) / 100.0
	}
	dist := DistributionFromDistances(dists, 0)
	if len(dist) <= 0 {
		t.Errorf("auto-bins should produce at least 1 bin, got %d", len(dist))
	}
	sum := 0.0
	for _, v := range dist {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("should sum to 1, got %f", sum)
	}
}

func TestClusterDistanceDistribution(t *testing.T) {
	vectors := [][]float64{
		{1, 0},
		{0, 1},
		{-1, 0},
	}
	clusters := [][]int{
		{0, 1},
		{2},
	}
	dists, err := ClusterDistanceDistribution(vectors, clusters, 3)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(dists) != 2 {
		t.Fatalf("expected 2 distributions, got %d", len(dists))
	}
	// Cluster 1 has 1 element -> should be [1]
	if len(dists[1]) != 1 || dists[1][0] != 1.0 {
		t.Errorf("single-element cluster should have [1], got %v", dists[1])
	}
}

func TestJSDMatrix(t *testing.T) {
	distributions := [][]float64{
		{0.9, 0.1},
		{0.1, 0.9},
		{0.5, 0.5},
	}
	mat, err := JSDMatrix(distributions)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(mat) != 3 {
		t.Fatalf("expected 3x3, got %dx%d", len(mat), len(mat[0]))
	}
	// Diagonal should be zero
	for i := 0; i < 3; i++ {
		if mat[i][i] != 0 {
			t.Errorf("diagonal entry [%d][%d] should be 0, got %f", i, i, mat[i][i])
		}
	}
	// Symmetry
	if math.Abs(mat[0][1]-mat[1][0]) > 1e-12 {
		t.Errorf("JSD matrix should be symmetric: %f vs %f", mat[0][1], mat[1][0])
	}
	// Clusters 0 and 1 are very different (high JSD)
	if mat[0][1] <= mat[0][2] && mat[0][1] <= mat[1][2] {
		t.Logf("JSD matrix looks reasonable: %v", mat)
	}
}
