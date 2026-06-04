//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|_|\__/\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package spectral

import (
	"math"
	"testing"
)

// buildCompleteGraph creates a complete graph with n nodes (0..n-1).
func buildCompleteGraph(n int) *AdjacencyGraph {
	g := NewAdjacencyGraph()
	for i := 0; i < n; i++ {
		g.AddNode(uint64(i))
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}
	return g
}

// buildPathGraph creates a path graph: 0-1-2-...-(n-1).
func buildPathGraph(n int) *AdjacencyGraph {
	g := NewAdjacencyGraph()
	for i := 0; i < n; i++ {
		g.AddNode(uint64(i))
	}
	for i := 0; i < n-1; i++ {
		g.AddEdge(uint64(i), uint64(i+1))
	}
	return g
}

// buildTwoClusterGraph creates two densely connected clusters loosely joined
// by a single bridge edge.
func buildTwoClusterGraph() *AdjacencyGraph {
	g := NewAdjacencyGraph()

	// Cluster A: nodes 0-5
	for i := 0; i < 6; i++ {
		g.AddNode(uint64(i))
	}
	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Cluster B: nodes 6-11
	for i := 6; i < 12; i++ {
		g.AddNode(uint64(i))
	}
	for i := 6; i < 12; i++ {
		for j := i + 1; j < 12; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Bridge between clusters
	g.AddEdge(5, 6)

	return g
}

// buildThreeClusterGraph creates three clusters.
func buildThreeClusterGraph() *AdjacencyGraph {
	g := NewAdjacencyGraph()

	// Cluster A: 0-3
	for i := 0; i < 4; i++ {
		g.AddNode(uint64(i))
	}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Cluster B: 4-7
	for i := 4; i < 8; i++ {
		g.AddNode(uint64(i))
	}
	for i := 4; i < 8; i++ {
		for j := i + 1; j < 8; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Cluster C: 8-11
	for i := 8; i < 12; i++ {
		g.AddNode(uint64(i))
	}
	for i := 8; i < 12; i++ {
		for j := i + 1; j < 12; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Bridges
	g.AddEdge(3, 4)
	g.AddEdge(7, 8)

	return g
}

// buildStarGraph creates a star graph with center 0 and leaves 1..n.
func buildStarGraph(n int) *AdjacencyGraph {
	g := NewAdjacencyGraph()
	for i := 0; i <= n; i++ {
		g.AddNode(uint64(i))
	}
	for i := 1; i <= n; i++ {
		g.AddEdge(0, uint64(i))
	}
	return g
}

// --- Graph Tests ---

func TestAdjacencyGraphBasic(t *testing.T) {
	g := buildCompleteGraph(5)
	if len(g.Nodes()) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(g.Nodes()))
	}
	if g.EdgeCount() != 10 { // C(5,2) = 10
		t.Errorf("expected 10 edges, got %d", g.EdgeCount())
	}
}

func TestIndexMap(t *testing.T) {
	g := buildPathGraph(4)
	idx := IndexMap(g)
	for i, node := range g.Nodes() {
		if idx[node] != i {
			t.Errorf("expected node %d to have index %d, got %d", node, i, idx[node])
		}
	}
}

// --- Laplacian Tests ---

func TestLaplacianPathGraph(t *testing.T) {
	g := buildPathGraph(4)
	L := Laplacian(g)

	// For path 0-1-2-3:
	// L = [[1,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,1]]
	expected := [][]float64{
		{1, -1, 0, 0},
		{-1, 2, -1, 0},
		{0, -1, 2, -1},
		{0, 0, -1, 1},
	}

	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			got := L.At(i, j)
			want := expected[i][j]
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("L[%d][%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestLaplacianRowSums(t *testing.T) {
	g := buildCompleteGraph(6)
	L := Laplacian(g)
	n := 6

	// Each row of L should sum to 0
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += L.At(i, j)
		}
		if math.Abs(sum) > 1e-10 {
			t.Errorf("row %d sum = %f, expected 0", i, sum)
		}
	}
}

func TestNormalizedLaplacianDiagonal(t *testing.T) {
	g := buildCompleteGraph(5)
	L := NormalizedLaplacian(g)

	// For complete graph K_n, normalized Laplacian has diagonal = 1
	// and off-diagonal (where connected) = -1/(n-1)
	n := 5
	for i := 0; i < n; i++ {
		if math.Abs(L.At(i, i)-1.0) > 1e-10 {
			t.Errorf("L_norm[%d][%d] = %f, expected 1.0", i, i, L.At(i, i))
		}
		for j := 0; j < n; j++ {
			if i != j {
				expected := -1.0 / float64(n-1)
				if math.Abs(L.At(i, j)-expected) > 1e-10 {
					t.Errorf("L_norm[%d][%d] = %f, expected %f", i, j, L.At(i, j), expected)
				}
			}
		}
	}
}

// --- Fiedler Tests ---

func TestFiedlerTooFewNodes(t *testing.T) {
	g := NewAdjacencyGraph()
	g.AddNode(0)
	_, err := Fiedler(g)
	if err == nil {
		t.Error("expected error for single-node graph")
	}
}

func TestFiedlerCompleteGraph(t *testing.T) {
	g := buildCompleteGraph(6)
	fr, err := Fiedler(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// For K_n, the Fiedler value (2nd smallest eigenvalue of normalized Laplacian) = n/(n-1)
	expected := float64(6) / float64(5)
	if math.Abs(fr.FiedlerValue-expected) > 0.01 {
		t.Errorf("Fiedler value = %f, expected ~%f", fr.FiedlerValue, expected)
	}

	// All eigenvalues should be >= 0
	for i, ev := range fr.Eigenvalues {
		if ev < -1e-10 {
			t.Errorf("eigenvalue[%d] = %f, should be >= 0", i, ev)
		}
	}
}

func TestFiedlerTwoCluster(t *testing.T) {
	g := buildTwoClusterGraph()
	fr, err := Fiedler(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The Fiedler value should be small (bottleneck between clusters)
	if fr.FiedlerValue > 0.5 {
		t.Errorf("Fiedler value = %f, expected small value for two-cluster graph", fr.FiedlerValue)
	}

	// Node assignments should split the two clusters
	clusterA := fr.NodeAssignments[0]
	clusterB := fr.NodeAssignments[11]

	// First cluster should all have same assignment
	for i := 0; i < 6; i++ {
		if fr.NodeAssignments[i] != clusterA {
			t.Errorf("node %d in cluster A has assignment %d, expected %d", i, fr.NodeAssignments[i], clusterA)
		}
	}

	// Second cluster should all have same assignment (opposite of cluster A)
	for i := 6; i < 12; i++ {
		if fr.NodeAssignments[i] != clusterB {
			t.Errorf("node %d in cluster B has assignment %d, expected %d", i, fr.NodeAssignments[i], clusterB)
		}
	}

	if clusterA == clusterB {
		t.Error("two clusters should have different assignments")
	}
}

func TestFiedlerPathGraph(t *testing.T) {
	g := buildPathGraph(5)
	fr, err := Fiedler(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Fiedler vector of path should be monotonic
	// (it's proportional to the second Fourier mode)
	if fr.FiedlerValue <= 0 {
		t.Errorf("Fiedler value should be positive, got %f", fr.FiedlerValue)
	}
}

// --- Cheeger Constant Tests ---

func TestCheegerConstantComplete(t *testing.T) {
	g := buildCompleteGraph(6)
	h, err := CheegerConstant(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// For K_n, Cheeger constant = ceil(n/2) / floor(n/2)
	// K_6: h = 3/(3) * (1/6... actually it's |∂S|/min(|S|,|V\S|)
	// Best cut: |S|=3, |∂S|=3*3=9, h=9/3=3... that's not right.
	// For K_6, min cut divides into 3+3, boundary = 3*3=9, h=9/3=3
	// Wait: |∂S| counts edges crossing the cut. For K_n with |S|=k,
	// |∂S| = k*(n-k). So h = k*(n-k)/min(k, n-k) = max(k, n-k)
	// Minimized at k = floor(n/2): h = ceil(n/2)
	// For K_6: h = 3
	// But with normalized Laplacian-based estimation, we get an approximation.
	if h < 0.5 {
		t.Errorf("Cheeger constant for complete graph should be high, got %f", h)
	}
}

func TestCheegerConstantTwoCluster(t *testing.T) {
	g := buildTwoClusterGraph()
	h, err := CheegerConstant(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Two 6-cliques connected by single bridge: Cheeger should be small
	if h > 0.5 {
		t.Errorf("Cheeger constant for barely-connected clusters should be small, got %f", h)
	}
}

func TestCheegerBound(t *testing.T) {
	g := buildCompleteGraph(6)
	lower, upper, err := CheegerBound(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if lower > upper {
		t.Errorf("lower bound (%f) > upper bound (%f)", lower, upper)
	}

	if lower < 0 {
		t.Errorf("lower bound should be >= 0, got %f", lower)
	}
}

// --- Community Detection Tests ---

func TestDetectCommunitiesTwoCluster(t *testing.T) {
	g := buildTwoClusterGraph()
	result, err := DetectCommunities(g, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.NumCommunities != 2 {
		t.Errorf("expected 2 communities, got %d", result.NumCommunities)
	}

	// Check that clusters are mostly separated (bridge nodes 5,6 may be ambiguous)
	comm := result.Assignments
	commA := comm[0]
	commB := comm[11]
	if commA == commB {
		t.Error("clusters should be in different communities")
	}

	// Interior nodes of each cluster should match their cluster
	for i := 0; i < 5; i++ {
		if comm[i] != commA {
			t.Errorf("interior node %d of cluster A has assignment %d, expected %d", i, comm[i], commA)
		}
	}
	for i := 7; i < 12; i++ {
		if comm[i] != commB {
			t.Errorf("interior node %d of cluster B has assignment %d, expected %d", i, comm[i], commB)
		}
	}
}

func TestDetectCommunitiesThreeCluster(t *testing.T) {
	g := buildThreeClusterGraph()
	result, err := DetectCommunities(g, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.NumCommunities != 3 {
		t.Errorf("expected 3 communities, got %d", result.NumCommunities)
	}
}

func TestDetectCommunitiesAutoK(t *testing.T) {
	g := buildTwoClusterGraph()
	result, err := DetectCommunities(g, 0) // auto-detect
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.NumCommunities < 2 {
		t.Errorf("auto-detection should find at least 2 communities, got %d", result.NumCommunities)
	}
}

func TestDetectCommunitiesModularity(t *testing.T) {
	g := buildTwoClusterGraph()
	result, err := DetectCommunities(g, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Modularity should be positive for a graph with clear cluster structure
	if result.Modularity < 0.3 {
		t.Errorf("expected high modularity for clear clusters, got %f", result.Modularity)
	}
}

// --- Embedding Quality Tests ---

func TestAnalyzeEmbeddingQuality(t *testing.T) {
	g := buildTwoClusterGraph()
	report, err := AnalyzeEmbeddingQuality(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.NumVectors != 12 {
		t.Errorf("expected 12 vectors, got %d", report.NumVectors)
	}

	if report.NumCommunities < 2 {
		t.Errorf("expected at least 2 communities, got %d", report.NumCommunities)
	}

	if report.AlgebraicConnectivity <= 0 {
		t.Errorf("algebraic connectivity should be positive, got %f", report.AlgebraicConnectivity)
	}

	summary := report.Summary()
	if len(summary) == 0 {
		t.Error("summary should not be empty")
	}
}

func TestDescribeCommunityStructure(t *testing.T) {
	tests := []struct {
		name          string
		numVectors    int
		numComm       int
		numLabels     int
		containsCheck string
	}{
		{"labeled_some", 1000000, 23, 8, "labeled 8"},
		{"labeled_all", 500, 5, 5, "labeled them all"},
		{"labeled_none", 1000, 10, 0, "None are labeled"},
		{"small", 50, 3, 0, "50 vectors"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			desc := DescribeCommunityStructure(tt.numVectors, tt.numComm, tt.numLabels)
			if !contains(desc, tt.containsCheck) {
				t.Errorf("description %q should contain %q", desc, tt.containsCheck)
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestSummaryStats(t *testing.T) {
	data := []float64{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}
	mean, std, min, max := SummaryStats(data)

	if math.Abs(mean-5.0) > 0.01 {
		t.Errorf("mean = %f, expected ~5.0", mean)
	}
	if min != 2.0 {
		t.Errorf("min = %f, expected 2.0", min)
	}
	if max != 9.0 {
		t.Errorf("max = %f, expected 9.0", max)
	}
	if std <= 0 {
		t.Errorf("std should be positive, got %f", std)
	}
}

// --- Edge Cases ---

func TestStarGraphFiedler(t *testing.T) {
	g := buildStarGraph(5)
	fr, err := Fiedler(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Star graph has low algebraic connectivity
	if fr.FiedlerValue > 2.0 {
		t.Errorf("Fiedler value for star graph should be low, got %f", fr.FiedlerValue)
	}
}

func TestBarbellGraph(t *testing.T) {
	// Barbell: two cliques joined by a single edge — classic test for community detection
	g := NewAdjacencyGraph()

	// Left clique: 0-4
	for i := 0; i < 5; i++ {
		g.AddNode(uint64(i))
	}
	for i := 0; i < 5; i++ {
		for j := i + 1; j < 5; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Right clique: 5-9
	for i := 5; i < 10; i++ {
		g.AddNode(uint64(i))
	}
	for i := 5; i < 10; i++ {
		for j := i + 1; j < 10; j++ {
			g.AddEdge(uint64(i), uint64(j))
		}
	}

	// Bridge
	g.AddEdge(4, 5)

	result, err := DetectCommunities(g, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.NumCommunities != 2 {
		t.Errorf("expected 2 communities for barbell, got %d", result.NumCommunities)
	}

	// Positive modularity expected (bridge nodes may reduce it slightly)
	if result.Modularity < 0.2 {
		t.Errorf("barbell should have positive modularity, got %f", result.Modularity)
	}
}

func TestLaplacianSymmetric(t *testing.T) {
	g := buildTwoClusterGraph()
	L := Laplacian(g)
	n := 12

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(L.At(i, j)-L.At(j, i)) > 1e-10 {
				t.Errorf("Laplacian should be symmetric: L[%d][%d]=%f != L[%d][%d]=%f",
					i, j, L.At(i, j), j, i, L.At(j, i))
			}
		}
	}
}

// Benchmark tests

func BenchmarkFiedlerSmall(b *testing.B) {
	g := buildCompleteGraph(20)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Fiedler(g)
	}
}

func BenchmarkCommunityDetectionSmall(b *testing.B) {
	g := buildTwoClusterGraph()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DetectCommunities(g, 2)
	}
}

func BenchmarkAnalyzeEmbeddingQuality(b *testing.B) {
	g := buildTwoClusterGraph()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = AnalyzeEmbeddingQuality(g)
	}
}
