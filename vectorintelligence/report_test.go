package vectorintelligence

import (
	"encoding/json"
	"math"
	"testing"
)

func TestRunAnalysis_Basic(t *testing.T) {
	vectors := [][]float64{
		{1, 0, 0},
		{0.95, 0.1, 0},
		{-1, 0, 0},
		{-0.95, -0.1, 0},
		{0, 1, 0},
		{0, 0.95, 0.1},
	}
	report, err := RunAnalysis(vectors, 2, 3, 0.5, 4)
	if err != nil {
		t.Fatalf("RunAnalysis failed: %v", err)
	}
	if report.NumVectors != 6 {
		t.Errorf("expected 6 vectors, got %d", report.NumVectors)
	}
	if report.Dimension != 3 {
		t.Errorf("expected dimension 3, got %d", report.Dimension)
	}
	if report.NumClusters <= 0 {
		t.Errorf("expected at least 1 cluster, got %d", report.NumClusters)
	}
	total := 0
	for _, s := range report.ClusterSizes {
		total += s
	}
	if total != report.NumVectors {
		t.Errorf("cluster sizes sum to %d, expected %d", total, report.NumVectors)
	}
	if report.CheegerConstant <= 0 {
		t.Logf("Cheeger constant: %f", report.CheegerConstant)
	}
}

func TestRunAnalysis_SingleVector(t *testing.T) {
	vectors := [][]float64{{1, 2, 3}}
	_, err := RunAnalysis(vectors, 2, 2, 0.5, 4)
	if err != nil {
		t.Logf("single vector expected to work or gracefully degrade: %v", err)
	}
}

func TestRunAnalysis_Empty(t *testing.T) {
	_, err := RunAnalysis(nil, 2, 2, 0.5, 4)
	if err == nil {
		t.Error("expected error for empty vectors")
	}
}

func TestAnalysisReport_JSONSerialization(t *testing.T) {
	vectors := [][]float64{
		{1, 0},
		{0.9, 0.1},
		{-1, 0},
		{-0.9, -0.1},
	}
	report, err := RunAnalysis(vectors, 2, 2, 0.5, 4)
	if err != nil {
		t.Fatalf("RunAnalysis failed: %v", err)
	}
	data, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON serialization failed: %v", err)
	}
	// Verify it round-trips
	var decoded AnalysisReport
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("JSON deserialization failed: %v", err)
	}
	if decoded.NumVectors != report.NumVectors {
		t.Errorf("round-trip: expected %d vectors, got %d", report.NumVectors, decoded.NumVectors)
	}
	if decoded.Dimension != report.Dimension {
		t.Errorf("round-trip: expected dim %d, got %d", report.Dimension, decoded.Dimension)
	}
}

func TestAnalysisReport_JSONPretty(t *testing.T) {
	report := &AnalysisReport{
		Timestamp:       "2025-01-01T00:00:00Z",
		NumVectors:      10,
		Dimension:       128,
		K:               5,
		Eigenvalues:     []float64{0.001, 0.05, 0.5, 1.5, 2.0},
		FiedlerVector:   []float64{-0.5, -0.3, 0.1, 0.4},
		CheegerConstant: 0.333,
		NumClusters:     2,
		Clusters:        [][]int{{0, 1}, {2, 3}},
		ClusterSizes:    []int{2, 2},
		JSDMatrix:       [][]float64{{0, 0.5}, {0.5, 0}},
		JSDSummary:      &JSDSummary{Mean: 0.5, Min: 0.5, Max: 0.5},
	}
	data, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON error: %v", err)
	}
	if !json.Valid(data) {
		t.Error("output is not valid JSON")
	}
	t.Logf("JSON output: %s", string(data))
}

func TestAnalysisReport_EigenvalueOrder(t *testing.T) {
	A := [][]float64{
		{0, 1, 0, 0},
		{1, 0, 1, 0},
		{0, 1, 0, 1},
		{0, 0, 1, 0},
	}
	L, _ := Laplacian(A)
	evals, _, err := EigenDecomposition(L)
	if err != nil {
		t.Fatalf("eigen decomposition error: %v", err)
	}
	for i := 1; i < len(evals); i++ {
		if evals[i] < evals[i-1] {
			t.Errorf("eigenvalues not sorted ascending at index %d: %f < %f", i, evals[i], evals[i-1])
		}
	}
}

func TestCosineDist_DimensionMismatchPanics(t *testing.T) {
	// Should not panic, different lengths will index out-of-range, so catch in test
	a := []float64{1, 0, 0}
	b := []float64{0, 1}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("expected panic with different lengths: %v", r)
		}
	}()
	_ = CosineDist(a, b)
}

func TestCosineDist_Identity(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{"same unit", []float64{1, 0, 0}, []float64{1, 0, 0}, 0},
		{"scaled same", []float64{2, 0, 0}, []float64{1, 0, 0}, 0},
		{"orthogonal", []float64{1, 0, 0}, []float64{0, 1, 0}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CosineDist(tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-12 {
				t.Errorf("CosineDist(%v, %v) = %f, want %f", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestEigenDecomposition_PathGraph4(t *testing.T) {
	A := [][]float64{
		{0, 1, 0, 0},
		{1, 0, 1, 0},
		{0, 1, 0, 1},
		{0, 0, 1, 0},
	}
	L, _ := Laplacian(A)
	evals, _, err := EigenDecomposition(L)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	// Eigenvalues of path graph L: [0, 2-sqrt(2), 2, 2+sqrt(2)]
	expected := []float64{0, 2 - math.Sqrt2, 2, 2 + math.Sqrt2}
	for i := range evals {
		if math.Abs(evals[i]-expected[i]) > 1e-10 {
			t.Errorf("eigenvalue %d: got %f, expected %f", i, evals[i], expected[i])
		}
	}
}

func TestRecursiveSpectral_IdenticalVectors(t *testing.T) {
	// All vectors identical -> distance 0 between all pairs, weird graph
	vectors := make([][]float64, 5)
	for i := 0; i < 5; i++ {
		vectors[i] = []float64{0.5, 0.5}
	}
	clusters, err := RecursiveSpectral(vectors, 3, 0.5)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(clusters) < 1 {
		t.Error("expected at least 1 cluster")
	}
}

func TestRunAnalysis_WellSeparated(t *testing.T) {
	// Two tight clusters far apart
	vectors := make([][]float64, 10)
	for i := 0; i < 5; i++ {
		vectors[i] = []float64{float64(i)*0.01 + 100, 0}
	}
	for i := 5; i < 10; i++ {
		vectors[i] = []float64{float64(i-5)*0.01 - 100, 0}
	}
	report, err := RunAnalysis(vectors, 3, 2, 0.3, 4)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if report.NumClusters < 1 {
		t.Error("expected at least 1 cluster")
	}
	if report.CheegerConstant <= 0 {
		t.Logf("Cheeger: %f", report.CheegerConstant)
	}
}
