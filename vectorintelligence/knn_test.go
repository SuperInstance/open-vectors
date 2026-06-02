package vectorintelligence

import (
	"math"
	"testing"
)

func TestCosineDistSame(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{1, 0, 0}
	d := CosineDist(a, b)
	if d > 1e-12 {
		t.Errorf("same vector should have 0 distance, got %f", d)
	}
}

func TestCosineDistOrthogonal(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{0, 1, 0}
	d := CosineDist(a, b)
	if math.Abs(d-1.0) > 1e-12 {
		t.Errorf("orthogonal vectors should have distance 1, got %f", d)
	}
}

func TestCosineDistOpposite(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{-1, 0, 0}
	d := CosineDist(a, b)
	if math.Abs(d-2.0) > 1e-12 {
		t.Errorf("opposite vectors should have distance 2, got %f", d)
	}
}

func TestCosineDistZero(t *testing.T) {
	a := []float64{0, 0, 0}
	b := []float64{1, 2, 3}
	d := CosineDist(a, b)
	if d != 1.0 {
		t.Errorf("zero vector should give distance 1, got %f", d)
	}
}

func TestCosineDistNearlyParallel(t *testing.T) {
	a := []float64{3, 4, 0}
	b := []float64{6, 8, 0.01}
	d := CosineDist(a, b)
	if d > 0.01 {
		t.Errorf("nearly parallel vectors should have small distance, got %f", d)
	}
}

func TestCosineDistMatrix(t *testing.T) {
	vectors := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{-1, 0, 0},
	}
	D := CosineDistMatrix(vectors)
	if len(D) != 3 {
		t.Fatalf("expected 3x3 matrix, got %dx%d", len(D), len(D[0]))
	}
	if D[0][0] != 0 || D[1][1] != 0 || D[2][2] != 0 {
		t.Error("diagonal should be zero")
	}
	if math.Abs(D[0][1]-1.0) > 1e-12 {
		t.Errorf("D[0][1] should be 1 (orthogonal), got %f", D[0][1])
	}
	if math.Abs(D[0][2]-2.0) > 1e-12 {
		t.Errorf("D[0][2] should be 2 (opposite), got %f", D[0][2])
	}
}

func TestKNNGraph_Small(t *testing.T) {
	vectors := [][]float64{
		{1, 0},
		{0, 1},
		{-1, 0},
		{0, -1},
	}
	adj := kNNGraph(vectors, 2)
	if len(adj) != 4 {
		t.Fatalf("expected 4 nodes, got %d", len(adj))
	}
	for i, neighbors := range adj {
		if len(neighbors) > 2 {
			t.Errorf("node %d has %d neighbors, expected at most 2", i, len(neighbors))
		}
		// No self-loops
		for _, nb := range neighbors {
			if nb == i {
				t.Errorf("node %d has self-loop", i)
			}
		}
	}
}

func TestKNNGraph_TooLargeK(t *testing.T) {
	vectors := [][]float64{
		{1, 0},
		{0, 1},
		{-1, 0},
	}
	// k=10 > n-1=2, so should clamp to n-1
	adj := kNNGraph(vectors, 10)
	for i, neighbors := range adj {
		if len(neighbors) != 2 {
			t.Errorf("node %d should have 2 neighbors, got %d", i, len(neighbors))
		}
		for _, nb := range neighbors {
			if nb == i {
				t.Errorf("node %d has self-loop", i)
			}
		}
	}
}

func TestDistMatrixFromAdjacency(t *testing.T) {
	adj := [][]int{
		{1, 2},
		{0, 2},
		{0, 1},
	}
	D := DistMatrixFromAdjacency(adj)
	if len(D) != 3 {
		t.Fatalf("expected 3x3, got %dx%d", len(D), len(D[0]))
	}
	if D[0][1] != 1.0 || D[0][2] != 1.0 {
		t.Errorf("adjacency edges should be 1.0")
	}
	if D[0][0] != 0 {
		t.Errorf("diagonal should be 0")
	}
	if D[1][0] != 1.0 || D[1][2] != 1.0 {
		t.Errorf("undirected should be symmetric")
	}
}

func TestLaplacian_Basic(t *testing.T) {
	A := [][]float64{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	}
	L, err := Laplacian(A)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	r, c := L.Dims()
	if r != 3 || c != 3 {
		t.Fatalf("expected 3x3, got %dx%d", r, c)
	}
	// L = D - A
	if L.At(0, 0) != 1.0 {
		t.Errorf("L[0,0] should be deg(0)=1, got %f", L.At(0, 0))
	}
	if L.At(1, 1) != 2.0 {
		t.Errorf("L[1,1] should be deg(1)=2, got %f", L.At(1, 1))
	}
	if L.At(0, 1) != -1.0 {
		t.Errorf("L[0,1] should be -1, got %f", L.At(0, 1))
	}
	if L.At(1, 0) != -1.0 {
		t.Errorf("L[1,0] should be -1, got %f", L.At(1, 0))
	}
}

func TestLaplacian_Empty(t *testing.T) {
	_, err := Laplacian([][]float64{})
	if err == nil {
		t.Error("expected error for empty adjacency")
	}
}

func TestLaplacian_Jagged(t *testing.T) {
	_, err := Laplacian([][]float64{
		{0, 1},
		{1, 0, 1},
	})
	if err == nil {
		t.Error("expected error for jagged matrix")
	}
}
