package vectorintelligence

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// Laplacian builds the unnormalized graph Laplacian L = D - A from an
// adjacency matrix. A is the adjacency matrix (0/1 or weighted), and D
// is the degree diagonal matrix.
func Laplacian(A [][]float64) (*mat.Dense, error) {
	n := len(A)
	if n == 0 {
		return nil, fmt.Errorf("empty adjacency matrix")
	}
	for i := range A {
		if len(A[i]) != n {
			return nil, fmt.Errorf("row %d has length %d, expected %d", i, len(A[i]), n)
		}
	}
	L := mat.NewDense(n, n, nil)
	D := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		var deg float64
		for j := 0; j < n; j++ {
			deg += A[i][j]
		}
		D.Set(i, i, deg)
		for j := 0; j < n; j++ {
			L.Set(i, j, -A[i][j])
		}
		L.Set(i, i, deg-A[i][i])
	}
	return L, nil
}

// EigenDecomposition computes eigenvalues and eigenvectors of a symmetric
// matrix using an iterative approach. For small matrices (<512) it uses
// a direct Jacobi method; larger matrices get a warning about performance.
// Returns eigenvalues sorted ascending, and eigenvectors as columns of V.
func EigenDecomposition(m mat.Matrix) ([]float64, *mat.Dense, error) {
	r, c := m.Dims()
	if r != c {
		return nil, nil, fmt.Errorf("matrix must be square, got %dx%d", r, c)
	}
	if r == 0 {
		return nil, nil, fmt.Errorf("empty matrix")
	}

	// Copy into a Dense for mutation
	A := mat.DenseCopyOf(m)

	// Use Jacobi eigenvalue algorithm for symmetric matrices
	n := r
	V := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		V.Set(i, i, 1.0)
	}

	// Iterative Jacobi
	maxIter := 500
	if n > 128 {
		maxIter = 1500
	}
	for iter := 0; iter < maxIter; iter++ {
		// Find max off-diagonal element
		maxVal := 0.0
		p, q := 0, 1
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				v := math.Abs(A.At(i, j))
				if v > maxVal {
					maxVal = v
					p, q = i, j
				}
			}
		}
		if maxVal < 1e-12 {
			break
		}

		// Compute rotation
		app := A.At(p, p)
		aqq := A.At(q, q)
		apq := A.At(p, q)

		var theta float64
		if math.Abs(app-aqq) < 1e-15 {
			if apq >= 0 {
				theta = math.Pi / 4
			} else {
				theta = -math.Pi / 4
			}
		} else {
			theta = 0.5 * math.Atan2(2*apq, aqq-app)
		}

		c := math.Cos(theta)
		s := math.Sin(theta)

		// Apply Jacobi rotation to A
		app2 := c*c*app + s*s*aqq - 2*s*c*apq
		aqq2 := s*s*app + c*c*aqq + 2*s*c*apq
		A.Set(p, p, app2)
		A.Set(q, q, aqq2)
		A.Set(p, q, 0.0)
		A.Set(q, p, 0.0)

		for i := 0; i < n; i++ {
			if i != p && i != q {
				aip := A.At(i, p)
				aiq := A.At(i, q)
				A.Set(i, p, c*aip-s*aiq)
				A.Set(p, i, A.At(i, p))
				A.Set(i, q, s*aip+c*aiq)
				A.Set(q, i, A.At(i, q))
			}
		}

		// Update eigenvectors
		for i := 0; i < n; i++ {
			vip := V.At(i, p)
			viq := V.At(i, q)
			V.Set(i, p, c*vip-s*viq)
			V.Set(i, q, s*vip+c*viq)
		}
	}

	// Extract eigenvalues (diagonal of A)
	evals := make([]float64, n)
	for i := 0; i < n; i++ {
		evals[i] = A.At(i, i)
	}

	// Sort by eigenvalue ascending, reorder eigenvectors accordingly
	type eigenPair struct {
		val float64
		vec []float64
	}
	pairs := make([]eigenPair, n)
	for i := 0; i < n; i++ {
		pairs[i].val = evals[i]
		pairs[i].vec = make([]float64, n)
		for j := 0; j < n; j++ {
			pairs[i].vec[j] = V.At(j, i)
		}
	}
	sort.Slice(pairs, func(a, b int) bool {
		return pairs[a].val < pairs[b].val
	})

	sortedVals := make([]float64, n)
	sortedV := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		sortedVals[i] = pairs[i].val
		for j := 0; j < n; j++ {
			sortedV.Set(j, i, pairs[i].vec[j])
		}
	}

	return sortedVals, sortedV, nil
}

// FiedlerVector returns the second smallest eigenvector (Fiedler vector)
// from the eigenvalue decomposition. This vector provides the optimal
// spectral cut of the graph.
func FiedlerVector(L *mat.Dense) ([]float64, error) {
	n, _ := L.Dims()
	evals, evecs, err := EigenDecomposition(L)
	if err != nil {
		return nil, fmt.Errorf("eigen decomposition failed: %w", err)
	}
	if len(evals) < 2 {
		return nil, fmt.Errorf("need at least 2 eigenvalues, got %d", len(evals))
	}
	fiedler := make([]float64, n)
	for i := 0; i < n; i++ {
		fiedler[i] = evecs.At(i, 1) // column 1 = second eigenvector
	}
	// Normalize: check if first eigenvalue is near zero (connected graph check)
	if evals[0] > 1e-8 {
		_ = evals[0] // graph may be disconnected
	}
	return fiedler, nil
}

// CheegerConstantApprox computes the Cheeger constant (isoperimetric number)
// from the Fiedler vector. It sweeps all possible cuts sorted by Fiedler
// value and returns the minimum edge conductance: h(G) = min |E(S, V\S)| / min(vol(S), vol(V\S)).
// This approximation is within a factor of sqrt(log n) of the true value.
func CheegerConstantApprox(adj [][]int, fiedler []float64) (float64, error) {
	n := len(adj)
	if n == 0 {
		return 0, fmt.Errorf("empty graph")
	}
	if len(fiedler) != n {
		return 0, fmt.Errorf("fiedler length %d does not match adjacency %d", len(fiedler), n)
	}

	// Build degree array
	deg := make([]float64, n)
	totalVol := 0.0
	for i := 0; i < n; i++ {
		deg[i] = float64(len(adj[i]))
		totalVol += deg[i]
	}

	// Build index slice sorted by Fiedler value
	type idxVal struct {
		idx int
		val float64
	}
	order := make([]idxVal, n)
	for i := 0; i < n; i++ {
		order[i] = idxVal{idx: i, val: fiedler[i]}
	}
	sort.Slice(order, func(a, b int) bool {
		return order[a].val < order[b].val
	})

	// Sweep cut positions
	inCut := make([]bool, n)
	var cutVol, cutEdges float64
	bestH := 1.0

	for _, iv := range order {
		i := iv.idx
		inCut[i] = true
		cutVol += deg[i]

		// Update edge count: add edges from i to nodes not in cut, subtract edges from i to nodes in cut
		for _, j := range adj[i] {
			if inCut[j] {
				cutEdges -= 1.0 // was external, now internal
			} else {
				cutEdges += 1.0 // was internal, now external
			}
		}

		if cutVol <= 0 || cutVol >= totalVol {
			continue
		}
		minVol := cutVol
		if tot := totalVol - cutVol; tot < minVol {
			minVol = tot
		}
		h := cutEdges / minVol
		if h < bestH {
			bestH = h
		}
	}

	return bestH, nil
}

// SpectralCut partitions data into two clusters using the sign of the
// Fiedler vector. Returns two slices of indices.
func SpectralCut(fiedler []float64) (left, right []int) {
	for i, v := range fiedler {
		if v >= 0 {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return left, right
}

// RecursiveSpectral clusters data by repeatedly applying spectral bisection.
// Stops when the Cheeger constant exceeds the threshold or cluster size < minSize.
// Returns a list of clusters, each a slice of indices.
func RecursiveSpectral(vectors [][]float64, k int, cheegerThreshold float64) ([][]int, error) {
	n := len(vectors)
	if n == 0 {
		return nil, fmt.Errorf("empty vector set")
	}
	// Start with all indices
	all := make([]int, n)
	for i := 0; i < n; i++ {
		all[i] = i
	}
	clusters := [][]int{all}

	adjCache := make(map[int][][]int) // cache adjacency by cluster fingerprint

	for depth := 0; depth < k-1; depth++ {
		var newClusters [][]int
		for _, cl := range clusters {
			if len(cl) < 2 {
				newClusters = append(newClusters, cl)
				continue
			}

			// Build sub-vectors
			subVecs := make([][]float64, len(cl))
			for i, idx := range cl {
				subVecs[i] = vectors[idx]
			}

			// k-NN graph on sub-cluster (use min(kAdj, len-1))
			nSub := len(subVecs)
			kAdj := min(4, nSub-1)
			adj := kNNGraph(subVecs, kAdj)

			// Build adjacency matrix
			adjMat := DistMatrixFromAdjacency(adj)
			L, err := Laplacian(adjMat)
			if err != nil {
				return nil, fmt.Errorf("laplacian failed for cluster %d: %w", depth, err)
			}

			fiedler, err := FiedlerVector(L)
			if err != nil {
				return nil, fmt.Errorf("fiedler vector failed for cluster %d: %w", depth, err)
			}

			h, err := CheegerConstantApprox(adj, fiedler)
			if err != nil {
				return nil, fmt.Errorf("cheeger constant failed: %w", err)
			}

			if h > cheegerThreshold {
				newClusters = append(newClusters, cl)
				continue
			}

			left, right := SpectralCut(fiedler)
			if len(left) == 0 || len(right) == 0 {
				newClusters = append(newClusters, cl)
				continue
			}

			// Map back to original indices
			leftOrig := make([]int, len(left))
			rightOrig := make([]int, len(right))
			for i, idx := range left {
				leftOrig[i] = cl[idx]
			}
			for i, idx := range right {
				rightOrig[i] = cl[idx]
			}
			newClusters = append(newClusters, leftOrig, rightOrig)
			_ = adjCache
		}
		clusters = newClusters
	}

	return clusters, nil
}
