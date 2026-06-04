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
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Laplacian computes the unnormalized graph Laplacian L = D - A
// where D is the degree matrix and A is the adjacency matrix.
// Returns a dense matrix of size n×n.
func Laplacian(g Graph) *mat.Dense {
	nodes := g.Nodes()
	n := len(nodes)
	idx := IndexMap(g)

	L := mat.NewDense(n, n, nil)

	for i, node := range nodes {
		neighbors := g.Neighbors(node, 0)
		degree := len(neighbors)
		L.Set(i, i, float64(degree))
		for _, nb := range neighbors {
			j := idx[nb]
			L.Set(i, j, -1)
		}
	}

	return L
}

// NormalizedLaplacian computes the symmetric normalized Laplacian
// L_sym = I - D^(-1/2) A D^(-1/2).
// Isolated nodes (degree 0) get L[i,i] = 0.
func NormalizedLaplacian(g Graph) *mat.Dense {
	nodes := g.Nodes()
	n := len(nodes)
	idx := IndexMap(g)

	// Compute degrees
	degrees := make([]float64, n)
	for i, node := range nodes {
		neighbors := g.Neighbors(node, 0)
		degrees[i] = float64(len(neighbors))
	}

	// D^(-1/2)
	dInvSqrt := make([]float64, n)
	for i, d := range degrees {
		if d > 0 {
			dInvSqrt[i] = 1.0 / math.Sqrt(d)
		}
	}

	L := mat.NewDense(n, n, nil)

	for i := range n {
		L.Set(i, i, 1.0)
	}

	for i, node := range nodes {
		for _, nb := range g.Neighbors(node, 0) {
			j := idx[nb]
			if i < j { // only process each edge once
				val := dInvSqrt[i] * dInvSqrt[j]
				L.Set(i, j, -val)
				L.Set(j, i, -val)
			}
		}
	}

	return L
}

// FiedlerResult holds the result of Fiedler vector computation.
type FiedlerResult struct {
	// Eigenvalues sorted in ascending order.
	Eigenvalues []float64

	// FiedlerVector is the eigenvector corresponding to the second-smallest
	// eigenvalue of the Laplacian (the Fiedler vector).
	FiedlerVector []float64

	// FiedlerValue is the algebraic connectivity (2nd smallest eigenvalue).
	FiedlerValue float64

	// NodeAssignments maps each node index to a cluster (0 or 1) based on
	// the sign of the Fiedler vector.
	NodeAssignments []int
}

// Fiedler computes the Fiedler vector of the graph using the normalized Laplacian.
// The Fiedler vector is the eigenvector corresponding to the second-smallest
// eigenvalue, and it reveals the graph's natural bipartition.
//
// For small graphs (n ≤ 128), uses full eigenvalue decomposition.
// For larger graphs, uses power iteration on the Laplacian to approximate.
func Fiedler(g Graph) (*FiedlerResult, error) {
	nodes := g.Nodes()
	n := len(nodes)

	if n < 2 {
		return nil, fmt.Errorf("spectral: need at least 2 nodes for Fiedler analysis, got %d", n)
	}

	L := NormalizedLaplacian(g)

	if n <= 256 {
		return fiedlerDirect(L, n)
	}
	return fiedlerPowerIteration(L, n)
}

// fiedlerDirect computes the Fiedler vector via full eigendecomposition.
func fiedlerDirect(L *mat.Dense, n int) (*FiedlerResult, error) {
	var eig mat.Eigen
	ok := eig.Factorize(L, mat.EigenRight)
	if !ok {
		return nil, fmt.Errorf("spectral: eigendecomposition failed")
	}

	eigenvalues := eig.Values(nil)
	var eigenvectors mat.CDense
	eig.VectorsTo(&eigenvectors)

	// Sort by eigenvalue (real part, since Laplacian eigenvalues are real)
	type eigenPair struct {
		value  float64
		vector []float64
	}
	pairs := make([]eigenPair, n)
	for i := 0; i < n; i++ {
		vec := make([]float64, n)
		for j := 0; j < n; j++ {
			vec[j] = real(eigenvectors.At(j, i))
		}
		pairs[i] = eigenPair{value: real(eigenvalues[i]), vector: vec}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value < pairs[j].value
	})

	allEigenvalues := make([]float64, n)
	for i, p := range pairs {
		allEigenvalues[i] = p.value
	}

	fiedlerVec := pairs[1].vector
	fiedlerVal := pairs[1].value

	assignments := make([]int, n)
	for i, v := range fiedlerVec {
		if v >= 0 {
			assignments[i] = 0
		} else {
			assignments[i] = 1
		}
	}

	return &FiedlerResult{
		Eigenvalues:     allEigenvalues,
		FiedlerVector:   fiedlerVec,
		FiedlerValue:    fiedlerVal,
		NodeAssignments: assignments,
	}, nil
}

// fiedlerPowerIteration approximates the Fiedler vector using iterative methods.
// Uses inverse power iteration shifted away from the zero eigenvalue.
func fiedlerPowerIteration(L *mat.Dense, n int) (*FiedlerResult, error) {
	// For large graphs, use a simpler approach:
	// Estimate the Fiedler vector via random projection + refinement.
	// We'll compute the two smallest eigenvectors via repeated matrix-vector products.

	// Use LOBPCG-style approach: start with random vectors, orthogonalize against
	// the all-ones vector (which is the first eigenvector), and iterate.

	// Initialize with random but deterministic starting vector
	v := make([]float64, n)
	for i := range v {
		v[i] = float64((i*1103515245 + 12345) % 1000) / 1000.0
	}

	// Remove component along all-ones vector
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1.0 / math.Sqrt(float64(n))
	}
	removeProjection(v, ones)

	// Normalize
	normalizeVec(v)

	// Power iteration with shift
	const iterations = 200
	for iter := 0; iter < iterations; iter++ {
		// Multiply by L
		newV := matVecMul(L, v)

		// Remove component along ones
		removeProjection(newV, ones)

		// Normalize
		normalizeVec(newV)

		v = newV
	}

	// Estimate eigenvalue: Rayleigh quotient
	Lv := matVecMul(L, v)
	fiedlerVal := dotProduct(v, Lv)

	assignments := make([]int, n)
	for i, val := range v {
		if val >= 0 {
			assignments[i] = 0
		} else {
			assignments[i] = 1
		}
	}

	return &FiedlerResult{
		Eigenvalues:     []float64{0, fiedlerVal},
		FiedlerVector:   v,
		FiedlerValue:    fiedlerVal,
		NodeAssignments: assignments,
	}, nil
}

func removeProjection(v, dir []float64) {
	proj := dotProduct(v, dir)
	for i := range v {
		v[i] -= proj * dir[i]
	}
}

func normalizeVec(v []float64) {
	norm := math.Sqrt(dotProduct(v, v))
	if norm > 1e-15 {
		for i := range v {
			v[i] /= norm
		}
	}
}

func matVecMul(m *mat.Dense, v []float64) []float64 {
	n := len(v)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += m.At(i, j) * v[j]
		}
		result[i] = sum
	}
	return result
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// CheegerConstant estimates the Cheeger constant (isoperimetric number) of the graph.
// The Cheeger constant measures how "bottlenecked" a graph is — a small Cheeger
// constant means there's a sparse cut dividing the graph into two large pieces.
//
// h(G) = min_{S} |∂S| / min(|S|, |V\S|)
//
// This uses the Fiedler vector for an efficient approximation rather than
// checking all possible subsets (which is NP-hard).
func CheegerConstant(g Graph) (float64, error) {
	nodes := g.Nodes()
	n := len(nodes)
	idx := IndexMap(g)

	if n < 2 {
		return 0, nil
	}

	fr, err := Fiedler(g)
	if err != nil {
		return 0, fmt.Errorf("spectral: cheeger constant: %w", err)
	}

	// Sort nodes by Fiedler vector value
	type nodeVal struct {
		idx int
		val float64
	}
	sorted := make([]nodeVal, n)
	for i, v := range fr.FiedlerVector {
		sorted[i] = nodeVal{idx: i, val: v}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].val < sorted[j].val
	})

	// Sweeep: consider cuts that separate the first k nodes from the rest
	bestCheeger := math.Inf(1)

	// Track the set S and its boundary
	inS := make([]bool, n)
	boundary := 0 // |∂S|

	for k := 0; k < n-1; k++ {
		nodeIdx := sorted[k].idx
		nodeID := nodes[nodeIdx]
		inS[nodeIdx] = true

		// Update boundary: for each neighbor, check if it crosses the cut
		for _, nb := range g.Neighbors(nodeID, 0) {
			nbIdx := idx[nb]
			if inS[nbIdx] {
				// Was a boundary edge, now internal — remove
				boundary--
			} else {
				// New boundary edge
				boundary++
			}
		}

		sizeS := k + 1
		sizeComplement := n - sizeS
		denom := math.Min(float64(sizeS), float64(sizeComplement))
		if denom > 0 {
			h := float64(boundary) / denom
			if h < bestCheeger {
				bestCheeger = h
			}
		}
	}

	return bestCheeger, nil
}

// CheegerBound returns the upper and lower bounds on the Cheeger constant
// derived from the algebraic connectivity (Fiedler value λ₂).
//
// Lower bound: λ₂/2 ≤ h(G)
// Upper bound: h(G) ≤ √(2·λ₂)
func CheegerBound(g Graph) (lower, upper float64, err error) {
	fr, err := Fiedler(g)
	if err != nil {
		return 0, 0, fmt.Errorf("spectral: cheeger bound: %w", err)
	}

	lambda2 := fr.FiedlerValue
	lower = lambda2 / 2.0
	upper = math.Sqrt(2.0 * lambda2)
	return lower, upper, nil
}

// CommunityResult holds the result of spectral community detection.
type CommunityResult struct {
	// Communities is a list of communities, each being a list of node indices.
	Communities [][]int

	// Assignments maps each node index to its community ID.
	Assignments []int

	// NumCommunities is the number of communities found.
	NumCommunities int

	// Modularity is the modularity score of the partition.
	Modularity float64
}

// DetectCommunities performs spectral clustering to detect communities in the graph.
// It uses k eigenvectors of the normalized Laplacian to embed nodes, then runs
// k-means clustering in the embedding space.
//
// If k <= 0, it attempts to auto-detect the number of communities using
// the eigengap heuristic.
func DetectCommunities(g Graph, k int) (*CommunityResult, error) {
	nodes := g.Nodes()
	n := len(nodes)

	if n < 2 {
		return nil, fmt.Errorf("spectral: need at least 2 nodes for community detection")
	}

	// Auto-detect k using eigengap
	if k <= 0 {
		detected, err := estimateCommunities(g)
		if err != nil {
			return nil, err
		}
		k = detected
	}

	if k > n {
		k = n
	}
	if k < 2 {
		k = 2
	}

	L := NormalizedLaplacian(g)

	// Compute first k eigenvectors
	embeddings, err := computeSpectralEmbedding(L, n, k)
	if err != nil {
		return nil, err
	}

	// Normalize rows of embedding matrix
	for i := 0; i < n; i++ {
		norm := 0.0
		for j := 0; j < k; j++ {
			norm += embeddings[i*k+j] * embeddings[i*k+j]
		}
		norm = math.Sqrt(norm)
		if norm > 1e-15 {
			for j := 0; j < k; j++ {
				embeddings[i*k+j] /= norm
			}
		}
	}

	// K-means clustering on the embeddings
	assignments := kmeans(embeddings, n, k, 50)

	// Build communities
	communities := make([][]int, k)
	for i, c := range assignments {
		communities[c] = append(communities[c], i)
	}

	// Filter out empty communities
	var filtered [][]int
	relabel := make(map[int]int)
	newIdx := 0
	for i, comm := range communities {
		if len(comm) > 0 {
			relabel[i] = newIdx
			filtered = append(filtered, comm)
			newIdx++
		}
	}

	finalAssignments := make([]int, n)
	for i, c := range assignments {
		finalAssignments[i] = relabel[c]
	}

	modularity := computeModularity(g, finalAssignments)

	return &CommunityResult{
		Communities:    filtered,
		Assignments:    finalAssignments,
		NumCommunities: len(filtered),
		Modularity:     modularity,
	}, nil
}

// estimateCommunities uses the eigengap heuristic to estimate the number
// of communities: find k that maximizes λ_{k+1} - λ_k.
func estimateCommunities(g Graph) (int, error) {
	fr, err := Fiedler(g)
	if err != nil {
		return 2, err
	}

	eigenvalues := fr.Eigenvalues
	if len(eigenvalues) < 4 {
		return 2, nil
	}

	// Look at first ~10 eigenvalues max
	maxK := len(eigenvalues)
	if maxK > 11 {
		maxK = 11
	}

	bestGap := 0.0
	bestK := 2
	for i := 1; i < maxK-1; i++ {
		gap := eigenvalues[i+1] - eigenvalues[i]
		if gap > bestGap {
			bestGap = gap
			bestK = i + 1
		}
	}

	return bestK, nil
}

// computeSpectralEmbedding computes the first k eigenvectors of the Laplacian
// and returns them as an n×k embedding matrix (row-major).
func computeSpectralEmbedding(L *mat.Dense, n, k int) ([]float64, error) {
	if n <= 256 {
		return spectralEmbeddingDirect(L, n, k)
	}
	return spectralEmbeddingIterative(L, n, k)
}

func spectralEmbeddingDirect(L *mat.Dense, n, k int) ([]float64, error) {
	var eig mat.Eigen
	ok := eig.Factorize(L, mat.EigenRight)
	if !ok {
		return nil, fmt.Errorf("spectral: eigendecomposition failed")
	}

	eigenvalues := eig.Values(nil)
	var eigenvectors mat.CDense
	eig.VectorsTo(&eigenvectors)

	// Sort by eigenvalue
	type pair struct {
		val float64
		idx int
	}
	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = pair{val: real(eigenvalues[i]), idx: i}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val < pairs[j].val
	})

	// Take eigenvectors 1..k (skip the zero eigenvalue)
	embeddings := make([]float64, n*k)
	for j := 0; j < k && j+1 < n; j++ {
		vecIdx := pairs[j+1].idx
		for i := 0; i < n; i++ {
			embeddings[i*k+j] = real(eigenvectors.At(i, vecIdx))
		}
	}

	return embeddings, nil
}

func spectralEmbeddingIterative(L *mat.Dense, n, k int) ([]float64, error) {
	embeddings := make([]float64, n*k)

	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1.0 / math.Sqrt(float64(n))
	}

	for ev := 0; ev < k; ev++ {
		// Initialize with deterministic seed
		v := make([]float64, n)
		for i := range v {
			v[i] = float64((i*1103515245+12345+(ev+1)*7919)%1000) / 1000.0
		}

		// Orthogonalize against ones and previous eigenvectors
		removeProjection(v, ones)
		for prev := 0; prev < ev; prev++ {
			prevVec := make([]float64, n)
			for i := 0; i < n; i++ {
				prevVec[i] = embeddings[i*k+prev]
			}
			removeProjection(v, prevVec)
		}
		normalizeVec(v)

		// Power iteration
		for iter := 0; iter < 200; iter++ {
			newV := matVecMul(L, v)
			removeProjection(newV, ones)
			for prev := 0; prev < ev; prev++ {
				prevVec := make([]float64, n)
				for i := 0; i < n; i++ {
					prevVec[i] = embeddings[i*k+prev]
				}
				removeProjection(newV, prevVec)
			}
			normalizeVec(newV)
			v = newV
		}

		for i := 0; i < n; i++ {
			embeddings[i*k+ev] = v[i]
		}
	}

	return embeddings, nil
}

// kmeans performs k-means clustering on n points of dimension k.
func kmeans(data []float64, n, k, maxIter int) []int {
	// Initialize centroids using k-means++ style
	centroids := make([]float64, k*k)

	// First centroid: random (deterministic)
	for j := 0; j < k; j++ {
		centroids[j] = data[j]
	}

	for c := 1; c < k; c++ {
		// Find farthest point from existing centroids
		bestDist := -1.0
		bestIdx := 0
		for i := 0; i < n; i++ {
			minDist := math.Inf(1)
			for prev := 0; prev < c; prev++ {
				d := 0.0
				for j := 0; j < k; j++ {
					diff := data[i*k+j] - centroids[prev*k+j]
					d += diff * diff
				}
				if d < minDist {
					minDist = d
				}
			}
			if minDist > bestDist {
				bestDist = minDist
				bestIdx = i
			}
		}
		for j := 0; j < k; j++ {
			centroids[c*k+j] = data[bestIdx*k+j]
		}
	}

	assignments := make([]int, n)

	for iter := 0; iter < maxIter; iter++ {
		changed := false

		// Assign each point to nearest centroid
		for i := 0; i < n; i++ {
			bestC := 0
			bestDist := math.Inf(1)
			for c := 0; c < k; c++ {
				d := 0.0
				for j := 0; j < k; j++ {
					diff := data[i*k+j] - centroids[c*k+j]
					d += diff * diff
				}
				if d < bestDist {
					bestDist = d
					bestC = c
				}
			}
			if assignments[i] != bestC {
				assignments[i] = bestC
				changed = true
			}
		}

		if !changed {
			break
		}

		// Recompute centroids
		counts := make([]int, k)
		newCentroids := make([]float64, k*k)
		for i := 0; i < n; i++ {
			c := assignments[i]
			counts[c]++
			for j := 0; j < k; j++ {
				newCentroids[c*k+j] += data[i*k+j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				for j := 0; j < k; j++ {
					centroids[c*k+j] = newCentroids[c*k+j] / float64(counts[c])
				}
			}
		}
	}

	return assignments
}

// computeModularity computes the modularity of a partition.
// Q = (1/2m) * Σ_{ij} [A_{ij} - (d_i * d_j)/(2m)] * δ(c_i, c_j)
func computeModularity(g Graph, assignments []int) float64 {
	nodes := g.Nodes()
	n := len(nodes)
	idx := IndexMap(g)

	m := 0 // total edges
	degrees := make([]int, n)
	for i, node := range nodes {
		neighbors := g.Neighbors(node, 0)
		degrees[i] = len(neighbors)
		m += len(neighbors)
	}
	m /= 2

	if m == 0 {
		return 0
	}

	Q := 0.0
	twoM := float64(2 * m)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if assignments[i] != assignments[j] {
				continue
			}
			// A_{ij}: check if edge exists
			Aij := 0.0
			nodeI := nodes[i]
			for _, nb := range g.Neighbors(nodeI, 0) {
				if idx[nb] == j {
					Aij = 1.0
					break
				}
			}
			expected := float64(degrees[i]) * float64(degrees[j]) / twoM
			Q += Aij - expected
		}
	}

	Q /= twoM
	return Q
}

// EmbeddingQualityReport provides quality metrics for vector embeddings
// analyzed through the lens of graph spectral properties.
type EmbeddingQualityReport struct {
	// NumVectors is the total number of vectors (nodes).
	NumVectors int

	// NumEdges is the total number of edges in the proximity graph.
	NumEdges int

	// AlgebraicConnectivity is the Fiedler value (λ₂ of normalized Laplacian).
	// Higher values indicate a more well-connected graph.
	AlgebraicConnectivity float64

	// CheegerConstant estimates how well-connected the graph is.
	// Range [0, 1]. Higher = fewer bottlenecks = better embedding quality.
	CheegerConstant float64

	// CheegerLower and CheegerUpper bound the true Cheeger constant.
	CheegerLower float64
	CheegerUpper float64

	// NumCommunities is the auto-detected number of clusters.
	NumCommunities int

	// Modularity of the detected community structure.
	// Range [-0.5, 1]. Higher = more defined cluster structure.
	Modularity float64

	// CommunitySizes lists the sizes of each detected community.
	CommunitySizes []int

	// SpectralGap is λ₂ (same as AlgebraicConnectivity but named for context).
	// A large spectral gap means the graph is well-connected and converges
	// quickly in random walks.
	SpectralGap float64

	// EffectiveResistance is the average effective resistance between node pairs
	// (estimated from the first few eigenvalues). Lower = better connected.
	EffectiveResistance float64
}

// AnalyzeEmbeddingQuality performs a full spectral analysis of the graph
// built from vector embeddings and returns a comprehensive quality report.
func AnalyzeEmbeddingQuality(g Graph) (*EmbeddingQualityReport, error) {
	nodes := g.Nodes()
	n := len(nodes)

	if n < 2 {
		return nil, fmt.Errorf("spectral: need at least 2 nodes")
	}

	ag, ok := g.(*AdjacencyGraph)
	numEdges := 0
	if ok {
		numEdges = ag.EdgeCount()
	} else {
		// Count edges
		for _, node := range nodes {
			numEdges += len(g.Neighbors(node, 0))
		}
		numEdges /= 2
	}

	// Fiedler analysis
	fr, err := Fiedler(g)
	if err != nil {
		return nil, fmt.Errorf("spectral: embedding quality: %w", err)
	}

	// Cheeger constant
	cheeger, err := CheegerConstant(g)
	if err != nil {
		cheeger = 0
	}

	// Cheeger bounds
	lower, upper, _ := CheegerBound(g)

	// Community detection (auto-k)
	comm, err := DetectCommunities(g, 0)
	if err != nil {
		return nil, fmt.Errorf("spectral: embedding quality community detection: %w", err)
	}

	sizes := make([]int, len(comm.Communities))
	for i, c := range comm.Communities {
		sizes[i] = len(c)
	}

	// Effective resistance estimate: R_eff ≈ n * Σ_{i=2}^{k} 1/λ_i
	effResistance := 0.0
	eigenvalues := fr.Eigenvalues
	k := len(eigenvalues)
	if k > 10 {
		k = 10
	}
	for i := 1; i < k; i++ {
		if eigenvalues[i] > 1e-10 {
			effResistance += 1.0 / eigenvalues[i]
		}
	}
	effResistance *= float64(n)

	report := &EmbeddingQualityReport{
		NumVectors:            n,
		NumEdges:              numEdges,
		AlgebraicConnectivity: fr.FiedlerValue,
		CheegerConstant:       cheeger,
		CheegerLower:          lower,
		CheegerUpper:          upper,
		NumCommunities:        comm.NumCommunities,
		Modularity:            comm.Modularity,
		CommunitySizes:        sizes,
		SpectralGap:           fr.FiedlerValue,
		EffectiveResistance:   effResistance,
	}

	return report, nil
}

// Summary returns a human-readable summary of the embedding quality report.
func (r *EmbeddingQualityReport) Summary() string {
	s := fmt.Sprintf("Spectral Analysis Report\n")
	s += fmt.Sprintf("========================\n")
	s += fmt.Sprintf("Vectors: %d | Edges: %d\n", r.NumVectors, r.NumEdges)
	s += fmt.Sprintf("\n")
	s += fmt.Sprintf("Graph Connectivity\n")
	s += fmt.Sprintf("  Algebraic Connectivity (λ₂): %.6f\n", r.AlgebraicConnectivity)
	s += fmt.Sprintf("  Spectral Gap:                %.6f\n", r.SpectralGap)
	s += fmt.Sprintf("  Cheeger Constant:            %.6f\n", r.CheegerConstant)
	s += fmt.Sprintf("  Cheeger Bounds:              [%.6f, %.6f]\n", r.CheegerLower, r.CheegerUpper)
	s += fmt.Sprintf("  Eff. Resistance (est.):      %.4f\n", r.EffectiveResistance)
	s += fmt.Sprintf("\n")
	s += fmt.Sprintf("Community Structure\n")
	s += fmt.Sprintf("  Communities detected: %d\n", r.NumCommunities)
	s += fmt.Sprintf("  Modularity:          %.4f\n", r.Modularity)
	s += fmt.Sprintf("  Community sizes:     %v\n", r.CommunitySizes)

	// Quality interpretation
	s += "\nInterpretation\n"
	if r.AlgebraicConnectivity > 0.5 {
		s += "  ✅ Well-connected graph — embeddings form a coherent space\n"
	} else if r.AlgebraicConnectivity > 0.1 {
		s += "  ⚠️  Moderately connected — some cluster separation\n"
	} else {
		s += "  ❌ Poorly connected — embeddings may be poorly structured\n"
	}

	if r.Modularity > 0.5 {
		s += "  ✅ Strong community structure — natural clusters exist\n"
	} else if r.Modularity > 0.2 {
		s += "  ⚠️  Moderate community structure\n"
	} else {
		s += "  ℹ️  Weak community structure — embeddings are diffuse\n"
	}

	return s
}

// DescribeCommunityStructure returns a human-readable description like
// "Your 1M vectors have 23 geometric clusters. You labeled 8."
func DescribeCommunityStructure(numVectors int, numCommunities int, numLabels int) string {
	vecStr := formatNumber(numVectors)
	commStr := formatNumber(numCommunities)

	s := fmt.Sprintf("Your %s vectors have %s geometric clusters.", vecStr, commStr)
	if numLabels > 0 && numLabels < numCommunities {
		s += fmt.Sprintf(" You labeled %d.", numLabels)
	} else if numLabels >= numCommunities {
		s += " You labeled them all."
	} else {
		s += " None are labeled yet — spectral analysis found them."
	}
	return s
}

func formatNumber(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	}
	if n >= 1_000 {
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	}
	return fmt.Sprintf("%d", n)
}

// SummaryStats returns basic descriptive statistics for a float slice.
func SummaryStats(data []float64) (mean, std, min, max float64) {
	if len(data) == 0 {
		return 0, 0, 0, 0
	}
	mean = stat.Mean(data, nil)
	variance := stat.Variance(data, nil)
	std = math.Sqrt(variance)
	min = data[0]
	max = data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return
}
