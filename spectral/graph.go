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

// Graph is the adjacency interface that spectral analysis operates on.
// Implementations can wrap an HNSW index, a test graph, or any graph
// represented as uint64-node IDs with unweighted edges.
type Graph interface {
	// Nodes returns all node IDs in the graph.
	Nodes() []uint64

	// Neighbors returns the neighbor IDs of the given node at the given level.
	// For single-level graphs, use level 0.
	Neighbors(node uint64, level int) []uint64

	// MaxLevel returns the highest level in the graph.
	MaxLevel() int
}

// AdjacencyGraph is a simple in-memory Graph implementation useful for
// testing and as a building block for spectral analysis.
type AdjacencyGraph struct {
	nodes    []uint64
	adjList  map[uint64][]uint64
	maxLevel int
}

// NewAdjacencyGraph creates a new empty adjacency graph.
func NewAdjacencyGraph() *AdjacencyGraph {
	return &AdjacencyGraph{
		adjList: make(map[uint64][]uint64),
	}
}

// AddNode adds a node to the graph.
func (g *AdjacencyGraph) AddNode(id uint64) {
	g.nodes = append(g.nodes, id)
	if _, exists := g.adjList[id]; !exists {
		g.adjList[id] = nil
	}
}

// AddEdge adds an undirected edge between two nodes.
func (g *AdjacencyGraph) AddEdge(a, b uint64) {
	g.adjList[a] = append(g.adjList[a], b)
	g.adjList[b] = append(g.adjList[b], a)
}

// Nodes returns all node IDs.
func (g *AdjacencyGraph) Nodes() []uint64 {
	return g.nodes
}

// Neighbors returns the neighbors of a node (level is ignored for this impl).
func (g *AdjacencyGraph) Neighbors(node uint64, level int) []uint64 {
	return g.adjList[node]
}

// MaxLevel returns the max level (always 0 for this impl).
func (g *AdjacencyGraph) MaxLevel() int {
	return g.maxLevel
}

// SetMaxLevel sets the max level value.
func (g *AdjacencyGraph) SetMaxLevel(l int) {
	g.maxLevel = l
}

// Degree returns the number of neighbors for a node.
func (g *AdjacencyGraph) Degree(node uint64) int {
	return len(g.adjList[node])
}

// EdgeCount counts total undirected edges.
func (g *AdjacencyGraph) EdgeCount() int {
	count := 0
	for _, neighbors := range g.adjList {
		count += len(neighbors)
	}
	return count / 2
}

// IndexMap creates a mapping from node ID to a 0-based index for matrix ops.
func IndexMap(g Graph) map[uint64]int {
	nodes := g.Nodes()
	m := make(map[uint64]int, len(nodes))
	for i, n := range nodes {
		m[n] = i
	}
	return m
}
