# Ternary Agent Ecosystem — Weaviate Integration Architecture

> **Status:** Design Document  
> **Scope:** `vectorintelligence/` + `spectral/` packages → full Weaviate integration  
> **Upstream Delta:** 122 commits ahead (weaviate/weaviate@main)  
> **Local Delta:** 2 commits ahead (spectral vector analysis)

---

## 1. Executive Summary

This document describes how the **Ternary Agent Ecosystem** integrates with Weaviate through four architectural pillars. Our fork adds two standalone Go packages—`vectorintelligence/` and `spectral/`—that provide spectral graph analysis of embedding spaces. These packages are currently **leaf nodes** in the dependency graph (zero internal Weaviate imports). This document specifies the **wiring layers** required to turn them into first-class subsystems of a production Weaviate deployment.

The four pillars are:

| Pillar | Weaviate Subsystem | Files / Modules Touched |
|--------|-------------------|------------------------|
| 1. Agent Memory | Vector DB storage + HNSW index | `adapters/repos/db/vector/hnsw/*`, `entities/storobj/*`, `usecases/objects/*` |
| 2. Ternary Inference | Spectral inference over vector spaces | `vectorintelligence/*`, `spectral/*`, `adapters/handlers/rest/*`, `adapters/handlers/graphql/*` |
| 3. Conservation Laws | Vector space invariants + telemetry | `usecases/telemetry/*`, `adapters/repos/db/vector/common/*`, `entities/additional/*` |
| 4. Construct API | Vector operation surface (REST/gRPC/GraphQL) | `adapters/handlers/rest/*`, `adapters/handlers/grpc/*`, `openapi-specs/*`, `modules/construct/*` |

---

## 2. Pillar 1: Agent Memory Backed by Vector DB

### 2.1 Concept

Agent memory is not a separate database—it is a **schema convention** within Weaviate. Each agent namespace is modeled as a Weaviate class with:

- A primary vector field (`memory_vector`) for episodic retrieval.
- A named vector (`context_vector`) for working-memory state.
- Metadata properties: `agent_id`, `session_id`, `turn_id`, `timestamp`, `memory_type` (episodic / semantic / procedural).

### 2.2 Storage Layer Integration

Weaviate already stores vectors in LSM-backed HNSW indices (`adapters/repos/db/vector/hnsw/`). Agent memory requires **no new storage engine**. Instead, we add:

1. **HNSW → Spectral Graph Bridge**
   - Implement `spectral.Graph` on a view of the HNSW index.
   - File: `adapters/repos/db/vector/hnsw/spectral_graph.go` (new)
   - Exposes `Nodes()`, `Neighbors(node, level)`, and `MaxLevel()` by reading the HNSW vertex layer structure without mutating the index.

   ```go
   type HNSWSpectralGraph struct {
       index *hnsw
       nodes []uint64
   }
   
   func (g *HNSWSpectralGraph) Neighbors(node uint64, level int) []uint64 {
       v := g.index.nodes[node]
       return v.connectionsAtLevelNoLock(level)
   }
   ```

2. **Shard-Level Spectral Cache**
   - File: `adapters/repos/db/vector/hnsw/spectral_cache.go` (new)
   - Periodic background task (via `entities/cyclemanager`) that subsamples the HNSW graph, runs `spectral.AnalyzeEmbeddingQuality`, and caches the `EmbeddingQualityReport` per shard.
   - This gives agents a **topological awareness** of their own memory space (e.g., "my episodic memory has 3 bottleneck clusters—consolidation recommended").

3. **Vector Lifecycle Hooks**
   - File: `adapters/repos/db/vector/common/index_stats.go` (extend)
   - Add `SpectralStats` to `IndexStats`:
     ```go
     type IndexStats struct {
         // ... existing fields ...
         SpectralStats *SpectralIndexStats `json:"spectral,omitempty"`
     }
     
     type SpectralIndexStats struct {
         AlgebraicConnectivity float64   `json:"algebraic_connectivity"`
         CheegerConstant       float64   `json:"cheeger_constant"`
         NumCommunities        int       `json:"num_communities"`
         LastAnalyzed          time.Time `json:"last_analyzed"`
     }
     ```

### 2.3 Files Changed

| File | Change |
|------|--------|
| `adapters/repos/db/vector/hnsw/spectral_graph.go` | **New** — HNSW → `spectral.Graph` adapter |
| `adapters/repos/db/vector/hnsw/spectral_cache.go` | **New** — Background spectral analysis cache |
| `adapters/repos/db/vector/hnsw/index.go` | **Modify** — Register cycle manager for spectral cache refresh |
| `adapters/repos/db/vector/common/index_stats.go` | **Modify** — Add `SpectralIndexStats` struct |
| `entities/storobj/storage_object.go` | **No change** — Schema-level convention only |

---

## 3. Pillar 2: Ternary Inference Over Vector Spaces

### 3.1 Concept

Ternary inference treats every embedding as a **superposition of three basis states**: 
- `+1` (aligned / promote)
- `0`  (neutral / uncertain)
- `-1` (opposed / suppress)

This is not a quantization scheme—it is an **inference semantics** layered on top of continuous vectors. The spectral packages provide the mathematical machinery:

- **Fiedler vector** → determines the primary inference axis (bipartition of the memory graph).
- **Cheeger constant** → confidence metric: low Cheeger = high confidence the bipartition is real.
- **Recursive spectral bisection** → hierarchical ternary decomposition: each split creates a new `{+, 0, -}` plane.
- **JSD matrix** → cross-cluster divergence used to resolve contradictions between agent beliefs.

### 3.2 Integration Layer

1. **Ternary Vectorizer Module**
   - File: `modules/terny-vectorizer/` (new module directory)
   - Implements `modulecapabilities.Vectorizer` and `modulecapabilities.AdditionalProperties`.
   - On vectorization, encodes text into a standard embedding, then **projects** it onto the Fiedler basis of the agent's current memory graph.
   - Adds an `additional` property `_ternaryState` to Get/List queries:
     ```json
     {
       "_ternaryState": {
         "axis": "fiedler_0",
         "value": 0.73,
         "confidence": 0.91,
         "cluster": 2
       }
     }
     ```

2. **Spectral Query Engine**
   - File: `usecases/traversal/spectral_query.go` (new)
   - New traversal primitive: `NearSpectral`.
   - Takes a query vector, builds a temporary k-NN graph from the top-`k` HNSW results, runs `vectorintelligence.RunAnalysis`, and returns objects annotated with cluster assignments and JSD divergence from the query.

3. **Inference Scheduler**
   - File: `usecases/ternary/scheduler.go` (new)
   - Orchestrates ternary inference as a **vector-native computation graph**:
     - Load agent memory vectors → `vectorintelligence.RunAnalysis`
     - Identify contradictory clusters (high JSD + overlapping semantic space)
     - Emit `ConsolidationTask` or `ConflictResolutionTask`
   - Uses Weaviate's existing `batch` infrastructure for vector I/O.

### 3.3 Files Changed

| File | Change |
|------|--------|
| `modules/terny-vectorizer/` | **New** — Ternary vectorizer + `_ternaryState` additional property |
| `usecases/traversal/spectral_query.go` | **New** — `NearSpectral` query usecase |
| `usecases/ternary/scheduler.go` | **New** — Ternary inference orchestrator |
| `adapters/handlers/graphql/local/get/` | **Modify** — Wire `_ternaryState` into GraphQL schema |
| `adapters/handlers/rest/traversals.go` | **Modify** — Wire `NearSpectral` into REST handlers |
| `entities/search/` | **Modify** — Add `TernaryState` to `Result` struct |

---

## 4. Pillar 3: Conservation Laws as Vector Space Invariants

### 4.1 Concept

In the ternary ecosystem, **conservation laws** are spectral invariants enforced at the vector-index level:

| Law | Invariant | Enforcement Mechanism |
|-----|-----------|----------------------|
| **Memory Conservation** | Total "cognitive load" (sum of effective resistances) is monotonic under merge | Spectral merge gate in `vectorintelligence/` |
| **Contradiction Preservation** | JSD between opposing clusters cannot drop below threshold without audit trail | `JSDMatrix` baseline + delta logging |
| **Identity Conservation** | Fiedler value of an agent's core memory graph ≥ λ_min | Shard health check rejects destructive updates |

### 4.2 Integration Layer

1. **Invariant Guard in HNSW Insert Path**
   - File: `adapters/repos/db/vector/hnsw/insert.go` (modify)
   - After batch insert, if `SpectralIndexStats` is enabled, recompute algebraic connectivity. If `λ₂ < λ_min`, flag the shard for review rather than failing (graceful degradation).

2. **Telemetry Invariant Stream**
   - File: `usecases/telemetry/payload.go` (modify)
   - Extend telemetry payload with spectral invariants:
     ```go
     type SpectralTelemetry struct {
         AlgebraicConnectivity float64 `json:"lambda_2"`
         CheegerConstant       float64 `json:"cheeger"`
         NumCommunities        int     `json:"communities"`
         JSDMean               float64 `json:"jsd_mean"`
     }
     ```
   - Upstream already has telemetry client/integration tracking (122 commits include `usecases/telemetry/client_tracker.go`, `client_middleware.go`). Our additions are **additive fields** in the payload struct—zero conflict with upstream telemetry architecture.

3. **Conservation Ledger**
   - File: `adapters/repos/db/vector/hnsw/conservation_ledger.go` (new)
   - LSM-backed append-only log of every spectral-invariant mutation.
   - Each entry: `{timestamp, shard_id, operation, pre_lambda2, post_lambda2, delta_jsd}`.
   - Agents replay this ledger to reconstruct their own cognitive evolution.

### 4.3 Files Changed

| File | Change |
|------|--------|
| `adapters/repos/db/vector/hnsw/insert.go` | **Modify** — Add post-insert spectral invariant check (gated by config) |
| `adapters/repos/db/vector/hnsw/conservation_ledger.go` | **New** — Append-only spectral mutation log |
| `usecases/telemetry/payload.go` | **Modify** — Add `SpectralTelemetry` struct |
| `usecases/config/runtime/values.go` | **Modify** — Add `SpectralInvariantConfig` runtime toggle |
| `entities/additional/` | **Modify** — Add `_conservationStatus` additional property |

---

## 5. Pillar 4: The Construct API Surface for Vector Operations

### 5.1 Concept

The **Construct API** exposes spectral vector operations as first-class API primitives, not just internal analytics. It is a new module (`modules/construct`) plus handler extensions that let clients:

1. `POST /v1/construct/analyze` — Run `spectral.AnalyzeEmbeddingQuality` on a class or shard.
2. `POST /v1/construct/fiedler` — Get the Fiedler vector for a set of IDs.
3. `POST /v1/construct/communities` — Detect communities and return cluster assignments.
4. `POST /v1/construct/ternary` — Project vectors onto a ternary basis.
5. GraphQL: `Get { ClassName { _spectral { clusters, cheeger, jsdMatrix } } }`

### 5.2 Integration Layer

1. **Construct Module**
   - Directory: `modules/construct/` (new)
   - Implements:
     - `modulecapabilities.Module` (name: `"construct"`)
     - `modulecapabilities.AdditionalProperties` (`_spectral`, `_ternaryState`)
     - `modulecapabilities.GraphQLAdditional` (field definitions for `_spectral`)
   - Has **no external API calls**—purely computational, operating on vectors already in Weaviate.

2. **REST Handler Extensions**
   - File: `adapters/handlers/rest/construct_api.go` (new)
   - Mounted under `/v1/construct/*` via `adapters/handlers/rest/handlers_api.go` (modify).
   - Reuses existing `db` and `schemaManager` dependencies from `restAPI` struct.

3. **gRPC Extension**
   - File: `adapters/handlers/grpc/v1/construct.go` (new)
   - Adds `Construct` service to the existing gRPC server.
   - Proto additions go in `grpc/proto/v1/` (new `.proto` file or extension of `weaviate.proto`).

4. **OpenAPI Spec Update**
   - File: `openapi-specs/schema.json` (modify)
   - Add `/construct` paths and `ConstructAnalyzeRequest`, `ConstructAnalyzeResponse` schemas.
   - Regenerate via `tools/gen-code-from-swagger.sh`.

### 5.3 Files Changed

| File | Change |
|------|--------|
| `modules/construct/` | **New** — Construct module with spectral additional properties |
| `adapters/handlers/rest/construct_api.go` | **New** — REST handlers for Construct endpoints |
| `adapters/handlers/rest/handlers_api.go` | **Modify** — Mount Construct routes |
| `adapters/handlers/rest/configure_api.go` | **Modify** — Register `modules.Construct` with module provider |
| `adapters/handlers/grpc/v1/construct.go` | **New** — gRPC service implementation |
| `grpc/proto/v1/construct.proto` | **New** — Proto definitions for Construct API |
| `openapi-specs/schema.json` | **Modify** — OpenAPI paths + schemas |
| `entities/modulecapabilities/` | **No change** — Reuse existing interfaces |

---

## 6. Dependency & Build Impact

### 6.1 Go Modules

Our packages depend only on `gonum.org/v1/gonum v0.17.0`, which is **already a transitive dependency** of Weaviate. No `go.mod` changes are required for the core spectral packages.

Integration files (HNSW bridge, Construct module) import internal Weaviate packages and therefore live **inside** the main module (`github.com/weaviate/weaviate`). No new `go.mod` files are needed.

### 6.2 Build Tags / Conditional Compilation

All spectral integration features are **gated by build tags** or runtime config:

```go
// +build spectral

package hnsw
```

This ensures upstream CI is unaffected if the feature is not compiled in.

---

## 7. Testing Strategy

| Layer | Test Location | Approach |
|-------|--------------|----------|
| Core spectral | `spectral/*_test.go`, `vectorintelligence/*_test.go` | Unit tests (already exist, 22 + 13 tests) |
| HNSW bridge | `adapters/repos/db/vector/hnsw/spectral_graph_test.go` | Property-based: random HNSW graph → spectral.Graph → invariant checks |
| Construct API | `test/acceptance/construct_api_test.go` | End-to-end: create class → insert vectors → call `/v1/construct/analyze` |
| Ternary inference | `test/acceptance/ternary_inference_test.go` | Agent simulation: multi-turn session → verify JSD divergence trends |
| Conservation ledger | `adapters/repos/db/vector/hnsw/conservation_ledger_test.go` | Mock LSM bucket → append → replay → verify monotonicity |

---

## 8. Security & Access Control

- **Construct API endpoints** respect existing RBAC (`usecases/auth/authorization`). New actions: `read_spectral`, `manage_conservation`.
- **Spectral cache** does not expose raw vectors—only aggregated statistics (connectivity, modularity, cluster sizes).
- **Conservation ledger** is stored in the same LSM store as the index; encryption-at-rest is inherited from Weaviate's storage layer.

---

## 9. Summary of Touch Points

```
New directories:
  modules/construct/
  usecases/ternary/
  usecases/traversal/
  adapters/repos/db/vector/hnsw/spectral_graph.go
  adapters/repos/db/vector/hnsw/spectral_cache.go
  adapters/repos/db/vector/hnsw/conservation_ledger.go
  adapters/handlers/rest/construct_api.go
  adapters/handlers/grpc/v1/construct.go
  grpc/proto/v1/construct.proto

Modified files (low conflict risk):
  adapters/repos/db/vector/hnsw/index.go          [cycle manager registration]
  adapters/repos/db/vector/hnsw/insert.go         [optional invariant gate]
  adapters/repos/db/vector/common/index_stats.go  [new struct field]
  adapters/handlers/rest/handlers_api.go          [route mounting]
  adapters/handlers/rest/configure_api.go         [module registration]
  usecases/telemetry/payload.go                  [additive JSON fields]
  usecases/config/runtime/values.go              [new config keys]
  openapi-specs/schema.json                      [new paths]
```

---

*Document generated for commit range `41ba3ca..03290ba` against upstream `HEAD`.*
