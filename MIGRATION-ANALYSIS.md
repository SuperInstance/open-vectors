# Migration Analysis: Spectral Fork → Upstream Weaviate

> **Source:** `SuperInstance/open-vectors` (2 commits ahead)  
> **Target:** `weaviate/weaviate` (122 commits ahead of source)  
> **Analysis Date:** 2026-06-04  
> **Merge Risk:** **Low** — additive changes, no upstream collisions in vector index core

---

## 1. Commit Delta Summary

### Our Commits (2 ahead)

| Commit | Message | Files Added | Lines |
|--------|---------|-------------|-------|
| `41ba3ca` | `feat: add vectorintelligence package for spectral cluster analysis` | `vectorintelligence/*` (8 files) | ~1,500 |
| `03290ba` | `feat: add spectral vector analysis package` | `spectral/*` (4 files) + `vectorintelligence/INTEGRATION.md` | ~2,100 |

**Total:** 14 new files, ~3,600 lines of Go + Markdown, **zero deletions or modifications** to existing upstream files.

### Upstream Commits (122 ahead)

**Breakdown by category:**

| Category | Count | Risk to Our Merge |
|----------|-------|-------------------|
| Stable-branch merge commits (v1.35→v1.36→v1.37→v1.38) | ~45 | **None** — branching noise |
| Test infrastructure / deflaking / shared containers | ~35 | **None** — `test/` only |
| Telemetry (client tracking, integration usage) | ~18 | **Low** — additive, different files |
| RBAC / dynamic users / authz | ~12 | **None** — `usecases/auth/` only |
| Backup & replication tests | ~8 | **None** — `test/` + `usecases/backup/` |
| Runtime config parsing fixes | ~4 | **Low** — `usecases/config/runtime/` |
| Release prep (version bumps) | ~3 | **None** — mechanical |
| HNSW flaky test fix | ~1 | **None** — test-only (`adapters/repos/db/vector/hnsw/search_test.go`) |

**Key upstream files changed that are *near* our integration surface:**

```
adapters/handlers/rest/configure_api.go        [module registration wiring]
adapters/handlers/rest/handlers_api.go         [route mounting — we will touch this]
usecases/config/runtime/values.go              [runtime config — we will touch this]
usecases/telemetry/payload.go                  [telemetry payload — we will touch this]
```

**Critical observation:** Upstream has **not** modified any of the following:

- `adapters/repos/db/vector/hnsw/index.go` (core HNSW struct)
- `adapters/repos/db/vector/hnsw/insert.go` (insert path)
- `adapters/repos/db/vector/hnsw/vertex.go` (vertex structure)
- `adapters/repos/db/vector/common/vector_id.go` (index interfaces)
- `entities/modulecapabilities/` (module contract)
- `go.mod` (except sroar patch bump: `0.0.14-0.20260511125437-5d49ac3cb4c0` → `0.0.14`)

This means our integration points (Pillars 1–4 in `INTEGRATION.md`) face **near-zero merge conflict risk** for the core wiring.

---

## 2. Merge Strategy Recommendation

### 2.1 Recommended Strategy: **Rebase + Feature-Branch Integration**

Do **not** merge upstream into our main branch blindly. Instead, use a structured rebase-integration workflow:

```bash
# 1. Create integration branch from our HEAD
git checkout -b integrate-spectral-to-upstream main

# 2. Rebase our 2 commits onto upstream main
git rebase upstream/main

# 3. Expected result: clean rebase (no conflicts — our commits are pure additions)
# If conflicts appear, they will be in go.sum due to sroar bump; accept upstream's.

# 4. Fast-forward our main only after CI passes
git checkout main
git merge --ff-only integrate-spectral-to-upstream
```

**Why rebase over merge?**

- Our commits are **leaf additions** (no modifications to existing files). Rebasing preserves a linear history that makes bisection easier.
- Upstream's 122 commits are mostly merge-commits from stable branches; a merge commit would create unnecessary diamond complexity.
- If we later need to upstream our spectral packages, a clean rebased branch is trivial to turn into a PR.

### 2.2 Alternative: **Merge Commit (Conservative)**

If the team prefers preserving the exact upstream commit SHAs (e.g., for audit trails):

```bash
git checkout main
git merge upstream/main -m "Merge upstream weaviate/weaviate main (122 commits)"
# Resolves go.sum conflict manually, then commit.
```

**Trade-off:** History is messier, but reference integrity is preserved. Given that our changes are entirely in new directories, this is safe but unnecessary.

---

## 3. Conflict Forecast

### 3.1 Definite Conflicts

| File | Conflict | Resolution |
|------|----------|------------|
| `go.mod` | sroar version bump | Accept upstream (`v0.0.14`); our code does not import sroar. |
| `go.sum` | Checksum drift from sroar + telemetry deps | Regenerate: `go mod tidy`. |

### 3.2 Potential Conflicts (Low Probability)

| File | Scenario | Resolution |
|------|----------|------------|
| `adapters/handlers/rest/configure_api.go` | Upstream adds new module registration calls near our insertion point | Trivial: both additions coexist. |
| `usecases/config/runtime/values.go` | Upstream adds new runtime toggles | Trivial: append our `SpectralInvariantConfig` struct. |
| `openapi-specs/schema.json` | Upstream schema changes | Merge JSON manually; our `/construct` paths are new. |

### 3.3 No Conflicts Expected

All files in `vectorintelligence/` and `spectral/` are **new directories** with no upstream counterparts. The HNSW core (`adapters/repos/db/vector/hnsw/*.go`) is untouched by upstream in the 122-commit delta.

---

## 4. Module-by-Module Impact Assessment

### 4.1 `vectorintelligence/` — **No Merge Risk**

- **Dependency:** `gonum.org/v1/gonum v0.17.0` (already in upstream `go.mod`)
- **Internal imports:** None
- **Upstream changes in this area:** None
- **Action:** None. Directory copies cleanly.

### 4.2 `spectral/` — **No Merge Risk**

- **Dependency:** Same as above
- **Internal imports:** None
- **Upstream changes in this area:** None
- **Action:** None. Directory copies cleanly.

### 4.3 `adapters/repos/db/vector/hnsw/` — **Low Risk**

Upstream changed **zero** production `.go` files in HNSW. The only HNSW-related upstream commit is:

- `3a1c786 Fix flaky HNSW search test` — touches `search_test.go` only.

Our planned integration files (`spectral_graph.go`, `spectral_cache.go`, `conservation_ledger.go`) are new. Existing files (`index.go`, `insert.go`) would receive **minimal, gated** additions (see `INTEGRATION.md` §2.2, §4.2).

**Action after rebase:**
1. Rebase applies cleanly (no HNSW file changes conflict).
2. Implement `spectral_graph.go` and `spectral_cache.go` as follow-up commits.

### 4.4 `adapters/handlers/rest/` — **Medium Risk**

Upstream changed:
- `configure_api.go` (module wiring, db_users, authz)
- `handlers_api.go` (route mounting)
- `handlers_debug.go`
- `middlewares.go`

Our Construct API needs to mount routes and register modules. The upstream changes are **orthogonal** (authz, db users, telemetry), but the files are high-churn.

**Action:** After rebase, apply handler modifications as a **dedicated commit** so it can be rebased again if upstream changes further.

### 4.5 `usecases/config/` — **Low Risk**

Upstream added runtime list-parsing fixes (`13e86e0`, `60abcea`, `4b4763c`) in `usecases/config/runtime/values.go`. Our additions (spectral invariant toggles) are **append-only struct fields** in a different logical area.

**Action:** Trivial merge if concurrent edits occur.

### 4.6 `usecases/telemetry/` — **Low Risk**

Upstream added extensive telemetry client/integration tracking (`73322fb`, `d98f375`, etc.). Our planned addition (`SpectralTelemetry` struct in `payload.go`) is **purely additive** to the JSON payload.

**Action:** Append fields; no logic overlap.

### 4.7 `modules/` — **No Risk (New Directory)**

Our `modules/construct/` and `modules/terny-vectorizer/` are entirely new. Upstream did not add modules in the 122-commit delta.

---

## 5. Step-by-Step Migration Plan

### Phase 0: Pre-Merge Validation (Now)

```bash
# Verify our tests pass on current HEAD
cd /path/to/weaviate
go test ./vectorintelligence/... ./spectral/... -race -count=1

# Verify upstream HEAD builds
git stash
git checkout upstream/main
go build ./...
git checkout -
git stash pop
```

### Phase 1: Rebase (Day 1)

```bash
git fetch upstream
git checkout -b feature/spectral-integration main
git rebase upstream/main
# Resolve go.mod / go.sum if necessary
go mod tidy
go test ./vectorintelligence/... ./spectral/...
git push origin feature/spectral-integration
```

### Phase 2: HNSW Bridge (Day 1–2)

Implement and commit:
1. `adapters/repos/db/vector/hnsw/spectral_graph.go`
2. `adapters/repos/db/vector/hnsw/spectral_cache.go`
3. `adapters/repos/db/vector/common/index_stats.go` (extend)

Run:
```bash
go test ./adapters/repos/db/vector/hnsw/... -run Spectral -v
```

### Phase 3: Construct API Skeleton (Day 2–3)

Implement and commit:
1. `modules/construct/` (module definition, `_spectral` additional property)
2. `adapters/handlers/rest/construct_api.go`
3. `adapters/handlers/rest/handlers_api.go` (route mount)
4. `adapters/handlers/rest/configure_api.go` (module registration)

### Phase 4: Ternary Inference & Conservation (Day 3–5)

Implement and commit:
1. `usecases/ternary/scheduler.go`
2. `usecases/traversal/spectral_query.go`
3. `adapters/repos/db/vector/hnsw/conservation_ledger.go`
4. `adapters/repos/db/vector/hnsw/insert.go` (invariant gate, behind config flag)
5. `usecases/telemetry/payload.go` (spectral fields)

### Phase 5: gRPC & OpenAPI (Day 5–6)

Implement and commit:
1. `grpc/proto/v1/construct.proto`
2. `adapters/handlers/grpc/v1/construct.go`
3. `openapi-specs/schema.json` update + regenerate

### Phase 6: Acceptance Tests & Merge (Day 6–7)

```bash
# Acceptance tests
go test ./test/acceptance/construct_api_test.go -v

# Final rebase cleanup
git rebase -i upstream/main  # squash fixups
git checkout main
git merge --ff-only feature/spectral-integration
```

---

## 6. Rollback Strategy

If any phase introduces instability, rollback is trivial because our changes are **additive and modular**:

```bash
# Revert a single commit (e.g., HNSW bridge)
git revert <commit-sha>

# Or disable at runtime via config
# WEAVIATE_SPECTRAL_ENABLED=false
```

The `vectorintelligence/` and `spectral/` packages can remain in-tree even if the integration layers are reverted—they are inert leaf packages with no import side effects.

---

## 7. Upstream Contribution Feasibility

Because our packages have **zero internal Weaviate dependencies**, they are excellent candidates for upstream contribution as a **standalone analytics module**:

1. **Phase A:** PR `vectorintelligence/` + `spectral/` as pure library additions (low review friction).
2. **Phase B:** PR `modules/construct/` as a standard Weaviate module (follows existing module patterns).
3. **Phase C:** PR HNSW bridge (`spectral_graph.go`) as an optional index introspection feature.

Upstream maintainers have shown appetite for vector analytics (existing `hfresh` index, `compressionhelpers` telemetry). The spectral packages align with this direction.

---

## 8. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| go.sum merge conflict | High | Low | `go mod tidy` |
| Handler file churn conflict | Medium | Low | Rebase frequently; isolate handler changes in dedicated commit |
| Performance regression from spectral cache | Low | Medium | Make cycle-manager interval configurable; default = 0 (disabled) |
| gonum version drift | Low | Low | Pin `gonum v0.17.0` in go.mod; upstream already uses it |
| HNSW lock contention in spectral graph | Low | High | Read-only view; no locks held during spectral analysis |
| CI timeout from new tests | Medium | Low | Tag spectral tests with `-tags spectral`; omit from fast group |

---

## 9. Conclusion

**Merge recommendation: Proceed with rebase.**

The 122-commit upstream delta is dominated by test infrastructure, telemetry, and stable-branch merges. There is **zero overlap** with our spectral vector analysis code. The integration architecture described in `INTEGRATION.md` touches high-churn files (`configure_api.go`, `handlers_api.go`) only in later phases, and those touches are small, additive, and easily rebased.

The spectral packages (`vectorintelligence/`, `spectral/`) are **immediately mergeable** without conflict. The integration layers should be developed as **follow-up commits** on the rebased branch to minimize rebase friction.

---

*Analysis based on commit range `e3f8739..8269596` (upstream) vs. `41ba3ca..03290ba` (local).*  
*136 files changed upstream; 14 files added locally; intersection = 0.*
