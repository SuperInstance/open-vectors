# Future Integration: open-vectors (Weaviate Fork)

## Current State
A fork of Weaviate — an open-source, cloud-native vector database supporting semantic search, RAG, and keyword filtering. Stores both objects and vectors. Supports automatic vectorization, multi-tenancy, and replication.

> **Note:** This is a fork of the Weaviate project. We respect their work and add fleet-specific integrations.

## Integration Opportunities

### With ternary embeddings storage
Weaviate stores the fleet's ternary embeddings. Every skill description, every strategy vector, every room profile is embedded using position-aware-embed and stored in Weaviate. When a room needs to find similar rooms, it queries Weaviate for the nearest neighbors in embedding space. When an agent needs a skill, it queries Weaviate for skills with similar capability descriptions.

### With room discovery
Rooms register their capabilities in Weaviate. When Oracle1 needs to route a query to the right room, it embeds the query and finds the closest room in vector space. "Engine vibration analysis" → finds the "sensor monitoring" room. "Music theory" → finds the "harmonic analysis" room. Semantic room routing via vector search.

### With oracle1-index enhancement
The current keyword-based index (674KB search, 247KB keyword) is augmented with Weaviate's semantic search. Same index, richer queries. "Find repos that deal with conservation laws in ternary systems" returns conservation-matrix-rs, conservation-verify, and conservation-spectral-topology-rs — not because they share keywords, but because they share meaning.

## Our Integration (Not Upstream Changes)
We do NOT modify Weaviate's core database. Our integration is:
- Fleet-specific Weaviate schema for ternary entities (rooms, skills, strategies)
- SuperInstance embedder plugin using position-aware-embed
- Query bridge from ternary-protocol to Weaviate's GraphQL API

## Potential in Mature Systems
Weaviate is the fleet's memory — the semantic search layer that makes everything findable. Every room, every skill, every strategy, every tile is stored as a vector. The fleet can search its own knowledge base by meaning, not just by name.

## Cross-Pollination Ideas
- **torch-vector-search**: GPU-accelerated search for Weaviate's vector index
- **position-aware-embed**: Custom embedder for Weaviate
- **oracle1-index**: Index data stored in Weaviate for semantic search

## Dependencies for Next Steps
- Weaviate schema for ternary entities
- position-aware-embed plugin for Weaviate
- Query bridge from ternary-protocol
