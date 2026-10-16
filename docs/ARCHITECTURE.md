# Lattice — Project Architecture & Flowcharts

A full visual map of the system: what each module does, what it depends on, how a write flows in, how a query flows out, and how the phased build plan gets there. This document describes the full intended design; the scope note below says which parts are built.

> **Scope note (current).** The diagrams below show the full phased plan. The shipped project covers the 1M-vector scope: `common`, `kernels`, `quant`, `index_flat`, `index_graph`, `filter`, `storage`, `segment`. The 10M/100M scale targets and the Phase 4/5 modules (`index_ivf`, `lexical`, `fusion`, `query`, `api`, `server`, `pybind`) are out of scope for this release. **SQ8 (8-bit) is the default codes format for graph traversal**; RaBitQ remains in `quant` as a tested primitive but is not used in traversal, because on real data at 1M vectors its 1-bit estimator was far less discriminative (see `docs/BENCHMARKS.md`). Measured results: see the root `README.md` and `docs/BENCHMARKS.md`.

---

## 1. System overview

```mermaid
flowchart TD
    subgraph EXT["Outside the core (D-ARCH-02)"]
        ING["Ingestion sidecar (Python)<br/>parse → chunk → embed → upsert"]
        CDC["Optional CDC pipeline<br/>Postgres → Debezium → Kafka (D-ARCH-11)"]
    end

    subgraph SURFACE["api / server / pybind"]
        API["api: Database, Collection<br/>upsert / delete / search / explain"]
        SRV["server: gRPC service"]
        PYB["pybind: nanobind Python module"]
    end

    subgraph ENGINE["Query & write engine"]
        QRY["query: Planner + Executor<br/>8-operator tree (D-QRY-06)"]
        FUS["fusion: RRF / convex / MaxSim<br/>ScoreBreakdown (D-RET-06)"]
        SEG["segment: GrowingSegment + SealedSegment<br/>PK index, snapshots, merge"]
    end

    subgraph INDEXES["Index layer"]
        IDXG["index_graph: Vamana<br/>in-RAM + disk-resident backends"]
        IDXI["index_ivf: coarse k-means + lists (Phase 5)"]
        IDXF["index_flat: SIMD brute force"]
        LEX["lexical: BM25 + sparse vectors"]
        FLT["filter: typed columns, bitmaps, selectivity"]
    end

    subgraph FOUNDATION["Foundation"]
        QNT["quant: SQ8, RaBitQ codes"]
        KRN["kernels: SIMD distance, ISA dispatch"]
        STO["storage: WAL, manifest, mmap, io_uring"]
        CMN["common: IDs, Error, RowFilter, schema"]
    end

    ING --> API
    CDC --> ING
    API --> SRV
    API --> PYB
    API --> QRY
    QRY --> SEG
    QRY --> FUS
    SEG --> IDXG
    SEG --> IDXI
    SEG --> IDXF
    SEG --> LEX
    SEG --> FLT
    SEG --> STO
    IDXG --> QNT
    IDXI --> QNT
    IDXG --> KRN
    IDXI --> KRN
    IDXF --> KRN
    QNT --> KRN
    FUS --> KRN
    FLT --> CMN
    STO --> CMN
    KRN --> CMN
    SEG --> CMN
```

**Reading it:** everything outside the dashed boundary of `api` is a separate process (Python sidecar, CDC pipeline) — the C++ core never embeds a model or parses documents (D-ARCH-02). Inside the core, `query` and the write path both terminate at `segment`, which is the only module allowed to touch every index and storage primitive at once.

---

## 2. Module dependency graph (one-way only)

```mermaid
flowchart BT
    common["common<br/>(IDs, Error, RowFilter, ThreadPool, TopKHeap, CollectionSchema)"]
    kernels["kernels<br/>(SIMD distance, ISA dispatch)"]
    quant["quant<br/>(SQ8, RaBitQ)"]
    storage["storage<br/>(WAL, manifest, mmap, io_uring)"]
    filter["filter<br/>(columns, bitmaps, selectivity)"]
    lexical["lexical<br/>(BM25, sparse vectors)"]
    index_flat["index_flat<br/>(brute force)"]
    index_graph["index_graph<br/>(Vamana, 2 backends)"]
    index_ivf["index_ivf<br/>(k-means + lists)"]
    fusion["fusion<br/>(RRF, convex, MaxSim)"]
    segment["segment<br/>(growing/sealed, PK index, merge)"]
    query["query<br/>(planner, executor, operator tree)"]
    api["api<br/>(Database, Collection)"]
    server["server (gRPC)"]
    pybind["pybind (nanobind)"]

    kernels --> common
    quant --> common
    quant --> kernels
    storage --> common
    filter --> common
    lexical --> common
    index_flat --> common
    index_flat --> kernels
    index_graph --> common
    index_graph --> kernels
    index_graph --> quant
    index_ivf --> common
    index_ivf --> kernels
    index_ivf --> quant
    fusion --> common
    fusion --> kernels
    segment --> common
    segment --> storage
    segment --> filter
    segment --> quant
    segment --> index_graph
    segment --> index_ivf
    segment --> index_flat
    segment --> lexical
    query --> common
    query --> segment
    query --> fusion
    query --> filter
    api --> common
    api --> query
    api --> segment
    api --> storage
    server --> api
    pybind --> api
```

**The one rule that matters (D-MOD-01):** `index_graph` / `index_ivf` / `index_flat` never depend on `filter` or `storage` directly. They receive an opaque `common::RowFilter` (a `bool(RowId)` predicate or bitmap view) and, for the disk backend, an abstract `NodeStore` interface that `segment` implements on top of `storage`'s io_uring file. This is what keeps the index code testable with synthetic predicates and swappable storage backends without ever touching the schema or the filesystem.

External libraries plug in at exactly two points: **RocksDB** inside `segment` only (row-durability buffer, D-ARCH-04), and **FAISS / hnswlib** inside `bench/` only (comparison baselines, never linked into `liblattice`).

---

## 3. Write path — upsert to durable, queryable row

```mermaid
sequenceDiagram
    participant C as Client / sidecar
    participant API as api::Collection
    participant GS as segment::GrowingSegment
    participant RDB as RocksDB (row buffer)
    participant MIR as In-RAM vector mirror
    participant SS as segment::SealedSegment
    participant MAN as storage::Manifest

    C->>API: upsert(id, fields, vectors, source_offset)
    API->>GS: upsert(...)
    GS->>RDB: put(row_id → columns+vectors), LWW by source_offset (D-STO-08)
    RDB-->>GS: RocksDB sequence number = Lsn
    GS->>MIR: append vector, then publish visible_count (release)
    GS-->>API: Lsn
    API-->>C: ack (Lsn = commit token, D-IDX-05 session-token seam)

    Note over GS: brute-force search only here (D-IDX-06) — no graph in the growing segment (D-CON-02, finding 13)

    rect rgb(235,235,245)
    Note over GS,MAN: Seal, triggered by size/time threshold
    GS->>SS: seal_from(growing, schema)
    SS->>SS: iterate RocksDB CF in sorted key order
    SS->>SS: build filter columns + bitmaps (via `filter`)
    SS->>SS: train RaBitQ/SQ8 codes (via `quant`)
    SS->>SS: build Vamana index (via `index_graph`) or flat if < threshold
    SS->>SS: build BM25 + sparse postings (via `lexical`)
    SS->>MAN: write SealedSegmentEntry{segment_id, source_cf, max_seq}
    MAN-->>SS: durable (fsync + atomic rename)
    SS->>RDB: drop column family (idempotent — safe to retry from manifest entry)
    end
```

**Why this shape:** RocksDB is the row-durability layer (D-ARCH-04/R3) — it survives a crash on its own, so Lattice's own WAL only has to log segment-lifecycle events (seal/merge/manifest-swap), not every row write. The two-phase handoff (manifest write *before* CF drop) is what makes a crash mid-seal safe to retry (finding 11).

---

## 4. Query path — search to explained result

```mermaid
flowchart TD
    Q["QueryRequest<br/>(query vectors, predicate, lexical_query, k, group_by, explain)"]
    Q --> FE["FilterEval (per segment)<br/>filter::PredicateEvaluator → EXACT selectivity count"]
    FE --> PLAN{"Planner (D-QRY-01)<br/>picks strategy from exact selectivity"}
    PLAN -->|"selectivity < ~1-2%"| PRE["Pre-filter + brute force<br/>(index_flat over passing rows)"]
    PLAN -->|"moderate selectivity"| INTRAV["In-traversal filter<br/>(index_graph admits only passing VectorIds)"]
    PLAN -->|"high selectivity"| POST["Post-filter<br/>(over-fetch k′≫k, drop failing rows)"]
    PLAN -->|"Phase 5"| ACORN["ACORN 2-hop expansion<br/>(bounded by max_ssd_reads_per_query)"]

    PRE --> DK["DenseKnn per query vector<br/>(m query vectors → m searches, D-RET-02a)"]
    INTRAV --> DK
    POST --> DK
    ACORN --> DK

    Q --> LTK["LexicalTopK<br/>lexical::InvertedIndex.bm25_top_k / SparseFieldIndex"]

    DK --> FU["Fuse<br/>fusion::reciprocal_rank_fusion / convex_fusion"]
    LTK --> FU
    FU --> GB["GroupBy (D-RET-07)<br/>group_id, adaptive over-fetch (D-QRY-05)"]
    GB --> RR["Rerank<br/>fusion::max_sim_score (exact MaxSim) + exact fp32 rescoring"]
    RR --> LIM["Limit → top-k"]
    LIM --> RES["QueryResult<br/>hits + ScoreBreakdown (D-RET-06)"]
    RES -.->|explain=true| PLANOUT["QueryPlan.per_segment_choices<br/>(strategy + exact selectivity, D-ARCH-09)"]
```

**The headline claim this diagram exists to support (D-QRY-01):** every branch out of the planner diamond is chosen from an *exact* roaring-bitmap cardinality, never an estimate — that's what `explain=true` surfaces in `per_segment_choices`.

---

## 5. Segment lifecycle (storage engine)

```mermaid
stateDiagram-v2
    [*] --> Growing: first write (RocksDB CF opened)
    Growing --> Sealing: size/time threshold (D-IDX-06 tunable)
    Sealing --> Sealed: manifest write durable, THEN RocksDB CF dropped
    Sealed --> Merging: tombstone ratio > 20% (D-IDX-03) or small-segment policy
    Merging --> Sealed: new merged segment published (rebuild, Phase 3; graph-merge, Advanced)
    Sealed --> Dropped: merged away AND no Snapshot pins it (refcounted, D-STO-05)
    Dropped --> [*]

    note right of Growing
        Brute-force search only (index_flat)
        over an in-RAM vector mirror.
        No graph exists yet.
    end note

    note right of Sealed
        Read-only. mmap + madvise.
        Vamana/IVF/flat index built.
        Lock-free reads (D-CON-02).
    end note
```

---

## 6. Disk-resident Vamana query (the DiskANN-style beam search)

```mermaid
flowchart LR
    subgraph RAM["Always in RAM"]
        CODES["RaBitQ/SQ8 codes<br/>(64-byte aligned, contiguous array)"]
        HEAP["TopKHeap / visited-set<br/>(fixed-capacity, D-CON-04)"]
    end
    subgraph SSD["On SSD (io_uring / O_DIRECT)"]
        REC["4KB-aligned node records<br/>(vector + degree + neighbour IDs, D-VEC-05)"]
    end

    START(["entry point"]) --> DIST["Compute code-distance<br/>to unvisited neighbours"]
    DIST --> CODES
    DIST --> PICK{"Closer than<br/>current frontier?"}
    PICK -->|yes| FETCH["AsyncNodeFile::submit_reads<br/>(prefetch batch)"]
    FETCH --> REC
    REC --> HEAP
    HEAP --> PICK
    PICK -->|frontier converged| RERANK["Rerank on exact vector<br/>(arrived free with the node record — D-VEC-03)"]
    RERANK --> TOPK(["top-k VectorIds"])
```

One SSD read per visited node (not two) is the entire point of the co-located node-record layout (D-VEC-05) — the exact vector needed for reranking arrives in the same read as the adjacency list.

---

## 7. Technology stack — what's used where

```mermaid
flowchart TD
    subgraph Core["liblattice (hand-built, D-CPP-06)"]
        direction LR
        C1["Vamana graph, IVF, flat index"]
        C2["BM25 + sparse inverted index"]
        C3["SQ8 + RaBitQ quantizers"]
        C4["Segment store, WAL, manifest"]
        C5["Cost-based query planner"]
        C6["SIMD kernels (AVX2/VNNI/AVX-512)"]
    end
    subgraph Vendored["Vendored — used, not rebuilt"]
        direction LR
        V1["RocksDB<br/>(segment's row buffer ONLY)"]
        V2["CRoaring<br/>(bitmaps, filter module)"]
        V3["liburing<br/>(disk-resident Vamana I/O)"]
        V4["mimalloc<br/>(global allocator)"]
        V5["gRPC + Protobuf<br/>(server)"]
        V6["nanobind<br/>(pybind)"]
        V7["toml++, spdlog,<br/>prometheus-cpp, simdjson"]
        V8["libc++ (Clang only)<br/>for std::expected — see D-CPP-01 addendum"]
    end
    subgraph BenchOnly["Benchmark-only — never linked into liblattice"]
        direction LR
        B1["FAISS (PQ, IndexFlatIP baselines)"]
        B2["hnswlib (HNSW baseline)"]
        B3["Python: sentence-transformers,<br/>h5py, beir, hnswlib-py"]
    end

    Core -.uses.-> Vendored
    BenchOnly -.compared against.-> Core
```

---

## 8. The three headline differentiators (D-ADV-01)

```mermaid
flowchart LR
    H1["Headline 1<br/>Multi-vector late interaction"] --> H1D["Naive per-query-vector kNN (Phase 3)<br/>→ MUVERA FDE collapse (Phase 5)<br/>vs. PLAID"]
    H2["Headline 2<br/>Selectivity-aware filtered search"] --> H2D["Pre/post/in-traversal + planner (Phase 3)<br/>→ ACORN 2-hop expansion (Phase 5)<br/>vs. FAISS post-filter, Weaviate ACORN"]
    H3["Headline 3<br/>RaBitQ compression"] --> H3D["Functional RaBitQ + error bound (Phase 3)<br/>→ bound-driven adaptive rerank at 100M (Phase 5)<br/>vs. FAISS PQ"]
```

Each one needs exactly one reproducible claim chart (D-ADV-02).

---

## 9. Phased build plan

```mermaid
flowchart TD
    P0["Phase 0 — Foundations ✅ DONE<br/>WSL2 env, datasets, embedding model,<br/>synthetic corpus, hnswlib recall gate"]
    P1["Phase 1 — Kernels + flat<br/>common, kernels, quant(SQ8), index_flat<br/>ASan/TSan CI from here on"]
    P2["Phase 2 — Vamana (in-RAM)<br/>index_graph in-RAM backend, 100K/1M only<br/>D-CON-04 micro-opts, thread pool"]
    P3["Phase 3 — Storage + disk Vamana + data model<br/>storage, segment, filter, RaBitQ, disk backend<br/>pulled forward from Phase 5 (R2)<br/>10K+ crash-injection gate"]
    P4["Phase 4 — Hybrid + API<br/>lexical, fusion, query, api, server, pybind<br/>BM25+RRF beats dense-only on nDCG@10"]
    P5["Phase 5 — Headlines<br/>100M run, RaBitQ maturation, ACORN, MUVERA, IVF<br/>one claim chart per headline"]
    P6["Phase 6 — Write-up<br/>Pareto curves, ablations, reproducible scripts"]

    P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6

    style P0 fill:#c8e6c9,stroke:#2e7d32
```

Each arrow is a hard gate: a phase exit-gate acceptance tests must pass before the next phase deliverables start.

---

## 10. Data model shape (what a "row" actually is)

```mermaid
flowchart TD
    ROW["Row (D-ARCH-03)"]
    ROW --> ID["id (external, stable)"]
    ROW --> GID["group_id (optional — document/entity)"]
    ROW --> VER["version"]
    ROW --> SCALAR["Typed scalar columns<br/>(int, float, timestamp, enum, tag set)<br/>+ checksum, source_offset (D-ARCH-11)"]
    ROW --> DENSE["Named dense vector field(s)<br/>(768-d, nomic-embed-text-v1.5, D-VEC-07)"]
    ROW --> SPARSE["Optional sparse vector field<br/>(SPLADE-style, D-RET-03)"]
    ROW --> MULTI["Optional multi-vector field<br/>(bag of chunk/token vectors)"]

    MULTI --> V1["VectorId 1 → RowId"]
    MULTI --> V2["VectorId 2 → RowId"]
    MULTI --> VN["VectorId N → RowId"]
    DENSE --> V1

    note1["VectorId ≠ RowId (D-MOD-02) —<br/>one row can own many indexed vectors;<br/>segment owns the bidirectional map"]
```

---

