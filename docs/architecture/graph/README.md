# Knowledge graph of the repo

A machine-built map of this repository, made by graphify 0.9.93 (PyPI package `graphifyy`). Code (Python, TypeScript, JSON, TOML) is read with the AST; the 21 Markdown, YAML and HTML
docs are read by LLM subagents, which extract concepts, cited papers and design rationale. Each
edge is tagged `EXTRACTED` (stated in the source) or `INFERRED` (a judgement, with a score). Nodes
are grouped into communities by graph clustering, and the community names were written by hand
from each community's members.

It complements the hand-drawn diagrams in the parent folder. Those are the source of truth for the
design. This graph is a navigation aid and can be wrong.

| File | What it is |
|---|---|
| [`GRAPH_REPORT.md`](GRAPH_REPORT.md) | Summary, god nodes, surprising connections, hyperedges, one section per community, knowledge gaps, suggested questions |
| [`graph.json`](graph.json) | The graph itself (NetworkX node-link JSON, with `community_name` on each node). This is what `graphify query` reads |

The static picture (`graph.svg`, about 5.1 MB, roughly 44k glyphs) is not committed: at 1,222 nodes it is
unreadable as an image and would add megabytes to every milestone's history. It is regenerated into
`graphify-out/graph.svg` (gitignored) by the command below. There is no `graph.html` either. The interactive page graphify writes loads `vis-network` from
`unpkg.com` when opened, so it is not self-contained and needs network access.

## Numbers (as printed by graphify)

- Generated: 2026-10-28 (full build, then refreshed with `--update` the same day), graphify 0.9.93
- Corpus: 102 files, about 103,817 words (81 code, 21 docs, 0 papers, 0 images)
- Graph: **1222 nodes, 2526 edges, 70 communities** (the report shows 59 and omits 11 thin ones)
- Extraction: 90% EXTRACTED, 10% INFERRED (241 edges, average confidence 0.89), 0% AMBIGUOUS
- The refresh re-extracted the 38 files that had changed or been added (21 code, 17 docs) and
  merged them into the first graph: 708 nodes and 1697 edges came from the new extraction
  (320 and 388 nodes from the AST and the docs), replacing 643 old nodes
- Token cost: 674,120 for the refresh's doc extraction (652,876 for the first build). The harness
  reports one total per subagent, so the report's "674,120 input, 0 output" is the combined
  figure, not an input and output split

Left out of the corpus: `app/node_modules`, `app/dist`, `app/coverage`, `.venv`, `uv.lock`,
`app/package-lock.json`, `docs/validation/`, `docs/architecture/graph/` (this folder, so the graph
does not read its own output), and everything `.gitignore` already ignores.

## Known limits

- graphify's own health check, run on the merged graph after the refresh, printed: 0 dangling-endpoint
  edges, 4 self-loop edges, 0 collapsed edges. It is not directly comparable with the first build,
  whose check ran on the raw extraction and printed 137 dangling-endpoint edges (imported
  standard-library modules, plus doc-to-doc links to file-level IDs that no node carried) and
  29 collapsed (directed) edges. The refresh gave every re-extracted doc a file-level node. ADR 0002
  was not re-extracted and has no file-level node, and `docs/validation/` is excluded, so links to
  either by file-level ID are not in the graph.
- God nodes mix the central design docs (ADR 0001, the data model, the CPU-only torch decision)
  with config keys (`compilerOptions`, `vitest`) and the test helper `example_envelope()`. Read the
  community sections for the architecture, not the god-node list.
- The largest communities have low cohesion (0.05 to 0.06), so treat their boundaries as rough.
- `data-flow.yaml`, `process.yaml` and `docs/adr/0002-compute-and-numeric-stack.md` were not
  changed since the first build, so the refresh did not re-read them.

## Regenerate

From the repo root, in Claude Code with the graphify skill installed:

```
/graphify . --svg
cp graphify-out/GRAPH_REPORT.md graphify-out/graph.json docs/architecture/graph/
```

`graphify-out/` is gitignored. The exclusions above live in `graphify-out/.graphify_build.json`
(`{"excludes": ["app/node_modules/", "app/dist/", "app/coverage/", ".venv/", "uv.lock",
"app/package-lock.json", "docs/validation/", "docs/architecture/graph/"], "gitignore": true}`), which does not survive a fresh
clone, so recreate it first or the run will include those paths. `--svg` needs `matplotlib`
importable by the interpreter graphify runs under. To refresh after edits, run
`/graphify . --update`, which re-extracts only changed files (and needs the exclusions above,
including `docs/architecture/graph/`). After code-only changes, `graphify update .` refreshes the
code nodes without any LLM call.

Before committing, check the copies for absolute local paths (`/home/`, a user name, `.claude`).
The 2026-10-28 run had none.

## Query

From the repo root, after regenerating:

```
graphify query "how does the deploy guard keep real data out of the build"
graphify path "guardDist.mjs" "ADR 0001: Python core, versioned JSON app" --undirected
graphify explain "example_envelope()"
graphify god-nodes
```

On a fresh clone, where `graphify-out/` does not exist, point at the committed copy:

```
graphify query "..." --graph docs/architecture/graph/graph.json
```

Answers come only from the graph, so quote the `src=` file and line it prints when you cite one.
