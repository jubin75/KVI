## KVI UI (MVP)

This is a **zero-dependency** local web UI for KVI (Key Value Injection) knowledge curation:

- Simple: manage sentence+meta Evidence Sets (JSONL) and compile KVBank
- Doc: curate knowledge from doc_meta + blocks.evidence.jsonl, then import to Evidence Sets
- Debug: run simple pipeline injection tests

### Run

Start server:

```bash
python -u external_kv_injection/authoring_app/server.py --port 8765
```

Open:

`http://127.0.0.1:8765`

### Notes

- Evidence Sets are stored under each topic `build.work_dir/evidence_sets/`.

