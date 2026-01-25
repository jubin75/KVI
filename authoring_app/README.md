## Authoring UI (MVP)

This is a **zero-dependency** local web UI for `docs/11_Knowledge_Authoring_Layer.md`:

- Evidence list (semantic_type / schema / status)
- Evidence edit (structured fields + original text)
- Evidence review (approve / reject)
- Rejection reason rendering using the **frozen** `rejection.code` set

### Run

Prepare an authoring DB file (JSONL):

```bash
cp external_kv_injection/examples/authoring_evidence_units.sample.jsonl /tmp/evidence_units.jsonl
```

Start server:

```bash
python -u external_kv_injection/authoring_app/server.py --db /tmp/evidence_units.jsonl --port 8765
```

Open:

`http://127.0.0.1:8765`

### Notes

- This UI edits the JSONL file **in place** (rewrites the file on each save).
- Only **approved** evidence should be exported to runtime KVBank (see `scripts/export_authoring_evidence_runtime_jsonl.py`).

