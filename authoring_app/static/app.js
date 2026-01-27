async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`GET ${path} ${res.status}`);
  return await res.json();
}

async function apiSend(path, method, body) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : "{}",
  });
  const txt = await res.text();
  let obj = null;
  try { obj = JSON.parse(txt); } catch { obj = { raw: txt }; }
  if (!res.ok) throw new Error(`${method} ${path} ${res.status}: ${obj?.message || obj?.error || txt}`);
  return obj;
}

function $(id) { return document.getElementById(id); }

function badgeClass(status) {
  const s = (status || "").toLowerCase();
  if (s === "approved") return "badge approved";
  if (s === "rejected") return "badge rejected";
  if (s === "reviewed") return "badge reviewed";
  return "badge draft";
}

function truncate(s, n) {
  s = (s || "").trim();
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}

function safeJsonParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}

function prettyJson(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return ""; }
}

let current = null;
let rejectionCodes = [];
let paging = { offset: 0, limit: 200, total: 0, lastQs: "" };
let serverDefaultKbId = "";

async function refreshList() {
  const qs = new URLSearchParams();
  const status = $("filter_status").value;
  const kb = $("filter_kb").value.trim();
  const sem = $("filter_semantic_type").value;
  const q = $("filter_q").value.trim();
  if (status) qs.set("status", status);
  if (kb) qs.set("kb_id", kb);
  if (sem) qs.set("semantic_type", sem);
  if (q) qs.set("q", q);
  qs.set("offset", String(paging.offset || 0));
  qs.set("limit", String(paging.limit || 200));
  const data = await apiGet(`/api/evidence?${qs.toString()}`);
  paging.total = data.total || data.count || 0;
  $("count").textContent = `${data.count} / ${paging.total} items (offset=${data.offset || 0})`;
  const el = $("list");
  if ((paging.offset || 0) === 0) el.innerHTML = "";
  for (const it of data.items || []) {
    const card = document.createElement("div");
    card.className = "card";
    card.onclick = () => loadEvidence(it.evidence_id);
    const top = document.createElement("div");
    top.className = "top";

    const left = document.createElement("div");
    left.className = "id";
    left.textContent = it.evidence_id || "(no id)";

    const badge = document.createElement("div");
    badge.className = badgeClass(it.status);
    badge.textContent = (it.status || "draft").toUpperCase();

    top.appendChild(left);
    top.appendChild(badge);

    const claim = document.createElement("div");
    claim.className = "claim";
    claim.textContent = truncate(it.claim || "", 220) || "(empty claim)";

    const meta = document.createElement("div");
    meta.className = "meta";
    const kid = it.kb_id || it.effective_kb_id || "";
    meta.textContent = `semantic_type=${it.semantic_type || ""}  kb_id=${kid}  polarity=${it.polarity || ""}`;

    card.appendChild(top);
    card.appendChild(claim);
    card.appendChild(meta);

    if ((it.status || "").toLowerCase() === "rejected" && it.rejection) {
      const rej = document.createElement("div");
      rej.className = "rejection";
      rej.innerHTML = `<div class="code">${it.rejection.code || "REJECTED"}</div><div class="msg">${it.rejection.message || ""}</div>`;
      card.appendChild(rej);
    }

    el.appendChild(card);
  }
}

function fillForm(u) {
  current = u;
  $("evidence_id").value = u.evidence_id || "";
  $("status").value = u.status || "draft";
  $("semantic_type").value = u.semantic_type || "generic";
  $("kb_id").value = u.kb_id || u.schema_id || serverDefaultKbId || "";
  $("polarity").value = u.polarity || "neutral";
  $("claim").value = u.claim || "";
  $("slot_projection").value = prettyJson(u.slot_projection || {});
  $("provenance").value = prettyJson(u.provenance || {});
  $("external_refs").value = prettyJson(u.external_refs || {});

  const rej = u.rejection || null;
  $("rej_block").style.display = rej ? "block" : "none";
  $("rej_code").value = rej?.code || (rejectionCodes[0] || "");
  $("rej_message").value = rej?.message || "";
  $("rej_details").value = prettyJson(rej?.details || {});
  $("rej_confidence").value = (rej?.confidence != null ? String(rej.confidence) : "0.9");
}

async function loadEvidence(evidenceId) {
  const u = await apiGet(`/api/evidence/${encodeURIComponent(evidenceId)}`);
  fillForm(u);
  $("editor_title").textContent = `Evidence ${u.evidence_id}`;
}

function buildPayloadFromForm() {
  const slotProj = safeJsonParse($("slot_projection").value) || {};
  const prov = safeJsonParse($("provenance").value) || {};
  const exr = safeJsonParse($("external_refs").value) || {};
  return {
    evidence_id: $("evidence_id").value.trim(),
    status: $("status").value,
    semantic_type: $("semantic_type").value,
    // Backing field is still `schema_id` in JSON for runtime compat, but UI calls it kb_id.
    schema_id: $("kb_id").value.trim(),
    kb_id: $("kb_id").value.trim(),
    polarity: $("polarity").value,
    claim: $("claim").value,
    slot_projection: slotProj,
    provenance: prov,
    external_refs: exr,
    rejection: current?.rejection || null,
  };
}

async function createNew() {
  const u = {
    evidence_id: "",
    status: "draft",
    semantic_type: "generic",
    schema_id: serverDefaultKbId || "",
    kb_id: serverDefaultKbId || "",
    polarity: "neutral",
    claim: "",
    slot_projection: {},
    provenance: { source_type: "guideline", organization: null, document_title: "", publication_year: null, page_range: null },
    external_refs: { document_id: null, pmid: null, orcid: null, title: null, abstract: null, authors: [], published_at: null },
    rejection: null,
  };
  fillForm(u);
  $("editor_title").textContent = "New Evidence (draft)";
}

async function saveDraft() {
  const payload = buildPayloadFromForm();
  // Required fields (product-facing):
  // - claim: required
  // - semantic_type: required (fixed set)
  // - kb_id: required (auto-filled from server default when available)
  if (!String(payload.claim || "").trim()) throw new Error("claim * is required.");
  if (!String(payload.semantic_type || "").trim()) throw new Error("semantic_type * is required.");
  if (!String(payload.schema_id || "").trim() && serverDefaultKbId) payload.schema_id = serverDefaultKbId;
  if (!String(payload.schema_id || "").trim()) {
    throw new Error("kb_id * is required (or start server with --default_kb_id).");
  }
  if (!payload.evidence_id) {
    // create
    const saved = await apiSend("/api/evidence", "POST", payload);
    fillForm(saved);
  } else {
    const saved = await apiSend(`/api/evidence/${encodeURIComponent(payload.evidence_id)}`, "PUT", payload);
    fillForm(saved);
  }
  await refreshList();
}

async function approve() {
  const id = $("evidence_id").value.trim();
  if (!id) throw new Error("Save before approve (needs evidence_id).");
  const saved = await apiSend(`/api/evidence/${encodeURIComponent(id)}/approve`, "POST", {});
  fillForm(saved);
  await refreshList();
}

async function reject() {
  const id = $("evidence_id").value.trim();
  if (!id) throw new Error("Save before reject (needs evidence_id).");
  const details = safeJsonParse($("rej_details").value) || {};
  const conf = Number($("rej_confidence").value || "0.9");
  const payload = {
    rejection: {
      code: $("rej_code").value,
      message: $("rej_message").value.trim(),
      details,
      confidence: Number.isFinite(conf) ? conf : 0.9,
    },
  };
  const saved = await apiSend(`/api/evidence/${encodeURIComponent(id)}/reject`, "POST", payload);
  fillForm(saved);
  await refreshList();
}

async function init() {
  // Load server defaults (kb_id etc.)
  try {
    const cfg = await apiGet("/api/config");
    if (cfg && cfg.default_kb_id) {
      serverDefaultKbId = cfg.default_kb_id;
      $("import_kb_id").value = cfg.default_kb_id;
      $("kb_id").value = cfg.default_kb_id;
    }
  } catch {}

  const codes = await apiGet("/api/rejection_codes");
  rejectionCodes = codes.codes || [];
  const sel = $("rej_code");
  sel.innerHTML = "";
  for (const c of rejectionCodes) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  }

  $("btn_import_blocks").onclick = () => importBlocksJsonl().catch(showErr);
  $("btn_import_evidence_blocks").onclick = () => importEvidenceBlocksJsonl().catch(showErr);

  $("btn_search").onclick = () => {
    paging.offset = 0;
    refreshList().catch(showErr);
  };
  $("btn_refresh").onclick = () => refreshList().catch(showErr);
  $("btn_new").onclick = () => createNew().catch(showErr);
  $("btn_save").onclick = () => saveDraft().catch(showErr);
  $("btn_approve").onclick = () => approve().catch(showErr);
  $("btn_reject").onclick = () => reject().catch(showErr);
  $("filters").oninput = debounce(() => { paging.offset = 0; refreshList().catch(showErr); }, 300);

  await refreshList();
  await createNew();
}

async function importBlocksJsonl() {
  const kbId = $("import_kb_id").value.trim();
  const sem = $("import_semantic_type").value;
  const pth = $("import_blocks_path").value.trim();
  const maxBlocks = Number(($("import_max_blocks").value || "0").trim());
  if (!pth) throw new Error("Import requires blocks.jsonl path.");
  $("import_result").value = "Importing blocks.jsonl ...";
  const out = await apiSend("/api/import/blocks", "POST", {
    blocks_jsonl: pth,
    kb_id: kbId || null,
    default_semantic_type: sem,
    max_blocks: Number.isFinite(maxBlocks) ? maxBlocks : 0,
    evidence_type: "pdf_block",
  });
  $("import_result").value = prettyJson(out);
  paging.offset = 0;
  await refreshList();
}

async function importEvidenceBlocksJsonl() {
  const kbId = $("import_kb_id").value.trim();
  const sem = $("import_semantic_type").value;
  const pth = $("import_evidence_blocks_path").value.trim();
  const metaPath = $("import_docs_meta_path").value.trim();
  if (!pth) throw new Error("Import requires blocks.evidence.jsonl path.");
  $("import_result").value = "Importing blocks.evidence.jsonl ...";
  const out = await apiSend("/api/import/blocks.evidence", "POST", {
    blocks_evidence_jsonl: pth,
    docs_meta_jsonl: metaPath || null,
    kb_id: kbId || null,
    default_semantic_type: sem,
    evidence_type: "extractive_suggestion",
  });
  $("import_result").value = prettyJson(out);
  paging.offset = 0;
  await refreshList();
}

function showErr(e) {
  const msg = (e && e.message) ? e.message : String(e);
  $("err").textContent = msg;
  $("err").style.display = "block";
  setTimeout(() => { $("err").style.display = "none"; }, 6000);
}

function debounce(fn, ms) {
  let t = null;
  return () => {
    if (t) clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

window.addEventListener("DOMContentLoaded", () => init().catch(showErr));

