async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    let detail = "";
    try { const j = await res.json(); detail = j.message || j.error || JSON.stringify(j); } catch { try { detail = await res.text(); } catch {} }
    throw new Error(`GET ${path} ${res.status}: ${detail}`);
  }
  return await res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const txt = await res.text();
  let obj = null;
  try { obj = JSON.parse(txt); } catch { obj = { raw: txt }; }
  if (!res.ok) throw new Error(`${path} ${res.status}: ${obj?.message || obj?.error || txt}`);
  return obj;
}

function $(id) { return document.getElementById(id); }

function setTab(active) {
  const tabs = [
    { id: "tab_simple", view: "view_simple", key: "simple" },
    { id: "tab_docs", view: "view_docs", key: "docs" },
    { id: "tab_debug", view: "view_debug", key: "debug" },
  ];
  for (const t of tabs) {
    const tabEl = $(t.id);
    const viewEl = $(t.view);
    if (tabEl) tabEl.classList.toggle("active", t.key === active);
    if (viewEl) viewEl.style.display = (t.key === active) ? "block" : "none";
  }
  const dp = $("sidebar_debug_params");
  if (dp) dp.style.display = (active === "debug") ? "block" : "none";
}

function pretty(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

let topics = [];
let selectedTopic = "";
let selectedDoc = null;
let selectedDocDetails = null;
let evidenceSets = [];
let currentSet = null;
let pipelineDebugPollTimer = null;

async function createTopicFromUi() {
  const name = ($("topic_new_name").value || "").trim();
  if (!name) throw new Error("Topic name required.");
  await apiPost("/api/kvi/topics/create", { topic: name });
  await loadTopics();
  if (topics.some(t => t.topic === name)) {
    $("topic_select").value = name;
    $("topic_select_docs").value = name;
    $("topic_select_debug").value = name;
    await onTopicChange(name);
  }
  $("topic_new_name").value = "";
}

async function deleteTopicFromUi() {
  if (!selectedTopic) throw new Error("No topic selected.");
  if (!window.confirm(`Delete "${selectedTopic}"?`)) return;
  await apiPost("/api/kvi/topics/delete", { topic: selectedTopic });
  await loadTopics();
}

async function loadTopics() {
  const resp = await apiGet("/api/kvi/topics");
  topics = resp.items || [];
  for (const sid of ["topic_select", "topic_select_docs", "topic_select_debug"]) {
    const sel = $(sid);
    sel.innerHTML = "";
    for (const t of topics) {
      const opt = document.createElement("option");
      opt.value = t.topic;
      opt.textContent = t.topic;
      sel.appendChild(opt);
    }
  }
  selectedTopic = (topics[0] && topics[0].topic) ? topics[0].topic : "";
  $("topic_select").value = selectedTopic;
  $("topic_select_docs").value = selectedTopic;
  $("topic_select_debug").value = selectedTopic;
  await onTopicChange(selectedTopic);
}

function topicByName(name) { return (topics || []).find(x => x.topic === name) || null; }

async function onTopicChange(topic) {
  selectedTopic = topic;
  // Sync all selectors
  for (const sid of ["topic_select", "topic_select_docs", "topic_select_debug"]) {
    const el = $(sid);
    if (el) el.value = topic;
  }
  const t = topicByName(topic);
  const goalEl = $("topic_goal"); if (goalEl) goalEl.textContent = t && t.goal ? String(t.goal) : "";
  try {
    await loadEvidenceSets();
  } catch (err) {
    console.error("loadEvidenceSets failed:", err);
    $("set_stats").textContent = "Error loading evidence sets: " + String(err.message || err);
  }
}

function parseJsonl(text) {
  const out = [];
  for (const ln of String(text || "").split(/\r?\n/)) {
    const s = ln.trim();
    if (!s) continue;
    try { out.push(JSON.parse(s)); } catch {}
  }
  return out;
}

function toJsonl(records) {
  return (records || []).map(r => { try { return JSON.stringify(r); } catch { return ""; } }).filter(x => x).join("\n") + "\n";
}

async function buildFullPipelineImpl(outputEl, opts) {
  opts = opts || {};
  const aliasesRaw = (($("graph_aliases") || {}).value || "").trim();
  const aliases = [];
  if (aliasesRaw) {
    for (const ln of aliasesRaw.split(/\r?\n/)) {
      const s = ln.trim();
      if (!s) continue;
      try { aliases.push(JSON.parse(s)); } catch {}
    }
  }
  const body = {
    aliases: aliases.length > 0 ? aliases : undefined,
    doc_id: opts.doc_id || undefined,
    use_base_llm_extraction: opts.use_base_llm_extraction,
  };
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/build_full_pipeline`, body);
  if (resp.status === "started") {
    const maxWait = 30 * 60 * 1000;
    const interval = 2000;
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      const st = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/build_full_pipeline/status`);
      if (!st.running) {
        if (st.last_result) return st.last_result;
        if (st.last_error) return { ok: false, message: st.last_error, pipeline_status: {}, triple_kv_items: [] };
        return { ok: false, message: "Pipeline finished but no result", pipeline_status: {}, triple_kv_items: [] };
      }
      await new Promise(r => setTimeout(r, interval));
    }
    throw new Error("Pipeline timed out (30 min)");
  }
  return resp;
}

function formatPipelineResult(resp) {
  if (!resp || !resp.ok) return (resp && resp.message) ? resp.message : "Pipeline FAILED";
  const ps = resp.pipeline_status || {};
  const g = ps.graph || {};
  const kv = ps.triple_kv || {};
  const lines = [
    `=== Build Pipeline Complete ===`,
    `Topic: ${resp.topic}`,
    `Source: ${ps.sentence_source || "(unknown)"}`,
    ``,
    `Sentences compiled: ${ps.sentences_compiled || 0}`,
    `Sentences tagged: ${ps.sentences_tagged ? "OK" : "skipped"}`,
    `Triples extracted: ${ps.triples_extracted || 0}`,
    ``,
    `=== Knowledge Graph ===`,
    `  Nodes (entities): ${g.num_nodes || 0}`,
    `  Triples (edges): ${g.num_triples || 0}`,
    `  Entity index: ${g.num_entity_index_entries || 0}`,
    ``,
    `=== Triple KV Bank ===`,
    `  KV items: ${kv.num_items || 0}`,
    `  Entities: ${kv.num_entities || 0}`,
    `  Layers: ${kv.num_layers || 0}`,
  ];
  if (ps.triple_kv_error) lines.push(`  Error: ${ps.triple_kv_error}`);
  return lines.join("\n");
}

function formatKvItems(items) {
  if (!items || items.length === 0) return "(no KV items)";
  // Group by entity
  const byEntity = {};
  for (const it of items) {
    const e = it.entity || "(unknown)";
    if (!byEntity[e]) byEntity[e] = [];
    byEntity[e].push(it);
  }
  const lines = [];
  for (const [entity, its] of Object.entries(byEntity)) {
    lines.push(`── ${entity} ──`);
    for (const it of its) {
      const tag = it.type === "subject_anchor" ? "[anchor]" : `[${it.relation || "triple"}]`;
      lines.push(`  ${tag} ${it.text} (layers ${it.layers}, ${it.tokens} tok)`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

async function compileGraph() {
  $("compile_log").textContent = "Building full pipeline (Sentences → Triples → Graph → KV)...\nThis may take 1-2 minutes.";
  try {
    const resp = await buildFullPipelineImpl($("compile_log"));
    $("compile_log").textContent = formatPipelineResult(resp) + "\n\n" + formatKvItems(resp.triple_kv_items || []);
  } catch (err) {
    $("compile_log").textContent = "Pipeline FAILED\n\n" + String(err.message || err);
  }
}

async function buildPipelineFromDocs() {
  const summaryEl = $("pipeline_summary");
  const itemsEl = $("pipeline_kv_items");
  const debugEl = $("pipeline_debug_log");
  const btn = $("btn_build_pipeline");
  $("pipeline_result_view").style.display = "block";
  summaryEl.textContent = "Building full pipeline (全部文档)...\nThis may take several minutes.";
  itemsEl.textContent = "Waiting...";
  if (debugEl) debugEl.textContent = "Polling backend runtime log...";
  if (btn) btn.disabled = true;
  startPipelineDebugPoll(selectedTopic);
  try {
    const resp = await buildFullPipelineImpl(summaryEl, {});
    summaryEl.textContent = formatPipelineResult(resp);
    itemsEl.textContent = formatKvItems(resp.triple_kv_items || []);
  } catch (err) {
    summaryEl.textContent = "Pipeline FAILED\n\n" + String(err.message || err);
    itemsEl.textContent = "";
  } finally {
    await stopPipelineDebugPoll(selectedTopic, true);
    if (btn) btn.disabled = false;
  }
}

async function buildPipelineFromCurrentDoc() {
  if (!selectedDoc || !selectedDoc.doc_id) {
    alert("请先在文档列表中打开一篇文档（View Details）");
    return;
  }
  const summaryEl = $("pipeline_summary");
  const itemsEl = $("pipeline_kv_items");
  const debugEl = $("pipeline_debug_log");
  const btn = $("btn_build_pipeline_current_doc");
  $("pipeline_result_view").style.display = "block";
  summaryEl.textContent = `Building graph for current doc only (doc_id=${selectedDoc.doc_id}), using base LLM extraction...`;
  itemsEl.textContent = "Waiting...";
  if (debugEl) debugEl.textContent = "Polling backend runtime log...";
  if (btn) btn.disabled = true;
  startPipelineDebugPoll(selectedTopic);
  try {
    const resp = await buildFullPipelineImpl(summaryEl, {
      doc_id: selectedDoc.doc_id,
      use_base_llm_extraction: true,
    });
    summaryEl.textContent = formatPipelineResult(resp);
    itemsEl.textContent = formatKvItems(resp.triple_kv_items || []);
  } catch (err) {
    summaryEl.textContent = "Pipeline FAILED\n\n" + String(err.message || err);
    itemsEl.textContent = "";
  } finally {
    await stopPipelineDebugPoll(selectedTopic, true);
    if (btn) btn.disabled = false;
  }
}

function renderPipelineDebugStatus(st) {
  const el = $("pipeline_debug_log");
  if (!el) return;
  const lines = [];
  lines.push(`running: ${st && st.running ? "yes" : "no"}`);
  if (st && st.updated_at) lines.push(`updated_at: ${st.updated_at}`);
  if (st && st.last_error) lines.push(`last_error: ${st.last_error}`);
  lines.push("");
  const logs = (st && Array.isArray(st.logs)) ? st.logs : [];
  if (!logs.length) lines.push("(no backend logs yet)");
  else lines.push(...logs);
  el.textContent = lines.join("\n");
}

function startPipelineDebugPoll(topic) {
  stopPipelineDebugPoll(topic, false);
  const tick = async () => {
    if (!topic) return;
    try {
      const st = await apiGet(`/api/kvi/topic/${encodeURIComponent(topic)}/build_full_pipeline/status`);
      renderPipelineDebugStatus(st);
      if (!st.running && pipelineDebugPollTimer) {
        clearInterval(pipelineDebugPollTimer);
        pipelineDebugPollTimer = null;
      }
    } catch (err) {
      const el = $("pipeline_debug_log");
      if (el) el.textContent = "Failed to poll backend log: " + String(err.message || err);
    }
  };
  tick();
  pipelineDebugPollTimer = setInterval(tick, 1500);
}

async function stopPipelineDebugPoll(topic, refreshOnce) {
  if (pipelineDebugPollTimer) {
    clearInterval(pipelineDebugPollTimer);
    pipelineDebugPollTimer = null;
  }
  if (refreshOnce && topic) {
    try {
      const st = await apiGet(`/api/kvi/topic/${encodeURIComponent(topic)}/build_full_pipeline/status`);
      renderPipelineDebugStatus(st);
    } catch {}
  }
}

async function loadEvidenceSets() {
  if (!selectedTopic) return;
  const t = topicByName(selectedTopic);
  $("evidence_sets_dir").value = (t && t.evidence_sets_dir) ? t.evidence_sets_dir : "";
  const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_sets`);
  evidenceSets = resp.items || [];
  const sel = $("set_select");
  sel.innerHTML = "";
  for (const s of evidenceSets) {
    const opt = document.createElement("option");
    opt.value = s.name;
    opt.textContent = `${s.enabled ? "+" : "-"} ${s.name} (${s.count || 0})`;
    sel.appendChild(opt);
  }
  if (evidenceSets.length > 0) {
    await loadEvidenceSet(evidenceSets[0].name);
  } else {
    currentSet = null;
    $("set_stats").textContent = "No evidence sets.";
    $("set_raw").value = "";
    $("doc_group_view").textContent = "";
  }
}

async function loadEvidenceSet(name) {
  try {
    const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_set/${encodeURIComponent(name)}`);
    currentSet = resp;
    $("set_select").value = name;
    $("set_enabled").value = resp.enabled ? "true" : "false";
    $("set_stats").textContent = `${name}  |  records: ${resp.count || 0}  |  enabled: ${resp.enabled}`;
    $("set_raw").value = toJsonl(resp.records || []);
    $("doc_group_view").textContent = pretty(resp.by_source || {});
  } catch (err) {
    console.error("loadEvidenceSet failed:", name, err);
    $("set_stats").textContent = "Error loading set: " + String(err.message || err);
    $("set_raw").value = "";
    $("doc_group_view").textContent = "Error: " + String(err.message || err);
  }
}

async function saveCurrentSet() {
  if (!currentSet) throw new Error("No set selected.");
  const name = currentSet.name;
  const enabled = ($("set_enabled").value || "true") === "true";
  const maxSentenceTokens = Number(($("sent_token_budget").value || "128").trim());
  const records = parseJsonl($("set_raw").value || "");
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_set/save`, {
    name, enabled, records,
    max_sentence_tokens: Number.isFinite(maxSentenceTokens) ? maxSentenceTokens : 128,
  });
  const ss = resp.split_stats || null;
  $("set_stats").textContent = `Saved: ${resp.name}  |  count: ${resp.count}  |  enabled: ${resp.enabled}` + (ss ? `  |  split: ${ss.split_records||0}->${ss.generated_records||0}` : "");
  await loadEvidenceSets();
  await loadEvidenceSet(name);
}

async function createEvidenceSet() {
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_sets/create`, {});
  await loadEvidenceSets();
  await loadEvidenceSet(resp.name);
}

function buildNewSentenceRecords() {
  const raw = ($("sent_claim").value || "");
  const parts = raw.includes("###") ? raw.split("###") : [raw];
  const claims = parts.map(x => String(x || "").trim()).filter(x => x);
  if (!claims.length) throw new Error("Claim required.");
  const sourceId = ($("sent_source_id").value || "").trim() || null;
  const doi = ($("sent_doi").value || "").trim() || null;
  const title = ($("sent_title").value || "").trim() || null;
  const author = ($("sent_author").value || "").trim() || null;
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ts = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  return claims.map(claim => ({ id: null, topic: selectedTopic, claim, source_id: sourceId, source_ref: { doi, title }, created_at: ts, updated_at: ts, author, tags: [] }));
}

async function addSentenceToSet() {
  if (!currentSet) throw new Error("No set selected.");
  const recs = buildNewSentenceRecords();
  const records = parseJsonl($("set_raw").value || "");
  for (const r of recs) records.push(r);
  $("set_raw").value = toJsonl(records);
  $("sent_claim").value = "";
  await saveCurrentSet();
}

async function loadDocsList() {
  if (!selectedTopic) return;
  try {
    const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/docs`);
    const el = $("docs_list");
    el.innerHTML = "";
    selectedDoc = null;
    selectedDocDetails = null;
    $("btn_import_doc_details").disabled = true;
    $("docs_list_view").style.display = "block";
    $("doc_detail_view").style.display = "none";
    for (const d of resp.items || []) {
      const card = document.createElement("div"); card.className = "card";
      const top = document.createElement("div"); top.className = "top";
      const label = d.title || d.pdf_name || d.doc_id;
      const left = document.createElement("div"); left.className = "mono"; left.textContent = label;
      const badge = document.createElement("span"); badge.className = "badge ok";
      badge.textContent = `${d.block_count || 0} blocks`;
      top.appendChild(left); top.appendChild(badge);
      const meta = document.createElement("div"); meta.className = "note";
      meta.textContent = `doc_id=${d.doc_id}` + (d.doi ? `  doi=${d.doi}` : "") + (d.publication_year ? `  year=${d.publication_year}` : "");
      const fileLine = document.createElement("div"); fileLine.className = "note";
      fileLine.style.color = "var(--text-muted)";
      fileLine.textContent = d.pdf_name ? `file: ${d.pdf_name}` : "";
      const btns = document.createElement("div"); btns.className = "btns";
      const open = document.createElement("button"); open.className = "primary"; open.textContent = "View Details";
      open.onclick = async (ev) => { ev.stopPropagation(); selectedDoc = d; $("docs_list_view").style.display = "none"; $("doc_detail_view").style.display = "block"; await loadDocDetails(d.doc_id); };
      btns.appendChild(open);
      card.appendChild(top); card.appendChild(fileLine); card.appendChild(meta); card.appendChild(btns); el.appendChild(card);
    }
    if ((resp.items || []).length === 0) {
      el.innerHTML = '<div class="note">No documents found. Run PDF ingestion first to create blocks.evidence.jsonl.</div>';
    }
  } catch (err) {
    $("docs_list").innerHTML = '<div class="note">Error loading documents: ' + String(err.message || err) + '</div>';
  }
}

function renderKeyNotes(notes) {
  const root = $("doc_key_notes");
  root.innerHTML = "";
  const arr = Array.isArray(notes) ? notes : [];
  if (arr.length === 0) arr.push("");
  for (const n of arr) {
    const row = document.createElement("div"); row.className = "row";
    const input = document.createElement("input"); input.placeholder = "Key note"; input.value = String(n || "");
    input.style.flex = "1";
    const del = document.createElement("button"); del.className = "danger"; del.textContent = "x";
    del.onclick = () => row.remove();
    row.appendChild(input); row.appendChild(del); root.appendChild(row);
  }
}

function renderRelations(relations) {
  const root = $("doc_relations");
  root.innerHTML = "";
  const arr = Array.isArray(relations) ? relations : [];
  if (arr.length === 0) arr.push({ variable: "", operator: ">=", value: "" });
  for (const r of arr) root.appendChild(createRelationRow(r));
}

const REL_OPS = [">", ">=", "<", "<=", "=", "!=", "range"];

function _splitRangeValue(value) {
  const parts = String(value || "").split(",").map(x => String(x || "").trim()).filter(Boolean);
  return [parts[0] || "", parts[1] || ""];
}

function createRelationRow(r) {
  const row = document.createElement("div"); row.className = "row"; row.style.alignItems = "center";
  const variable = document.createElement("input");
  variable.placeholder = "变量名";
  variable.style.flex = "1";
  variable.value = String((r && r.variable) || "");

  const op = document.createElement("select");
  for (const o of REL_OPS) {
    const opt = document.createElement("option"); opt.value = o; opt.textContent = o;
    if (String((r && r.operator) || "") === o) opt.selected = true;
    op.appendChild(opt);
  }

  const valueWrap = document.createElement("div");
  valueWrap.style.display = "flex";
  valueWrap.style.gap = "6px";
  valueWrap.style.flex = "1";
  valueWrap.className = "rel-value-wrap";

  const del = document.createElement("button"); del.className = "danger"; del.textContent = "x";
  del.onclick = () => row.remove();

  function renderValueInputs() {
    valueWrap.innerHTML = "";
    const curOp = String(op.value || "");
    if (curOp === "range") {
      const [mn, mx] = _splitRangeValue((r && r.value) || "");
      const minInp = document.createElement("input");
      minInp.placeholder = "min(文本)";
      minInp.value = mn;
      minInp.dataset.role = "range_min";
      minInp.style.width = "110px";
      const maxInp = document.createElement("input");
      maxInp.placeholder = "max(文本)";
      maxInp.value = mx;
      maxInp.dataset.role = "range_max";
      maxInp.style.width = "110px";
      const hint = document.createElement("span");
      hint.className = "note";
      hint.textContent = "导出为 min,max 字符串";
      valueWrap.appendChild(minInp); valueWrap.appendChild(maxInp); valueWrap.appendChild(hint);
    } else {
      const val = document.createElement("input");
      val.placeholder = "数值（文本）";
      val.value = String((r && r.value) || "");
      val.dataset.role = "single_value";
      val.style.flex = "1";
      valueWrap.appendChild(val);
    }
  }

  op.onchange = () => {
    // reset cached r.value so UI won't carry wrong shape value
    r.value = "";
    renderValueInputs();
  };
  renderValueInputs();

  row.appendChild(variable);
  row.appendChild(op);
  row.appendChild(valueWrap);
  row.appendChild(del);
  return row;
}

function validateAndNormalizeRelations(relations) {
  const out = [];
  for (let i = 0; i < (relations || []).length; i++) {
    const r = relations[i] || {};
    const variable = String(r.variable || "").trim();
    const operator = String(r.operator || "").trim();
    const value = String(r.value || "").trim();
    if (!(variable || operator || value)) continue;
    if (!variable) throw new Error(`Relation #${i + 1}: 变量名不能为空`);
    if (!REL_OPS.includes(operator)) throw new Error(`Relation #${i + 1}: 比较符无效（支持: ${REL_OPS.join(", ")}）`);
    if (!value) throw new Error(`Relation #${i + 1}: 数值不能为空`);

    if (operator === "range") {
      const parts = value.split(",").map(x => String(x || "").trim()).filter(Boolean);
      if (parts.length !== 2) throw new Error(`Relation #${i + 1}: range 格式应为 min,max`);
      out.push({ variable, operator, value: `${parts[0]},${parts[1]}` });
      continue;
    }
    out.push({ variable, operator, value });
  }
  return out;
}

function collectDocDetailsFromForm() {
  const abstract = ($("doc_abstract").value || "").trim();
  const key_notes = [];
  for (const row of ($("doc_key_notes").children || [])) {
    const inp = row.querySelector("input");
    const t = inp ? String(inp.value || "").trim() : "";
    if (t) key_notes.push(t);
  }
  const relations = [];
  for (const row of ($("doc_relations").children || [])) {
    const sel = row.querySelector("select");
    const variableInp = row.querySelector("input");
    const variable = variableInp ? String(variableInp.value || "").trim() : "";
    const operator = sel ? String(sel.value || "").trim() : "";
    let value = "";
    if (operator === "range") {
      const minInp = row.querySelector('input[data-role="range_min"]');
      const maxInp = row.querySelector('input[data-role="range_max"]');
      const mn = minInp ? String(minInp.value || "").trim() : "";
      const mx = maxInp ? String(maxInp.value || "").trim() : "";
      value = [mn, mx].filter(Boolean).join(",");
    } else {
      const valInp = row.querySelector('input[data-role="single_value"]');
      value = valInp ? String(valInp.value || "").trim() : "";
    }
    if (variable || operator || value) relations.push({ variable, operator, value });
  }
  const relationsNorm = validateAndNormalizeRelations(relations);
  return { abstract, key_notes, relations: relationsNorm };
}

async function loadDocDetails(docId) {
  const detailsResp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/doc/${encodeURIComponent(docId)}/details`);
  const d = (detailsResp && detailsResp.details) ? detailsResp.details : {};
  selectedDocDetails = d;
  $("doc_abstract").value = d.abstract || "";
  renderKeyNotes(d.key_notes || []);
  renderRelations(d.relations || []);
  $("btn_import_doc_details").disabled = !(selectedDoc && selectedDoc.approved);
}

async function saveDocDetails() {
  if (!selectedDoc) throw new Error("Select a doc.");
  const payload = collectDocDetailsFromForm();
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/doc/${encodeURIComponent(selectedDoc.doc_id)}/details/save`, payload);
  selectedDocDetails = resp.details || payload;
  alert(`Saved details at ${selectedDocDetails.updated_at || "(now)"}`);
}

async function importDocDetails() {
  if (!selectedDoc) throw new Error("Select a doc.");
  if (!selectedDoc.approved) throw new Error("Not approved.");
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/doc/${encodeURIComponent(selectedDoc.doc_id)}/import_details_to_evidence`, {});
  alert(`Imported details: set=${resp.evidence_set} appended=${resp.appended} lines=${resp.num_lines}`);
  await loadEvidenceSets();
}

/* ==================== Inference Debug ==================== */

async function runDebug() {
  const prompt = ($("debug_prompt").value || "").trim();
  if (!prompt) throw new Error("Prompt required.");
  const topK = Number(($("debug_top_k").value || "2").trim());
  const mode = ($("debug_mode").value || "graphC").trim();
  const wAnn = Number(($("route_w_ann").value || "1.0").trim());
  const wIntent = Number(($("route_w_intent").value || "0.6").trim());
  const wQuality = Number(($("route_w_quality").value || "0.2").trim());
  const rerankNoAnn = ($("route_rerank_without_ann").value || "false") === "true";
  const modeAUseLlmIntent = ($("modeA_use_llm_intent").value || "false") === "true";

  // Clear all output fields
  const allFields = [
    "out_cli", "out_debug_log",
    "out_graph", "out_graph_raw", "out_graph_rag", "out_graph_base_llm",
    "out_graph_entity_ctx", "out_graph_evidence", "out_graph_kv_injection",
  ];
  for (const f of allFields) { const el = $(f); if (el) el.textContent = "Running..."; }
  const allStatus = ["out_graph_status", "out_graph_rag_status"];
  for (const s of allStatus) { const el = $(s); if (el) el.textContent = ""; }

  const params = {
    prompt,
    top_k: Number.isFinite(topK) ? topK : 8,
    route_w_ann: Number.isFinite(wAnn) ? wAnn : 1.0,
    route_w_intent: Number.isFinite(wIntent) ? wIntent : 0.6,
    route_w_quality: Number.isFinite(wQuality) ? wQuality : 0.2,
    route_rerank_without_ann: rerankNoAnn,
    route_llm_intent_enable: modeAUseLlmIntent,
  };

  let resp = null, respRag = null;

  // --- Mode A (KVI+RAG) ---
  resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/modeA_graph`, params);
  // Also get RAG-only for comparison
  try {
    respRag = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/modeA_rag`, params);
  } catch (e) { /* RAG comparison is optional */ }

  const r = resp && resp.result ? resp.result : {};
  $("out_cli").textContent = (resp && resp.cmd) ? resp.cmd : "(no cmd)";

  // --- Build debug log ---
  try {
    const debugObj = {};
    if (r.graph_debug) debugObj.graph_debug = r.graph_debug;
    if (r.kv_injection_debug) debugObj.kv_injection_debug = r.kv_injection_debug;
    if (r.grounding_report) debugObj.graph_grounding = r.grounding_report;
    if (r.intent) debugObj.graph_intent = r.intent;
    if (resp && resp.stderr_tail) debugObj.graph_stderr = resp.stderr_tail;
    if (respRag && respRag.result) {
      if (respRag.result.routing_debug) debugObj.rag_routing_debug = respRag.result.routing_debug;
      if (respRag.result.grounding_report) debugObj.rag_grounding = respRag.result.grounding_report;
    }
    $("out_debug_log").textContent = pretty(debugObj);
  } catch (e) { $("out_debug_log").textContent = String(e && e.message ? e.message : e); }

  // --- Render outputs ---
  $("out_graph").textContent = r.diagnosis_result || "";
  $("out_graph_status").textContent = r.diagnosis_result ? "status: OK" : "status: EMPTY";
  $("out_graph_raw").textContent = r.diagnosis_result_raw || "";
  $("out_graph_base_llm").textContent = r.base_llm_result || "";
  $("out_graph_entity_ctx").textContent = r.entity_context || "(none)";
  const evTexts = r.evidence_texts || [];
  const evSrc = r.evidence_source || [];
  const verbatimSet = new Set((r.verbatim_evidence || []).map(v => v.trim()));
  $("out_graph_evidence").textContent = evTexts.map((t, i) => {
    const src = evSrc[i] || "graph";
    let tag = src.startsWith("text_search") ? " [text]" : "";
    if (verbatimSet.has(t.trim())) tag += " [verbatim→直出]";
    return `${i+1}. ${t}${tag}`;
  }).join("\n") || "(none)";
  // KV Injection info (with DRM scores)
  const kvDbg = r.kv_injection_debug || {};
  if (kvDbg.enabled) {
    const lines = [];
    // DRM scoring summary
    const drmScores = kvDbg.drm_scores || [];
    if (drmScores.length > 0) {
      lines.push(`=== DRM Scoring (${drmScores.length} walk triples) ===`);
      for (const ds of drmScores) {
        const marker = ds.drm_score >= (kvDbg.drm_threshold || 0.05) ? '✓' : '✗';
        lines.push(`  ${marker} [${ds.relation}] ${ds.subject}→${ds.object} drm=${ds.drm_score}`);
      }
      lines.push(`--- threshold=${kvDbg.drm_threshold}, passed=${kvDbg.drm_passed}, gated=${kvDbg.gated_count}, budget=${kvDbg.budget_selected}`);
      lines.push('');
    }
    // Active KV items
    if (kvDbg.active_items && kvDbg.active_items.length > 0) {
      lines.push(`=== Injected KV Items (${kvDbg.active_items.length}) ===`);
      for (const it of kvDbg.active_items) {
        lines.push(`[${it.type}] ${it.text} (${it.relation || 'anchor'}, layers ${it.layers}, ${it.tokens} tok)`);
      }
      lines.push(`--- total KV tokens: ${kvDbg.total_kv_tokens || 0}, seq_len: ${kvDbg.assembled_seq_len || 0}`);
    } else {
      lines.push('(no KV items injected after DRM filtering)');
    }
    $("out_graph_kv_injection").textContent = lines.join("\n");
  } else {
    $("out_graph_kv_injection").textContent = kvDbg.enabled === false
      ? `KV injection disabled (${kvDbg.reason || 'no triple_kvbank'})`
      : "(no KV items matched)";
  }
  const rr = respRag && respRag.result ? respRag.result : {};
  $("out_graph_rag").textContent = rr.diagnosis_result || "";
  $("out_graph_rag_status").textContent = rr.diagnosis_result ? "status: OK" : "";
}

/* ==================== Wiring ==================== */

function wire() {
  try {
    const tabSimple = $("tab_simple"), tabDocs = $("tab_docs"), tabDebug = $("tab_debug");
    if (tabSimple) tabSimple.onclick = () => { setTab("simple"); loadEvidenceSets().catch(err => console.warn("tab reload evidence:", err)); };
    if (tabDocs) tabDocs.onclick = () => { setTab("docs"); loadDocsList().catch(err => console.warn("tab reload docs:", err)); };
    if (tabDebug) tabDebug.onclick = () => setTab("debug");

    const topicSelect = $("topic_select"), topicSelectDocs = $("topic_select_docs"), topicSelectDebug = $("topic_select_debug");
    if (topicSelect) topicSelect.onchange = async (e) => await onTopicChange(e.target.value);
    if (topicSelectDocs) topicSelectDocs.onchange = async (e) => { await onTopicChange(e.target.value); await loadDocsList(); };
    if (topicSelectDebug) topicSelectDebug.onchange = async (e) => await onTopicChange(e.target.value);

    const btnCompile = $("btn_compile_graph"), btnBuild = $("btn_build_pipeline");
    if (btnCompile) btnCompile.onclick = () => compileGraph().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    if (btnBuild) btnBuild.onclick = () => buildPipelineFromDocs().catch(err => { const el = $("pipeline_summary"); if (el) el.textContent = String(err.message || err); });
    const btnBuildCurrentDoc = $("btn_build_pipeline_current_doc");
    if (btnBuildCurrentDoc) btnBuildCurrentDoc.onclick = () => buildPipelineFromCurrentDoc().catch(err => { alert(String(err.message || err)); const el = $("pipeline_summary"); if (el) el.textContent = "Pipeline FAILED\n\n" + String(err.message || err); });
    const btnCreateTopic = $("btn_create_topic"), btnDeleteTopic = $("btn_delete_topic");
    if (btnCreateTopic) btnCreateTopic.onclick = () => createTopicFromUi().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    if (btnDeleteTopic) btnDeleteTopic.onclick = () => deleteTopicFromUi().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    const btnReloadSets = $("btn_reload_sets"), setSelect = $("set_select");
    if (btnReloadSets) btnReloadSets.onclick = () => loadEvidenceSets().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    if (setSelect) setSelect.onchange = async (e) => { await loadEvidenceSet(e.target.value).catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); }); };
    const btnSaveSet = $("btn_save_set"), btnCreateSet = $("btn_create_set"), btnAddSentence = $("btn_add_sentence");
    if (btnSaveSet) btnSaveSet.onclick = () => saveCurrentSet().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    if (btnCreateSet) btnCreateSet.onclick = () => createEvidenceSet().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    if (btnAddSentence) btnAddSentence.onclick = () => addSentenceToSet().catch(err => { const el = $("compile_log"); if (el) el.textContent = String(err.message || err); });
    const btnSaveDetails = $("btn_save_doc_details"), btnImportDetails = $("btn_import_doc_details");
    if (btnSaveDetails) btnSaveDetails.onclick = () => saveDocDetails().catch(err => { alert(String(err.message || err)); });
    if (btnImportDetails) btnImportDetails.onclick = () => importDocDetails().catch(err => { alert(String(err.message || err)); });
    const btnAddKeyNote = $("btn_add_key_note");
    if (btnAddKeyNote) btnAddKeyNote.onclick = () => {
      const root = $("doc_key_notes");
      if (!root) return;
      const row = document.createElement("div"); row.className = "row";
      const input = document.createElement("input"); input.placeholder = "Key note"; input.style.flex = "1";
      const del = document.createElement("button"); del.className = "danger"; del.textContent = "x";
      del.onclick = () => row.remove();
      row.appendChild(input); row.appendChild(del); root.appendChild(row);
    };
    const btnAddRelation = $("btn_add_relation");
    if (btnAddRelation) btnAddRelation.onclick = () => {
      const root = $("doc_relations");
      if (root) root.appendChild(createRelationRow({ variable: "", operator: ">=", value: "" }));
    };
    const btnBackToDocs = $("btn_back_to_docs"), btnRunDebug = $("btn_run_debug");
    if (btnBackToDocs) btnBackToDocs.onclick = () => { selectedDoc = null; $("docs_list_view").style.display = "block"; $("doc_detail_view").style.display = "none"; };
    if (btnRunDebug) btnRunDebug.onclick = () => runDebug().catch(err => { const el = $("out_debug_log"); if (el) el.textContent = String(err.message || err); });

    const slider = $("debug_topk_slider"), show = $("debug_topk_n"), topKInput = $("debug_top_k");
    if (slider && show) {
      const sync = () => { show.textContent = String(slider.value || "2"); if (topKInput) topKInput.value = String(slider.value || "2"); };
      slider.addEventListener("input", sync);
      sync();
    }
  } catch (err) {
    console.error("wire() failed:", err);
  }
}

async function init() {
  wire();
  try {
    await loadTopics();
    $("topic_select_docs").value = selectedTopic;
    $("topic_select_debug").value = selectedTopic;
  } catch (err) {
    console.error("init: loadTopics failed:", err);
    // UI navigation should still work even if topic loading fails
  }
}

window.addEventListener("DOMContentLoaded", () => init().catch(err => console.error("init fatal:", err)));
