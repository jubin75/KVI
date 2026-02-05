async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`GET ${path} ${res.status}`);
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
    $(t.id).classList.toggle("active", t.key === active);
    $(t.view).style.display = (t.key === active) ? "block" : "none";
  }
}

function pretty(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

let topics = [];
let selectedTopic = "";
let selectedDoc = null; // {doc_id, approved, ...}
let evidenceSets = []; // [{name,path,enabled,count}]
let currentSet = null; // {name, enabled, records, by_source}

async function createTopicFromUi() {
  const name = ($("topic_new_name").value || "").trim();
  if (!name) throw new Error("请输入主题库名字。");
  await apiPost("/api/kvi/topics/create", { topic: name });
  await loadTopics();
  // select the newly created topic if present
  if (topics.some(t => t.topic === name)) {
    $("topic_select").value = name;
    $("topic_select_docs").value = name;
    $("topic_select_debug").value = name;
    await onTopicChange(name);
  }
  $("topic_new_name").value = "";
}

async function deleteTopicFromUi() {
  if (!selectedTopic) throw new Error("未选择主题库。");
  const ok = window.confirm(`确定删除主题库 "${selectedTopic}" 吗？此操作不可恢复。`);
  if (!ok) return;
  await apiPost("/api/kvi/topics/delete", { topic: selectedTopic });
  await loadTopics();
}

async function loadTopics() {
  const resp = await apiGet("/api/kvi/topics");
  topics = resp.items || [];
  const sels = ["topic_select", "topic_select_docs", "topic_select_debug"];
  for (const sid of sels) {
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

function topicByName(name) {
  return (topics || []).find(x => x.topic === name) || null;
}

async function onTopicChange(topic) {
  selectedTopic = topic;
  const t = topicByName(topic);
  $("topic_goal").textContent = t && t.goal ? String(t.goal) : "";
  await loadEvidenceSets();
}

function parseJsonl(text) {
  const out = [];
  const lines = String(text || "").split(/\r?\n/);
  for (const ln of lines) {
    const s = ln.trim();
    if (!s) continue;
    try { out.push(JSON.parse(s)); } catch {}
  }
  return out;
}

function toJsonl(records) {
  return (records || []).map(r => {
    try { return JSON.stringify(r); } catch { return ""; }
  }).filter(x => x).join("\n") + "\n";
}

async function compileSimple() {
  const maxSentenceTokens = Number(($("sent_token_budget").value || "128").trim());
  $("compile_log").textContent = "编译中...（sentence-KVBank：逐条 sentence forward 抽取 KV cache 并建 FAISS；可能需要数分钟）\n";
  const useLlmIntent = ($("compile_use_llm_intent").value || "false") === "true";
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/compile_simple`, {
    max_sentence_tokens: Number.isFinite(maxSentenceTokens) ? maxSentenceTokens : 128,
    use_llm_intent: useLlmIntent,
  });
  const ok = !!resp.ok;
  $("compile_log").textContent = (ok ? "✅ OK\n\n" : "❌ FAILED\n\n") + pretty(resp);
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
    opt.textContent = `${s.enabled ? "✓" : " "} ${s.name} (${s.count || 0})`;
    sel.appendChild(opt);
  }
  if (evidenceSets.length > 0) {
    await loadEvidenceSet(evidenceSets[0].name);
  } else {
    currentSet = null;
    $("set_stats").textContent = "暂无 evidence set。请先新建。";
    $("set_raw").value = "";
    $("doc_group_view").textContent = "未加载";
  }
}

async function loadEvidenceSet(name) {
  const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_set/${encodeURIComponent(name)}`);
  currentSet = resp;
  $("set_select").value = name;
  $("set_enabled").value = resp.enabled ? "true" : "false";
  $("set_stats").textContent = `当前：${name}  records=${resp.count || 0}  enabled=${resp.enabled}`;
  $("set_raw").value = toJsonl(resp.records || []);
  $("doc_group_view").textContent = pretty(resp.by_source || {});
}

async function saveCurrentSet() {
  if (!currentSet) throw new Error("未选择 evidence set。");
  const name = currentSet.name;
  const enabled = ($("set_enabled").value || "true") === "true";
  const maxSentenceTokens = Number(($("sent_token_budget").value || "128").trim());
  const records = parseJsonl($("set_raw").value || "");
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_set/save`, {
    name, enabled, records,
    max_sentence_tokens: Number.isFinite(maxSentenceTokens) ? maxSentenceTokens : 128,
  });
  const ss = resp.split_stats || null;
  $("set_stats").textContent = `已保存：${resp.name}  count=${resp.count}  enabled=${resp.enabled}` + (ss ? `  split=${ss.split_records||0}→${ss.generated_records||0}  truncated=${ss.truncated_records||0}` : "");
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
  // Split by delimiter "###" (each claim ends with ###). Also allow plain single claim.
  const parts = raw.includes("###") ? raw.split("###") : [raw];
  const claims = parts.map(x => String(x || "").trim()).filter(x => x);
  if (claims.length === 0) throw new Error("claim * 必填");
  const sourceId = ($("sent_source_id").value || "").trim() || null;
  const doi = ($("sent_doi").value || "").trim() || null;
  const title = ($("sent_title").value || "").trim() || null;
  const author = ($("sent_author").value || "").trim() || null;
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ts = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  return claims.map((claim) => ({
    id: null,
    topic: selectedTopic,
    claim,
    source_id: sourceId,
    source_ref: { doi, title },
    created_at: ts,
    updated_at: ts,
    author,
    tags: [],
  }));
}

async function addSentenceToSet() {
  if (!currentSet) throw new Error("未选择 evidence set。");
  const recs = buildNewSentenceRecords();
  const records = parseJsonl($("set_raw").value || "");
  for (const r of recs) records.push(r);
  $("set_raw").value = toJsonl(records);
  $("sent_claim").value = "";
  await saveCurrentSet();
}

async function loadDocsList() {
  const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/docs`);
  const el = $("docs_list");
  el.innerHTML = "";
  selectedDoc = null;
  $("btn_import_doc_blocks").disabled = true;
  $("docs_list_view").style.display = "block";
  $("doc_detail_view").style.display = "none";
  for (const d of resp.items || []) {
    const card = document.createElement("div");
    card.className = "card";
    const top = document.createElement("div");
    top.className = "top";
    const left = document.createElement("div");
    left.className = "mono";
    left.textContent = d.pdf_name || d.doc_id;
    const badge = document.createElement("span");
    badge.className = "badge " + (d.approved ? "ok" : "warn");
    badge.textContent = d.approved ? "approved" : "not approved";
    top.appendChild(left);
    top.appendChild(badge);
    const meta = document.createElement("div");
    meta.className = "note";
    meta.textContent = `doc_id=${d.doc_id}  year=${d.publication_year || ""}  doi=${d.doi || ""}`;
    const btns = document.createElement("div");
    btns.className = "btns";
    const toggle = document.createElement("button");
    toggle.className = d.approved ? "bad" : "ok";
    toggle.textContent = d.approved ? "取消 approved" : "标记 approved";
    toggle.onclick = async (ev) => {
      ev.stopPropagation();
      await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/doc/${encodeURIComponent(d.doc_id)}/set_approved`, { approved: !d.approved });
      await loadDocsList();
    };
    btns.appendChild(toggle);
    const open = document.createElement("button");
    open.className = "primary";
    open.textContent = "打开";
    open.onclick = async (ev) => {
      ev.stopPropagation();
      selectedDoc = d;
      $("docs_list_view").style.display = "none";
      $("doc_detail_view").style.display = "block";
      await loadDocBlocks(d.doc_id);
    };
    btns.appendChild(open);
    card.appendChild(top);
    card.appendChild(meta);
    card.appendChild(btns);
    el.appendChild(card);
  }
}

async function loadDocBlocks(docId) {
  const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/doc/${encodeURIComponent(docId)}/blocks`);
  const lines = [];
  lines.push(`doc_id=${docId}`);
  lines.push(`blocks=${resp.count}`);
  lines.push("");
  let idx = 0;
  for (const b of resp.items || []) {
    idx += 1;
    lines.push(`[${String(idx).padStart(3,"0")}] (${b.block_type}) ${b.claim}`);
  }
  $("doc_detail").textContent = lines.join("\n");
  $("btn_import_doc_blocks").disabled = !(selectedDoc && selectedDoc.approved);
}

async function importDocBlocks() {
  if (!selectedDoc) throw new Error("请先选择一个 doc。");
  if (!selectedDoc.approved) throw new Error("doc 未 approved，无法导入。");
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/doc/${encodeURIComponent(selectedDoc.doc_id)}/import_to_evidence`, {});
  $("doc_detail").textContent = $("doc_detail").textContent + `\n\n---\n已导入到 evidence set=${resp.evidence_set}：appended=${resp.appended}`;
  await loadEvidenceSets();
}

async function runDebug() {
  const prompt = ($("debug_prompt").value || "").trim();
  if (!prompt) throw new Error("请先输入 prompt。");
  const topK = Number(($("debug_top_k").value || "2").trim());
  const mode = ($("debug_mode").value || "modeA").trim();
  const wAnn = Number(($("route_w_ann").value || "1.0").trim());
  const wIntent = Number(($("route_w_intent").value || "0.6").trim());
  const wQuality = Number(($("route_w_quality").value || "0.2").trim());
  const rerankNoAnn = ($("route_rerank_without_ann").value || "false") === "true";
  const modeAUseLlmIntent = ($("modeA_use_llm_intent").value || "false") === "true";
  const routeLlmIntent = (mode === "modeA") && modeAUseLlmIntent;
  const routeTraceText = ($("route_trace_text").value || "").trim();
  $("out_cli").textContent = "运行中...";
  $("out_modeA").textContent = "运行中...";
  $("out_modeB").textContent = "运行中...";
  $("out_route").textContent = "运行中...";
  $("out_debug_log").textContent = "运行中...";
  $("out_base_llm").textContent = "运行中...";
  $("out_modeA_status").textContent = "";
  $("out_modeB_status").textContent = "";
  $("out_route_status").textContent = "";
  let resp = null;
  if (mode === "modeB") {
    resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/modeB`, {
      prompt,
      top_k: Number.isFinite(topK) ? topK : 8,
      route_w_ann: Number.isFinite(wAnn) ? wAnn : 1.0,
      route_w_intent: Number.isFinite(wIntent) ? wIntent : 0.6,
      route_w_quality: Number.isFinite(wQuality) ? wQuality : 0.2,
      route_rerank_without_ann: rerankNoAnn,
      route_llm_intent_enable: routeLlmIntent,
      route_trace_text: routeTraceText,
    });
  } else if (mode === "route") {
    resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/route`, {
      prompt,
      top_k: Number.isFinite(topK) ? topK : 8,
      route_w_ann: Number.isFinite(wAnn) ? wAnn : 1.0,
      route_w_intent: Number.isFinite(wIntent) ? wIntent : 0.6,
      route_w_quality: Number.isFinite(wQuality) ? wQuality : 0.2,
      route_rerank_without_ann: rerankNoAnn,
      route_llm_intent_enable: routeLlmIntent,
      route_trace_text: routeTraceText,
    });
  } else {
    resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/modeA`, {
      prompt,
      top_k: Number.isFinite(topK) ? topK : 8,
      route_w_ann: Number.isFinite(wAnn) ? wAnn : 1.0,
      route_w_intent: Number.isFinite(wIntent) ? wIntent : 0.6,
      route_w_quality: Number.isFinite(wQuality) ? wQuality : 0.2,
      route_rerank_without_ann: rerankNoAnn,
      route_llm_intent_enable: modeAUseLlmIntent,
      route_trace_text: routeTraceText,
    });
  }
  const r = resp.result || {};
  $("out_cli").textContent = resp.cmd || "(no cmd)";
  // Debug log: always fetch full /route for routing/evidence inspection.
  try {
    const routeResp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/route`, {
      prompt,
      top_k: Number.isFinite(topK) ? topK : 8,
      route_w_ann: Number.isFinite(wAnn) ? wAnn : 1.0,
      route_w_intent: Number.isFinite(wIntent) ? wIntent : 0.6,
      route_w_quality: Number.isFinite(wQuality) ? wQuality : 0.2,
      route_rerank_without_ann: rerankNoAnn,
      route_llm_intent_enable: routeLlmIntent,
      route_trace_text: routeTraceText,
    });
    const fullRoute = routeResp.result || {};
    const debugObj = { route: fullRoute };
    // For Mode A, show LLM intent debug separately to avoid mixing with /route evidence.
    if (mode === "modeA" && r.routing_debug) {
      debugObj.modeA_routing_debug = r.routing_debug || {};
    }
    $("out_debug_log").textContent = pretty(debugObj);
  } catch (e) {
    $("out_debug_log").textContent = String(e && e.message ? e.message : e);
  }
  if (mode === "modeB") {
    const texts = (r.evidence_texts || []).map(t => String(t || "")).filter(x => x);
    $("out_modeB").textContent = texts.length ? texts.join("\n") : "(no evidence_texts)";
    $("out_modeB_status").textContent = r.status ? `status: ${r.status}` : "";
    $("out_modeA").textContent = "";
    $("out_route").textContent = "";
    $("out_base_llm").textContent = "";
  } else if (mode === "route") {
    $("out_route").textContent = pretty(r);
    $("out_route_status").textContent = r.status ? `status: ${r.status}` : "";
    $("out_modeA").textContent = "";
    $("out_modeB").textContent = "";
    $("out_base_llm").textContent = "";
  } else {
    $("out_modeA").textContent = r.diagnosis_result || "";
    $("out_modeA_status").textContent = "status: OK";
    $("out_modeB").textContent = "";
    $("out_route").textContent = "";
    $("out_base_llm").textContent = r.base_llm_result || "";
  }
}

function wire() {
  $("tab_simple").onclick = () => setTab("simple");
  $("tab_docs").onclick = () => setTab("docs");
  $("tab_debug").onclick = () => setTab("debug");

  $("topic_select").onchange = async (e) => { await onTopicChange(e.target.value); };
  $("topic_select_docs").onchange = async (e) => {
    const v = e.target.value;
    $("topic_select").value = v;
    $("topic_select_debug").value = v;
    await onTopicChange(v);
    await loadDocsList();
  };
  $("topic_select_debug").onchange = async (e) => {
    const v = e.target.value;
    $("topic_select").value = v;
    $("topic_select_docs").value = v;
    await onTopicChange(v);
  };

  $("btn_compile").onclick = () => compileSimple().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_create_topic").onclick = () => createTopicFromUi().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_delete_topic").onclick = () => deleteTopicFromUi().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_reload_sets").onclick = () => loadEvidenceSets().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("set_select").onchange = async (e) => { await loadEvidenceSet(e.target.value).catch(err => { $("compile_log").textContent = String(err.message || err); }); };
  $("btn_save_set").onclick = () => saveCurrentSet().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_create_set").onclick = () => createEvidenceSet().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_add_sentence").onclick = () => addSentenceToSet().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_load_docs_topic").onclick = () => loadDocsList().catch(err => { $("doc_detail").textContent = String(err.message || err); });
  $("btn_import_doc_blocks").onclick = () => importDocBlocks().catch(err => { $("doc_detail").textContent = String(err.message || err); });
  $("btn_back_to_docs").onclick = () => {
    selectedDoc = null;
    $("docs_list_view").style.display = "block";
    $("doc_detail_view").style.display = "none";
  };
  $("btn_run_debug").onclick = () => runDebug().catch(err => { $("out_debug").textContent = String(err.message || err); });
  // top_k slider
  const slider = $("debug_topk_slider");
  const show = $("debug_topk_n");
  const topKInput = $("debug_top_k");
  if (slider && show) {
    const sync = () => {
      show.textContent = String(slider.value || "2");
      if (topKInput) topKInput.value = String(slider.value || "2");
    };
    slider.addEventListener("input", sync);
    sync();
  }
}

async function init() {
  wire();
  await loadTopics();
  // keep doc/debug topic selectors consistent with simple selector
  $("topic_select_docs").value = selectedTopic;
  $("topic_select_debug").value = selectedTopic;
}

window.addEventListener("DOMContentLoaded", () => init().catch(err => console.error(err)));

