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
  await loadEvidenceTxt();
}

async function loadEvidenceTxt() {
  if (!selectedTopic) return;
  const resp = await apiGet(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_txt`);
  $("evidence_path").value = resp.path || "";
  $("evidence_editor").value = resp.content || "";
}

async function saveEvidenceTxt() {
  const content = $("evidence_editor").value || "";
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/evidence_txt`, { content });
  $("compile_log").textContent = `已保存：${resp.path}\n`;
}

async function compileSimple() {
  $("compile_log").textContent = "编译中...（该过程会跑 tokenizer 分块、pattern 侧车、KVBank 构建，可能需要数分钟）\n";
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/compile_simple`, {});
  $("compile_log").textContent = pretty(resp);
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
  $("doc_detail").textContent = $("doc_detail").textContent + "\n\n---\n已导入到 Evidence.txt：appended=" + resp.appended;
  await loadEvidenceTxt();
}

async function runDebug() {
  const prompt = ($("debug_prompt").value || "").trim();
  if (!prompt) throw new Error("请先输入 prompt。");
  const topK = Number(($("debug_top_k").value || "8").trim());
  const showBaseline = ($("debug_show_baseline").value || "true") === "true";
  $("out_base").textContent = "运行中...";
  $("out_injected").textContent = "运行中...";
  $("out_debug").textContent = "运行中...";
  const resp = await apiPost(`/api/kvi/topic/${encodeURIComponent(selectedTopic)}/run_simple`, {
    prompt,
    top_k: Number.isFinite(topK) ? topK : 8,
    show_baseline: showBaseline,
  });
  const r = resp.result || {};
  $("out_base").textContent = r.base_answer || "(baseline disabled or empty)";
  $("out_injected").textContent = r.injected_answer || "";
  $("out_debug").textContent = pretty({ steps: r.steps, semantic_type_router: r.semantic_type_router });
}

function wire() {
  $("tab_simple").onclick = () => setTab("simple");
  $("tab_docs").onclick = () => setTab("docs");
  $("tab_debug").onclick = () => setTab("debug");

  $("topic_select").onchange = async (e) => { await onTopicChange(e.target.value); };
  $("topic_select_docs").onchange = async (e) => { selectedTopic = e.target.value; await loadDocsList(); };
  $("topic_select_debug").onchange = async (e) => { selectedTopic = e.target.value; };

  $("btn_save_evidence").onclick = () => saveEvidenceTxt().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_compile").onclick = () => compileSimple().catch(err => { $("compile_log").textContent = String(err.message || err); });
  $("btn_load_docs_topic").onclick = () => loadDocsList().catch(err => { $("doc_detail").textContent = String(err.message || err); });
  $("btn_import_doc_blocks").onclick = () => importDocBlocks().catch(err => { $("doc_detail").textContent = String(err.message || err); });
  $("btn_back_to_docs").onclick = () => {
    selectedDoc = null;
    $("docs_list_view").style.display = "block";
    $("doc_detail_view").style.display = "none";
  };
  $("btn_run_debug").onclick = () => runDebug().catch(err => { $("out_debug").textContent = String(err.message || err); });
}

async function init() {
  wire();
  await loadTopics();
}

window.addEventListener("DOMContentLoaded", () => init().catch(err => console.error(err)));

