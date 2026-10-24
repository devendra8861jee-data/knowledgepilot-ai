import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDown, ArrowUpRight, Bot, BookOpen, Check, CheckCircle2,
  ChevronDown, CircleHelp, Clock3, CloudUpload, FileText, Files, Gauge,
  LayoutDashboard, LoaderCircle, MessageSquare, MoreHorizontal, Plus,
  Search, Send, Settings2, ShieldCheck, Sparkles, Trash2, UploadCloud,
  X, Zap, PanelLeftClose, PanelLeftOpen, SlidersHorizontal, RefreshCw
} from "lucide-react";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const navItems = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "documents", label: "Documents", icon: Files },
  { id: "chat", label: "AI Chat", icon: MessageSquare },
  { id: "evaluation", label: "Evaluation", icon: Activity }
];

async function api(path, options = {}) {
  const response = await fetch(API + path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Request failed. Check that the API is running.");
  return data;
}

function initials(name = "") {
  return name.split(/[ _-]/).filter(Boolean).slice(0, 2).map(part => part[0]).join("").toUpperCase() || "KP";
}

function StatCard({ label, value, note, icon: Icon, tone = "blue" }) {
  return <article className="stat-card">
    <div className={"stat-icon " + tone}><Icon size={19} /></div>
    <div className="stat-copy"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
    <ArrowUpRight className="stat-arrow" size={17} />
  </article>;
}

function EmptyState({ icon: Icon = FileText, title, body, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={25} /></div><h3>{title}</h3><p>{body}</p>{action}</div>;
}

function SourceCard({ source }) {
  return <details className="source-card">
    <summary><span className="source-file"><FileText size={16} /><span>{source.filename}</span></span><span className="source-meta">Page {source.page} <ChevronDown size={14} /></span></summary>
    <p>{source.excerpt || "Source excerpt unavailable."}</p>
    <div className="source-foot"><span>Source [{source.index}]</span><span>Relevance {(Number(source.score || 0) * 100).toFixed(1)}%</span></div>
  </details>;
}

function ChatPage({ documents, onRefresh }) {
  const [messages, setMessages] = useState([{ role: "assistant", content: "Hi! I'm KnowledgePilot. Ask me a question about your indexed PDFs, and I'll answer using evidence with page-level citations.", citations: [] }]);
  const [question, setQuestion] = useState("");
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, busy]);
  const suggestions = ["Summarize the key points", "What are the important numbers?", "What does the document say about policies?"];

  async function sendQuestion(text = question) {
    const prompt = text.trim();
    if (!prompt || busy) return;
    setQuestion("");
    setError("");
    setMessages(prev => [...prev, { role: "user", content: prompt }]);
    setBusy(true);
    try {
      const result = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: prompt, doc_ids: selectedDocs.length ? selectedDocs : null })
      });
      setMessages(prev => [...prev, { role: "assistant", content: result.answer || "No answer was returned.", citations: result.citations || [], mode: result.mode, latency: result.total_latency_s, warning: result.warnings?.join(" ") }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      onRefresh();
    }
  }

  return <div className="chat-layout">
    <section className="chat-main panel">
      <div className="chat-topline"><div className="chat-avatar"><Bot size={19} /></div><div><strong>KnowledgePilot Assistant</strong><span><i className="status-dot" /> Answers grounded in your documents</span></div><button className="icon-button clear-chat" title="Clear conversation" onClick={() => { setMessages([]); setError(""); }}><RefreshCw size={16} /></button></div>
      <div className="chat-thread">
        {messages.length === 0 && <div className="chat-empty"><div className="hero-orb"><Sparkles size={26} /></div><h2>What would you like to know?</h2><p>Ask a question. I'll find relevant passages and show where the answer came from.</p></div>}
        {messages.map((message, i) => <div className={"message-row " + message.role} key={i}>
          {message.role === "assistant" && <div className="message-avatar"><Sparkles size={15} /></div>}
          <div className="message-content"><div className="message-bubble">{message.content}</div>
            {message.warning && <p className="inline-warning">{message.warning}</p>}
            {message.citations?.length > 0 && <div className="sources-list"><div className="sources-heading"><ShieldCheck size={15} /> Verified sources <span>{message.citations.length}</span></div>{message.citations.map((source, j) => <SourceCard source={source} key={source.chunk_id || j} />)}</div>}
            {message.mode && <div className="message-foot"><span>{message.mode === "llm" ? "AI generated" : message.mode === "fallback" ? "Evidence check" : "Extractive answer"}</span>{message.latency != null && <span><Clock3 size={12} /> {Number(message.latency).toFixed(2)}s</span>}</div>}
          </div>
          {message.role === "user" && <div className="user-avatar">You</div>}
        </div>)}
        {busy && <div className="message-row assistant"><div className="message-avatar"><Sparkles size={15} /></div><div className="message-bubble loading-bubble"><span className="dot-pulse" /> Searching your documents…</div></div>}
        {error && <div className="error-banner"><X size={16} /> {error}</div>}
        <div ref={endRef} />
      </div>
      <div className="chat-composer-wrap">
        {messages.length < 3 && <div className="suggestions">{suggestions.map(s => <button key={s} onClick={() => sendQuestion(s)}>{s}</button>)}</div>}
        <form className="chat-composer" onSubmit={e => { e.preventDefault(); sendQuestion(); }}><textarea rows="1" value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); } }} placeholder="Ask anything about your PDFs…" /><button type="submit" disabled={!question.trim() || busy || !documents.length} aria-label="Send question"><Send size={17} /></button></form>
        <div className="composer-note"><ShieldCheck size={13} /> Responses use indexed document evidence. Always verify important details.</div>
      </div>
    </section>
    <aside className="chat-side">
      <div className="panel side-panel"><div className="section-title"><div><h3>Knowledge sources</h3><p>Choose which files to search</p></div><Files size={18} /></div>
        {!documents.length ? <p className="muted-small">Upload and index a PDF to start chatting.</p> : <div className="doc-selector">{documents.map(doc => <label key={doc.doc_id} className="doc-check"><input type="checkbox" checked={selectedDocs.includes(doc.doc_id)} onChange={e => setSelectedDocs(prev => e.target.checked ? [...prev, doc.doc_id] : prev.filter(id => id !== doc.doc_id))} /><span className="pdf-mini"><FileText size={15} /></span><span className="doc-check-copy"><b>{doc.filename}</b><small>{doc.pages} pages · {doc.chunks} chunks</small></span></label>)}</div>}
        <div className="side-tip"><ShieldCheck size={16} /><span><b>Evidence-first answers</b><br />If a document doesn't contain enough evidence, the assistant should say so.</span></div>
      </div>
      <div className="panel side-panel"><div className="section-title"><div><h3>Quick tips</h3><p>Get better answers</p></div><CircleHelp size={18} /></div><ul className="tips-list"><li>Ask one clear question at a time.</li><li>Include exact names, numbers, or model IDs.</li><li>Open a source card to inspect the supporting excerpt.</li></ul></div>
    </aside>
  </div>;
}

function DocumentsPage({ documents, chunkCount, refresh }) {
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const fileInput = useRef(null);
  const filtered = documents.filter(doc => doc.filename.toLowerCase().includes(query.toLowerCase()));

  async function upload() {
    if (!files.length) return;
    setBusy(true); setError(""); setMessage("");
    const form = new FormData();
    files.forEach(file => form.append("files", file));
    try {
      const result = await api("/api/documents", { method: "POST", body: form });
      const indexed = (result.reports || []).filter(r => r.status === "indexed").length;
      const failed = (result.reports || []).filter(r => r.status === "failed").length;
      setMessage(indexed + " PDF(s) indexed successfully" + (failed ? "; " + failed + " failed" : "") + ".");
      setFiles([]);
      if (fileInput.current) fileInput.current.value = "";
      await refresh();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function remove(doc) {
    if (!window.confirm("Remove " + doc.filename + " from the index?")) return;
    try { await api("/api/documents/" + encodeURIComponent(doc.doc_id), { method: "DELETE" }); await refresh(); }
    catch (err) { setError(err.message); }
  }

  return <div className="page-stack">
    <div className="page-heading"><div><div className="eyebrow">YOUR KNOWLEDGE BASE</div><h1>Documents</h1><p>Upload PDFs, index their contents, and build your searchable knowledge base.</p></div><div className="heading-badge"><Files size={18} /> {documents.length} indexed</div></div>
    <section className="upload-panel panel">
      <div className="upload-illustration"><UploadCloud size={28} /></div><h2>Upload your documents</h2><p>Choose text-based PDF files. Text is extracted page by page to preserve citations.</p>
      <button className="button-primary" onClick={() => fileInput.current?.click()}><Plus size={17} /> Choose PDF files</button>
      <input ref={fileInput} className="visually-hidden" type="file" accept="application/pdf,.pdf" multiple onChange={e => setFiles(Array.from(e.target.files || []))} />
      <div className="upload-drop-hint">PDF files only · multiple files supported</div>
      {files.length > 0 && <div className="selected-files">{files.map(f => <div className="selected-file" key={f.name}><FileText size={16} /><span>{f.name}</span><small>{(f.size / 1024 / 1024).toFixed(2)} MB</small><button onClick={() => setFiles(prev => prev.filter(item => item !== f))}><X size={14} /></button></div>)}
        <button className="button-primary upload-submit" disabled={busy} onClick={upload}>{busy ? <LoaderCircle className="spin" size={16} /> : <CloudUpload size={16} />}{busy ? "Indexing PDFs…" : "Index selected PDFs"}</button></div>}
      {message && <div className="success-banner"><CheckCircle2 size={16} />{message}</div>}{error && <div className="error-banner"><X size={16} />{error}</div>}
    </section>
    <div className="section-title document-list-heading"><div><h2>Indexed library</h2><p>Your PDFs and their searchable content</p></div><div className="search-field"><Search size={16} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search files" /></div></div>
    <div className="panel document-table-panel">
      <div className="table-summary"><span>{documents.length} documents</span><span>{chunkCount} indexed passages</span></div>
      {!filtered.length ? <EmptyState icon={Files} title={documents.length ? "No matching files" : "Your library is empty"} body={documents.length ? "Try another search term." : "Upload your first PDF above to start asking questions."} /> :
        <div className="document-list">{filtered.map(doc => <div className="document-row" key={doc.doc_id}><div className="file-icon"><FileText size={20} /></div><div className="document-info"><strong>{doc.filename}</strong><span>Added to searchable knowledge base</span></div><div className="doc-stat"><b>{doc.pages}</b><span>pages</span></div><div className="doc-stat"><b>{doc.chunks}</b><span>chunks</span></div><span className="indexed-pill"><Check size={13} /> Indexed</span><button className="icon-button danger-hover" title="Remove document" onClick={() => remove(doc)}><Trash2 size={16} /></button></div>)}</div>}
    </div>
  </div>;
}

function EvaluationPage({ evaluation, refresh, settings }) {
  const [busy, setBusy] = useState(false);
  const [useLlm, setUseLlm] = useState(false);
  const [judge, setJudge] = useState(false);
  const [error, setError] = useState("");
  const report = evaluation?.report;
  const summary = report?.summary;
  const gt = summary?.ground_truth_retrieval_metrics || {};
  const abst = summary?.abstention_behaviour || {};
  const ops = summary?.operational_metrics || {};
  const k = report?.config?.k || settings?.top_k || 4;
  async function run() {
    setBusy(true); setError("");
    try { await api("/api/evaluation?use_llm=" + useLlm + "&judge=" + judge, { method: "POST" }); await refresh(); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  const metrics = [
    ["Recall@" + k, gt["recall@" + k], "Relevant target pages found", "blue"],
    ["Hit rate@" + k, gt["hit_rate@" + k], "Questions with a target hit", "green"],
    ["Precision@" + k, gt["precision@" + k], "Relevant results in top-k", "violet"],
    ["Mean reciprocal rank", gt.mrr, "Correct result ranking", "orange"]
  ];
  const items = report?.items || [];
  return <div className="page-stack">
    <div className="page-heading"><div><div className="eyebrow">QUALITY & RELIABILITY</div><h1>RAG evaluation</h1><p>Measure retrieval quality, abstention behavior, and operational performance.</p></div><button className="button-primary" onClick={run} disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />}{busy ? "Running evaluation…" : "Run evaluation"}</button></div>
    <div className="evaluation-config panel"><div className="config-icon"><SlidersHorizontal size={19} /></div><div className="config-copy"><strong>Evaluation configuration</strong><span>{report?.config?.embedder || settings?.embedding_model || "Current embedding model"} · chunk {report?.config?.chunk_size || settings?.chunk_size || 800} / overlap {report?.config?.chunk_overlap || settings?.chunk_overlap || 120} · threshold {report?.config?.min_relevance ?? settings?.min_relevance ?? 0.1}</span></div><span className="sample-tag">15 sample questions</span></div>
    <div className="eval-options panel"><div><strong>Answer quality checks</strong><p>LLM-based checks make API calls and may incur provider costs.</p></div><label className="switch-row"><input type="checkbox" checked={useLlm} onChange={e => setUseLlm(e.target.checked)} disabled={!settings?.llm_configured || settings?.llm_provider === "none"} /><span>Generate LLM answers</span></label><label className="switch-row"><input type="checkbox" checked={judge} onChange={e => setJudge(e.target.checked)} disabled={!settings?.llm_configured || settings?.llm_provider === "none"} /><span>LLM judge metrics</span></label></div>
    {error && <div className="error-banner"><X size={16} />{error}</div>}
    {!summary ? <div className="panel"><EmptyState icon={Gauge} title="No evaluation results yet" body="Run the sample evaluation to see retrieval, abstention, and latency metrics." action={<button className="button-primary" onClick={run} disabled={busy}>Run first evaluation</button>} /></div> :
      <>
        <div className="section-title"><div><h2>Retrieval quality</h2><p>Ground-truth page matching · no LLM required</p></div><span className="last-run"><CheckCircle2 size={14} /> Results loaded</span></div>
        <div className="stats-grid">{metrics.map(([label, value, note, tone]) => <StatCard key={label} label={label} value={value == null ? "n/a" : (Number(value) * 100).toFixed(1) + "%"} note={note} icon={Gauge} tone={tone} />)}</div>
        <div className="two-col">
          <section className="panel metric-panel"><div className="section-title"><div><h3>Abstention behavior</h3><p>Does the assistant know when evidence is missing?</p></div><ShieldCheck size={18} /></div>
            {[["Correct abstention", abst.correct_abstention_rate_unanswerable], ["False abstention", abst.false_abstention_rate_answerable], ["Keyword coverage", summary.proxy_metrics?.context_keyword_coverage]].map(([label, val]) => <div className="bar-metric" key={label}><div><span>{label}</span><b>{val == null ? "n/a" : (Number(val) * 100).toFixed(1) + "%"}</b></div><div className="meter"><i style={{ width: Math.max(0, Math.min(100, Number(val || 0) * 100)) + "%" }} /></div></div>)}
          </section>
          <section className="panel metric-panel"><div className="section-title"><div><h3>Operational metrics</h3><p>Latency and reliability</p></div><Zap size={18} /></div>
            <div className="ops-grid"><div><span>Mean latency</span><b>{ops.mean_latency_s == null ? "n/a" : Number(ops.mean_latency_s).toFixed(3) + "s"}</b></div><div><span>P95 latency</span><b>{ops.p95_latency_s == null ? "n/a" : Number(ops.p95_latency_s).toFixed(3) + "s"}</b></div><div><span>Error rate</span><b>{ops.error_rate == null ? "n/a" : (Number(ops.error_rate) * 100).toFixed(1) + "%"}</b></div><div><span>Estimated cost</span><b>{ops.estimated_cost_usd == null ? "n/a" : "$" + Number(ops.estimated_cost_usd).toFixed(4)}</b></div></div>
            <div className="judge-status"><Activity size={16} /><div><b>LLM-judged metrics</b><span>Faithfulness: {summary.llm_judged_metrics?.faithfulness ?? "n/a"} · Relevance: {summary.llm_judged_metrics?.answer_relevance ?? "n/a"}</span></div></div>
          </section>
        </div>
        <section className="panel results-panel"><div className="section-title"><div><h3>Per-question results</h3><p>{items.length} labelled test cases</p></div><span className="sample-tag">Human review recommended</span></div>
          <div className="table-scroll"><table className="results-table"><thead><tr><th>ID</th><th>Question</th><th>Answerable</th><th>Abstained</th><th>Recall</th><th>Top match</th><th>Latency</th></tr></thead><tbody>{items.map(item => <tr key={item.id}><td><b>{item.id}</b></td><td className="question-cell">{item.question}</td><td><span className={item.answerable ? "status-good" : "status-neutral"}>{item.answerable ? "Yes" : "No"}</span></td><td><span className={item.abstained ? "status-good" : "status-neutral"}>{item.abstained ? "Yes" : "No"}</span></td><td>{item.recall == null ? "—" : Number(item.recall).toFixed(2)}</td><td>{item.retrieved?.[0] ? item.retrieved[0].filename + " · p" + item.retrieved[0].page : "—"}</td><td>{Number(item.total_latency_s || 0).toFixed(2)}s</td></tr>)}</tbody></table></div>
        </section>
      </>}
  </div>;
}

function SettingsPage({ settings, onSave }) {
  const [form, setForm] = useState(settings || {});
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  useEffect(() => setForm(settings || {}), [settings]);
  function change(key, value) { setForm(prev => ({ ...prev, [key]: value })); }
  async function save(e) {
    e.preventDefault(); setBusy(true); setNotice("");
    try {
      const payload = { ...form, chunk_size: Number(form.chunk_size), chunk_overlap: Number(form.chunk_overlap), top_k: Number(form.top_k), min_relevance: Number(form.min_relevance) };
      const updated = await onSave(payload);
      setForm(updated); setNotice("Settings applied for this API session.");
    } catch (err) { setNotice(err.message); }
    finally { setBusy(false); }
  }
  return <div className="page-stack">
    <div className="page-heading"><div><div className="eyebrow">PERSONALIZE YOUR WORKSPACE</div><h1>Settings</h1><p>Configure retrieval behavior and your compatible LLM provider.</p></div></div>
    <form className="settings-grid" onSubmit={save}>
      <section className="panel settings-panel"><div className="settings-heading"><div className="settings-icon"><SlidersHorizontal size={18} /></div><div><h2>Retrieval & chunking</h2><p>Controls how PDF text is indexed and searched.</p></div></div>
        <label className="field-label">Embedding backend<select value={form.embedding_backend || "sentence-transformers"} onChange={e => change("embedding_backend", e.target.value)}><option value="sentence-transformers">Sentence Transformers (semantic)</option><option value="hashing">Hashing (offline / demo)</option></select></label>
        <div className="form-two"><label className="field-label">Chunk size <input type="number" min="200" max="2000" step="50" value={form.chunk_size ?? 800} onChange={e => change("chunk_size", e.target.value)} /></label><label className="field-label">Chunk overlap <input type="number" min="0" max="500" step="10" value={form.chunk_overlap ?? 120} onChange={e => change("chunk_overlap", e.target.value)} /></label></div>
        <div className="form-two"><label className="field-label">Top-k passages <input type="number" min="1" max="10" value={form.top_k ?? 4} onChange={e => change("top_k", e.target.value)} /></label><label className="field-label">Minimum relevance <input type="number" min="0" max="1" step="0.01" value={form.min_relevance ?? 0.1} onChange={e => change("min_relevance", e.target.value)} /></label></div>
        <div className="settings-note"><CircleHelp size={16} /> Chunk size and overlap apply when indexing new PDFs. Re-index documents after changing them.</div>
      </section>
      <section className="panel settings-panel"><div className="settings-heading"><div className="settings-icon violet"><Bot size={18} /></div><div><h2>Answer generation</h2><p>Use an OpenAI-compatible endpoint such as Groq.</p></div></div>
        <label className="field-label">LLM provider<select value={form.llm_provider || "none"} onChange={e => change("llm_provider", e.target.value)}><option value="none">Retrieval only (no LLM)</option><option value="openai_compatible">OpenAI-compatible API</option></select></label>
        <label className="field-label">Model ID<input value={form.llm_model || ""} onChange={e => change("llm_model", e.target.value)} placeholder="openai/gpt-oss-20b" /></label>
        <label className="field-label">Base URL<input value={form.llm_base_url || ""} onChange={e => change("llm_base_url", e.target.value)} placeholder="https://api.groq.com/openai/v1" /></label>
        <div className={form.llm_configured ? "api-key-status connected" : "api-key-status"}><span className="status-dot" />{form.llm_configured ? "API key is configured on the backend" : "API key missing — add LLM_API_KEY to the backend .env"}</div>
        <div className="settings-note"><ShieldCheck size={16} /> Your API key is never sent to or displayed in the browser.</div>
      </section>
      <div className="settings-save-row">{notice && <span className={notice.includes("Error") ? "error-text" : "success-text"}>{notice}</span>}<button className="button-primary" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}{busy ? "Saving…" : "Apply settings"}</button></div>
    </form>
  </div>;
}

export default function App() {
  const [active, setActive] = useState("overview");
  const [collapsed, setCollapsed] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [chunkCount, setChunkCount] = useState(0);
  const [settings, setSettings] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const [loadError, setLoadError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [globalSearch, setGlobalSearch] = useState("");

  async function refreshAll() {
    setRefreshing(true);
    try {
      const [docs, config, evalData] = await Promise.all([api("/api/documents"), api("/api/settings"), api("/api/evaluation")]);
      setDocuments(docs.documents || []);
      setChunkCount(docs.chunk_count || 0);
      setSettings(config);
      setEvaluation(evalData);
      setApiStatus("connected");
      setLoadError("");
    } catch (err) {
      setApiStatus("offline");
      setLoadError(err.message);
    } finally { setRefreshing(false); }
  }
  useEffect(() => { refreshAll(); }, []);

  const pageTitle = useMemo(() => ({
    overview: "Overview", documents: "Documents", chat: "AI Chat", evaluation: "Evaluation", settings: "Settings"
  }[active] || "Overview"), [active]);

  async function saveSettings(payload) {
    const updated = await api("/api/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    setSettings(updated);
    return updated;
  }

  return <div className={"app-shell " + (collapsed ? "sidebar-collapsed" : "")}>
    <aside className="sidebar">
      <div className="brand-row"><div className="brand-mark"><Sparkles size={21} /></div>{!collapsed && <div className="brand-text"><strong>KnowledgePilot</strong><span>AI KNOWLEDGE WORKSPACE</span></div>}<button className="collapse-button" onClick={() => setCollapsed(!collapsed)} title="Toggle sidebar">{collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}</button></div>
      {!collapsed && <div className="workspace-label">WORKSPACE</div>}
      <nav className="main-nav">{navItems.map(item => { const Icon = item.icon; return <button key={item.id} className={"nav-item " + (active === item.id ? "active" : "")} onClick={() => setActive(item.id)} title={item.label}><Icon size={19} /><span>{!collapsed && item.label}</span>{item.id === "documents" && documents.length > 0 && !collapsed && <em>{documents.length}</em>}</button>; })}</nav>
      {!collapsed && <div className="sidebar-bottom"><div className="sidebar-tip"><div className="tip-icon"><ShieldCheck size={17} /></div><b>Evidence, not guesses.</b><p>Answers are grounded in your documents with verifiable page citations.</p></div><button className={"nav-item " + (active === "settings" ? "active" : "")} onClick={() => setActive("settings")}><Settings2 size={19} /><span>Settings</span></button><div className="profile-row"><div className="profile-avatar">KP</div><div><b>KnowledgePilot AI</b><span>Local workspace</span></div><MoreHorizontal size={17} /></div></div>}
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{pageTitle}</strong></div><div className="topbar-actions"><div className="global-search"><Search size={16} /><input value={globalSearch} onChange={e => { setGlobalSearch(e.target.value); if (e.target.value) setActive("documents"); }} placeholder="Search your workspace" /></div><button className="icon-button" title="Refresh data" onClick={refreshAll}>{refreshing ? <LoaderCircle size={17} className="spin" /> : <RefreshCw size={16} />}</button><span className={"connection-pill " + (apiStatus === "connected" ? "online" : apiStatus === "offline" ? "offline" : "")}><i />{apiStatus === "connected" ? "API connected" : apiStatus === "offline" ? "API offline" : "Connecting"}</span></div></header>
      <div className="content-area">
        {loadError && <div className="error-banner top-error"><X size={16} /><div><b>Can't connect to the KnowledgePilot API.</b><br />{loadError}<br /><small>Start the backend with: <code>uvicorn api:app --reload</code></small></div><button onClick={refreshAll}>Retry</button></div>}
        {active === "overview" && <div className="page-stack">
          <section className="welcome-banner"><div className="welcome-content"><div className="eyebrow light">YOUR DOCUMENTS, UNDERSTOOD</div><h1>Good knowledge starts<br />with the right questions.</h1><p>Search across your PDFs, get grounded answers, and trace every claim back to its source.</p><div className="welcome-actions"><button className="button-light" onClick={() => setActive("chat")}><MessageSquare size={16} /> Ask a question <ArrowUpRight size={15} /></button><button className="button-ghost" onClick={() => setActive("documents")}><Plus size={16} /> Add documents</button></div></div><div className="welcome-art"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="art-document"><FileText size={42} /><span /><span /><span /></div><div className="art-spark spark-one"><Sparkles size={17} /></div><div className="art-spark spark-two"><Zap size={15} /></div></div></section>
          <div className="section-title overview-title"><div><h2>Your workspace at a glance</h2><p>Monitor your knowledge base and system activity.</p></div><span className="live-label"><i /> LIVE WORKSPACE</span></div>
          <div className="stats-grid"><StatCard label="Indexed documents" value={documents.length} note="PDFs in your library" icon={Files} tone="blue" /><StatCard label="Searchable passages" value={chunkCount.toLocaleString()} note="Chunks available to retrieve" icon={BookOpen} tone="violet" /><StatCard label="Embedding model" value={settings?.embedding_backend === "hashing" ? "Hashing" : "Semantic"} note={settings?.embedding_model || "Sentence Transformers"} icon={Zap} tone="green" /><StatCard label="Answer generation" value={settings?.llm_provider === "none" ? "Retrieval only" : "LLM enabled"} note={settings?.llm_configured ? "Provider configured" : "API key needs setup"} icon={Bot} tone="orange" /></div>
          <div className="overview-grid"><section className="panel recent-panel"><div className="section-title"><div><h3>Recent knowledge sources</h3><p>Your indexed PDF library</p></div><button className="text-button" onClick={() => setActive("documents")}>View all <ArrowUpRight size={14} /></button></div>{documents.length ? <div className="recent-list">{documents.slice(0, 4).map(doc => <div className="recent-row" key={doc.doc_id}><div className="file-icon"><FileText size={19} /></div><div className="recent-doc-name"><b>{doc.filename}</b><span>{doc.pages} pages · {doc.chunks} chunks</span></div><span className="indexed-pill"><Check size={13} /> Ready</span></div>)}</div> : <EmptyState icon={Files} title="Start your knowledge base" body="Upload a PDF to make its content searchable and citeable." action={<button className="button-primary" onClick={() => setActive("documents")}><Plus size={16} /> Upload your first PDF</button>} />}</section>
            <section className="panel quick-panel"><div className="section-title"><div><h3>Quick actions</h3><p>Jump back into your workflow</p></div><Zap size={18} /></div><button className="quick-action" onClick={() => setActive("chat")}><div className="quick-icon blue"><MessageSquare size={18} /></div><div><b>Ask your documents</b><span>Get cited answers in seconds</span></div><ArrowUpRight size={16} /></button><button className="quick-action" onClick={() => setActive("documents")}><div className="quick-icon violet"><CloudUpload size={18} /></div><div><b>Upload a PDF</b><span>Expand your knowledge base</span></div><ArrowUpRight size={16} /></button><button className="quick-action" onClick={() => setActive("evaluation")}><div className="quick-icon green"><Activity size={18} /></div><div><b>Evaluate RAG quality</b><span>Review retrieval and reliability</span></div><ArrowUpRight size={16} /></button></section></div>
          <div className="footer-note"><ShieldCheck size={14} /> Your local-first document assistant. Answers are grounded in retrieved passages, with citations to verify.</div>
        </div>}
        {active === "documents" && <DocumentsPage documents={documents.filter(d => !globalSearch || d.filename.toLowerCase().includes(globalSearch.toLowerCase()))} chunkCount={chunkCount} refresh={refreshAll} />}
        {active === "chat" && <div className="page-stack"><div className="page-heading compact-heading"><div><div className="eyebrow">ASK YOUR KNOWLEDGE BASE</div><h1>AI Chat</h1><p>Answers grounded in your PDFs, with citations you can verify.</p></div><span className="heading-badge"><i className="status-dot" /> {documents.length} sources available</span></div><ChatPage documents={documents} onRefresh={refreshAll} /></div>}
        {active === "evaluation" && <EvaluationPage evaluation={evaluation} refresh={refreshAll} settings={settings} />}
        {active === "settings" && <SettingsPage settings={settings} onSave={saveSettings} />}
      </div>
    </main>
  </div>;
}
