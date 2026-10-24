import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDownToLine, ArrowRight, BookOpen, Bot, Check, ChevronDown,
  CircleHelp, Clock3, CloudUpload, Command, FileCheck2, FileText, Gauge,
  Github, LayoutDashboard, LoaderCircle, MessageSquare, MoreHorizontal,
  PanelLeftClose, Plus, Search, Send, Settings2, ShieldCheck, Sparkles,
  Trash2, Upload, X, Zap,
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const nav = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "chat", label: "AI workspace", icon: MessageSquare },
  { id: "evaluation", label: "Evaluation", icon: Gauge },
];
const prompts = [
  "Summarize the key points in my documents",
  "What are the important dates and numbers?",
  "Compare the main ideas across documents",
];

async function request(path, options = {}) {
  const response = await fetch(API + path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Request failed. Is the API running?");
  return data;
}
function fmtTime(seconds) {
  if (seconds == null) return "—";
  return seconds < 1 ? Math.round(seconds * 1000) + " ms" : seconds.toFixed(2) + " s";
}
function shortId(value = "") { return value.slice(0, 8); }

export default function App() {
  const [page, setPage] = useState("overview");
  const [docs, setDocs] = useState([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [settingsDraft, setSettingsDraft] = useState(null);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);

  async function refresh() {
    try {
      const [h, d, c, e] = await Promise.all([
        request("/api/health"), request("/api/documents"), request("/api/config"), request("/api/evaluation"),
      ]);
      setHealth(h.status === "ok");
      setDocs(d.documents || []);
      setTotalChunks(d.total_chunks || 0);
      setConfig(c);
      setSettingsDraft(c);
      setEvaluation(e);
      setError("");
    } catch (err) {
      setHealth(false);
      setError(err.message);
    }
  }
  useEffect(() => { refresh(); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, busy]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 3000);
    return () => clearTimeout(timer);
  }, [toast]);

  const filteredDocs = useMemo(() => docs.filter(d => d.filename.toLowerCase().includes(search.toLowerCase())), [docs, search]);
  const currentTitle = nav.find(item => item.id === page)?.label || "Settings";

  async function uploadFiles(files) {
    const list = Array.from(files || []);
    if (!list.length) return;
    const invalid = list.find(f => !f.name.toLowerCase().endsWith(".pdf"));
    if (invalid) { setError("Please select PDF files only."); return; }
    const body = new FormData();
    list.forEach(file => body.append("files", file));
    setUploadBusy(true); setError("");
    try {
      const data = await request("/api/documents", { method: "POST", body });
      await refresh();
      const good = (data.reports || []).filter(r => r.status === "indexed").length;
      setToast(good ? `${good} PDF${good === 1 ? "" : "s"} indexed successfully` : "Upload finished — check the results");
      setPage("documents");
      const failures = (data.reports || []).filter(r => r.status === "failed");
      if (failures.length) setError(failures.map(r => `${r.filename}: ${r.message}`).join(" · "));
    } catch (err) { setError(err.message); }
    finally { setUploadBusy(false); if (fileRef.current) fileRef.current.value = ""; }
  }

  async function deleteDoc(doc) {
    if (!window.confirm(`Remove "${doc.filename}" from the index?`)) return;
    try {
      await request("/api/documents/" + encodeURIComponent(doc.doc_id), { method: "DELETE" });
      await refresh(); setToast("Document removed from your index");
    } catch (err) { setError(err.message); }
  }

  async function saveSettings(event) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const updated = await request("/api/config", {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settingsDraft),
      });
      setConfig(updated); setSettingsDraft(updated); setToast("Settings applied to this API session");
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function sendQuestion(text = question) {
    const value = text.trim();
    if (!value || busy) return;
    if (!docs.length) { setPage("documents"); setError("Upload a PDF first, then ask a question."); return; }
    const userMessage = { role: "user", content: value, id: crypto.randomUUID() };
    setMessages(prev => [...prev, userMessage]);
    setQuestion(""); setPage("chat"); setBusy(true); setError("");
    try {
      const result = await request("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, doc_ids: selectedDocs.length ? selectedDocs : null }),
      });
      setMessages(prev => [...prev, { role: "assistant", content: result.answer, result, id: crypto.randomUUID() }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: "assistant", content: err.message, failed: true, id: crypto.randomUUID() }]);
    } finally { setBusy(false); }
  }

  const changeDraft = (key, value) => setSettingsDraft(prev => ({ ...prev, [key]: value }));
  const openChat = (prompt) => { setQuestion(prompt); setPage("chat"); if (prompt) setTimeout(() => document.querySelector(".chat-input")?.focus(), 50); };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Sparkles size={20} strokeWidth={2.2} /></div>
          <div className="brand-copy"><strong>KnowledgePilot</strong><span>AI DOCUMENT INTELLIGENCE</span></div>
          <button className="icon-button sidebar-collapse" title="Collapse sidebar"><PanelLeftClose size={17} /></button>
        </div>
        <button className="workspace-select"><span className="workspace-avatar">KP</span><span className="workspace-text"><b>Personal workspace</b><small>Free workspace</small></span><ChevronDown size={15} /></button>
        <div className="nav-label">WORKSPACE</div>
        <nav className="main-nav">
          {nav.map(item => <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => setPage(item.id)}>
            <item.icon size={18} strokeWidth={1.8} /><span>{item.label}</span>
            {item.id === "documents" && docs.length > 0 && <span className="nav-count">{docs.length}</span>}
          </button>)}
        </nav>
        <div className="sidebar-divider" />
        <div className="nav-label row-between">YOUR LIBRARY <button className="tiny-icon" onClick={() => setPage("documents")} title="Add documents"><Plus size={14} /></button></div>
        <div className="sidebar-docs">
          {docs.slice(0, 5).map(doc => <button key={doc.doc_id} className="sidebar-doc" onClick={() => setPage("documents")}><span className="pdf-icon"><FileText size={14} /></span><span title={doc.filename}>{doc.filename}</span></button>)}
          {!docs.length && <div className="sidebar-empty">Your PDFs will appear here.</div>}
          {docs.length > 5 && <button className="view-all" onClick={() => setPage("documents")}>View all {docs.length} documents <ArrowRight size={13} /></button>}
        </div>
        <div className="sidebar-spacer" />
        <div className="plan-card"><div className="plan-top"><span className="plan-icon"><Zap size={15} /></span><span>Workspace overview</span></div><p>{docs.length} documents · {totalChunks} searchable passages</p><div className="usage-track"><span style={{ width: docs.length ? "68%" : "4%" }} /></div><small>Local document index</small></div>
        <button className={`connection-status ${health ? "online" : "offline"}`} onClick={refresh}><span className="status-dot" />{health ? "API connected" : "API disconnected"}<span className="status-refresh">↻</span></button>
        <button className={`nav-item settings-link ${page === "settings" ? "active" : ""}`} onClick={() => setPage("settings")}><Settings2 size={18} /><span>Settings</span></button>
        <div className="sidebar-footer"><div className="user-avatar">D</div><div className="user-copy"><b>My workspace</b><small>Local instance</small></div><MoreHorizontal size={18} className="muted-icon" /></div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{currentTitle}</strong></div>
          <div className="top-actions"><div className="api-pill"><span className={`status-dot ${health ? "" : "red-dot"}`} />{health ? "All systems operational" : "API unavailable"}</div><button className="help-button" onClick={() => setToast("Tip: upload PDFs, then ask questions with page citations.")}><CircleHelp size={17} /> Help</button><div className="top-avatar">D</div></div>
        </header>

        <div className="page-content">
          {error && <div className="alert error-alert"><span>{error}</span><button onClick={() => setError("")}><X size={16} /></button></div>}
          {!health && <div className="alert warning-alert"><Activity size={16} /><span>Start the FastAPI server to connect this dashboard to your documents.</span><code>uvicorn api:app --reload --port 8000</code></div>}

          {page === "overview" && <Overview docs={docs} chunks={totalChunks} config={config} evaluation={evaluation} onUpload={() => fileRef.current?.click()} uploadBusy={uploadBusy} onNavigate={setPage} onChat={openChat} />}
          {page === "documents" && <Documents docs={filteredDocs} allCount={docs.length} search={search} setSearch={setSearch} onUpload={() => fileRef.current?.click()} uploadBusy={uploadBusy} onDelete={deleteDoc} onDrop={uploadFiles} />}
          {page === "chat" && <Chat messages={messages} busy={busy} question={question} setQuestion={setQuestion} sendQuestion={sendQuestion} bottomRef={bottomRef} docs={docs} selectedDocs={selectedDocs} setSelectedDocs={setSelectedDocs} onGoDocs={() => setPage("documents")} onClear={() => setMessages([])} />}
          {page === "evaluation" && <Evaluation evaluation={evaluation} />}
          {page === "settings" && <SettingsPage draft={settingsDraft} changeDraft={changeDraft} save={saveSettings} busy={busy} config={config} />}
        </div>
      </main>
      <input ref={fileRef} className="hidden-input" type="file" accept=".pdf,application/pdf" multiple onChange={e => uploadFiles(e.target.files)} />
      {toast && <div className="toast"><span className="toast-check"><Check size={15} /></span>{toast}<button onClick={() => setToast("")}><X size={14} /></button></div>}
    </div>
  );
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}
function Overview({ docs, chunks, config, evaluation, onUpload, uploadBusy, onNavigate, onChat }) {
  const summary = evaluation?.report?.summary;
  const retrieval = summary?.ground_truth_retrieval_metrics;
  const abstention = summary?.abstention_metrics || summary?.abstention;
  return <div className="overview-page">
    <div className="welcome-banner"><div className="welcome-orb orb-one" /><div className="welcome-orb orb-two" /><div className="welcome-content"><div className="welcome-tag"><Sparkles size={13} /> YOUR AI KNOWLEDGE SPACE</div><h1>Make your documents<br /><span>work smarter.</span></h1><p>Ask questions, uncover insights, and get answers you can trace back to the source.</p><div className="welcome-actions"><button className="btn btn-white" onClick={onUpload} disabled={uploadBusy}>{uploadBusy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}{uploadBusy ? "Uploading…" : "Upload documents"}</button><button className="btn btn-glass" onClick={() => onChat("")}>Open AI workspace <ArrowRight size={16} /></button></div></div><div className="hero-art"><div className="orbit orbit-a" /><div className="orbit orbit-b" /><div className="hero-doc hero-doc-back"><FileText size={24} /></div><div className="hero-doc hero-doc-main"><div className="hero-doc-head"><span /><span /><span /></div><div className="hero-lines"><i /><i /><i /><i /></div><div className="hero-citation"><ShieldCheck size={13} /> Verified source</div></div><div className="float-chip chip-source"><span className="mini-check"><Check size={11} /></span> Page citations</div><div className="float-chip chip-ai"><Sparkles size={13} /> AI powered</div></div></div>
    <div className="section-title-row"><div><h2>Your workspace at a glance</h2><p>Everything you need to find answers faster.</p></div><span className="updated-label"><span className="status-dot" /> Live workspace</span></div>
    <div className="stats-grid">
      <StatCard icon={FileText} tone="violet" label="Indexed documents" value={docs.length} foot="PDF files in your library" />
      <StatCard icon={BookOpen} tone="blue" label="Searchable passages" value={chunks.toLocaleString()} foot="Context chunks indexed" />
      <StatCard icon={Zap} tone="amber" label="AI provider" value={config?.llm_provider === "openai_compatible" ? "Connected" : "Extractive"} foot={config?.llm_model || "Evidence-only answers"} />
      <StatCard icon={ShieldCheck} tone="green" label="Citations" value="Verifiable" foot="Source pages attached to answers" />
    </div>
    <div className="overview-grid">
      <section className="panel recent-panel"><div className="panel-header"><div><h3>Recent documents</h3><p>Your indexed knowledge base</p></div><button className="text-action" onClick={() => onNavigate("documents")}>View library <ArrowRight size={14} /></button></div>
        {docs.length ? <div className="recent-list">{docs.slice(0, 4).map(doc => <div className="recent-row" key={doc.doc_id}><div className="file-tile"><FileText size={19} /></div><div className="recent-meta"><b>{doc.filename}</b><span>{doc.pages} pages · {doc.chunks} passages</span></div><span className="indexed-badge"><Check size={12} /> Indexed</span><button className="row-more" onClick={() => onNavigate("documents")}><ArrowRight size={15} /></button></div>)}</div> : <EmptyState icon={FileText} title="Your library is waiting" text="Upload a PDF to start building your searchable knowledge base." action={<button className="btn btn-primary" onClick={onUpload}><Plus size={16} /> Add your first PDF</button>} />}
      </section>
      <section className="panel quick-panel"><div className="panel-header"><div><h3>Start with a question</h3><p>Try one of these prompts</p></div><span className="sparkle-tile"><Sparkles size={17} /></span></div><div className="prompt-list">{prompts.map((p, i) => <button key={p} className="prompt-option" onClick={() => onChat(p)}><span className="prompt-number">0{i + 1}</span><span>{p}</span><ArrowRight size={15} /></button>)}</div><div className="quick-foot"><span className="secure-icon"><ShieldCheck size={14} /></span><span>Answers grounded in your documents, not guesswork.</span></div></section>
    </div>
    <div className="bottom-note"><div className="note-icon"><Activity size={16} /></div><div><b>Retrieval health</b><span>{retrieval ? `Recall@4: ${Number(retrieval["recall@4"] ?? retrieval.recall_at_k ?? 0).toFixed(2)} · Your evaluation report is available.` : "Run your evaluation to monitor retrieval quality and abstention behaviour."}</span></div><button className="text-action" onClick={() => onNavigate("evaluation")}>View evaluation <ArrowRight size={14} /></button></div>
  </div>;
}
function StatCard({ icon: Icon, tone, label, value, foot }) {
  return <div className="stat-card"><div className={`stat-icon ${tone}`}><Icon size={19} /></div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-foot">{foot}</div></div>;
}
function EmptyState({ icon: Icon, title, text, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={23} /></div><h3>{title}</h3><p>{text}</p>{action}</div>;
}

function Documents({ docs, allCount, search, setSearch, onUpload, uploadBusy, onDelete, onDrop }) {
  const [dragging, setDragging] = useState(false);
  return <div className="documents-page">
    <PageHeading eyebrow="KNOWLEDGE LIBRARY" title="Your documents" description="Upload PDFs, organize your sources, and keep every answer grounded in evidence." action={<button className="btn btn-primary" onClick={onUpload} disabled={uploadBusy}>{uploadBusy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}{uploadBusy ? "Indexing…" : "Upload PDFs"}</button>} />
    <div className={`upload-zone ${dragging ? "dragging" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); onDrop(e.dataTransfer.files); }}>
      <div className="upload-illustration"><CloudUpload size={27} /></div><div className="upload-copy"><h3>Drop your PDFs here</h3><p>or <button className="inline-link" onClick={onUpload}>browse files</button> from your computer</p><span>PDF only · Up to 30 MB per file · Multiple files supported</span></div><div className="upload-side"><div className="upload-mini"><FileText size={16} /><span>PDF</span></div><div className="upload-mini"><ShieldCheck size={16} /><span>Private</span></div></div>
    </div>
    <div className="library-toolbar"><div><h2>Indexed library <span className="count-pill">{allCount}</span></h2><p>Files are searchable once indexing completes.</p></div><label className="search-field"><Search size={16} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search documents…" /><kbd>⌘ K</kbd></label></div>
    <div className="panel document-table"><div className="table-head"><span>DOCUMENT</span><span>STATUS</span><span>PAGES</span><span>PASSAGES</span><span /></div>
      {docs.map(doc => <div className="table-row" key={doc.doc_id}><div className="table-doc"><div className="file-tile"><FileText size={18} /></div><div><b>{doc.filename}</b><small>ID: {shortId(doc.doc_id)}</small></div></div><div><span className="indexed-badge"><Check size={12} /> Indexed</span></div><div className="table-number">{doc.pages}</div><div className="table-number">{doc.chunks}</div><div className="table-actions"><button title="Remove document" className="icon-button danger-hover" onClick={() => onDelete(doc)}><Trash2 size={16} /></button></div></div>)}
      {!docs.length && <EmptyState icon={FileCheck2} title="No documents indexed yet" text="Upload a PDF above to make it searchable. Your existing data folder is used by the local API." action={null} />}
    </div>
    <div className="privacy-note"><ShieldCheck size={16} /><span><b>Your documents stay yours.</b> The React interface sends files to your local API for indexing; API keys stay on the Python backend.</span></div>
  </div>;
}

function Chat({ messages, busy, question, setQuestion, sendQuestion, bottomRef, docs, selectedDocs, setSelectedDocs, onGoDocs, onClear }) {
  return <div className="chat-page">
    <PageHeading eyebrow="AI DOCUMENT INTELLIGENCE" title="AI workspace" description="Ask in natural language. Get concise answers with source pages you can verify." action={<button className="btn btn-secondary" onClick={onClear}><Plus size={15} /> New conversation</button>} />
    <div className="chat-layout">
      <section className="chat-main panel">
        <div className="chat-topline"><div className="assistant-avatar"><Sparkles size={17} /></div><div><b>KnowledgePilot assistant</b><span><span className="status-dot" /> Grounded answers enabled</span></div><div className="chat-model-pill"><span className="model-dot" /> {selectedModelLabel(docs)}</div></div>
        <div className="chat-stream">
          {!messages.length && <div className="chat-welcome"><div className="welcome-ai-icon"><Bot size={29} /></div><h2>What would you like to know?</h2><p>I'll search your indexed PDFs and include page-level citations so you can verify every answer.</p>{!docs.length ? <button className="btn btn-primary" onClick={onGoDocs}><Upload size={15} /> Upload a document</button> : <div className="chat-suggestions">{prompts.map((p, i) => <button key={p} onClick={() => sendQuestion(p)}><Sparkles size={14} />{p}<ArrowRight size={14} /></button>)}</div>}</div>}
          {messages.map(message => <div className={`message-row ${message.role === "user" ? "user-row" : ""}`} key={message.id}>
            {message.role === "assistant" && <div className="message-avatar"><Sparkles size={15} /></div>}
            <div className={`message-body ${message.role === "user" ? "user-bubble" : "assistant-bubble"}`}><div className="message-label">{message.role === "user" ? "You" : "KnowledgePilot"}</div><div className="message-text">{message.content}</div>
              {message.result?.citations?.length > 0 && <div className="citation-block"><div className="citation-heading"><ShieldCheck size={14} /> SOURCES USED <span>{message.result.citations.length}</span></div>{message.result.citations.map(c => <div className="citation-card" key={c.index}><div className="citation-file-icon"><FileText size={15} /></div><div className="citation-details"><b>[{c.index}] {c.filename}</b><span>Page {c.page} · relevance {(c.score * 100).toFixed(0)}%</span><p>{c.excerpt}</p></div></div>)}</div>}
              {message.result && <div className="answer-meta"><span><Clock3 size={12} /> {fmtTime(message.result.total_latency_s)}</span><span><ShieldCheck size={12} /> {message.result.mode === "fallback" ? "Insufficient evidence" : message.result.mode === "llm" ? "LLM answer" : "Extractive answer"}</span>{message.result.retrieved?.length > 0 && <details className="retrieval-details"><summary>Retrieved context ({message.result.retrieved.length})</summary>{message.result.retrieved.map(r => <div className="retrieval-row" key={r.rank}><b>#{r.rank} · {r.filename}, p.{r.page}</b><span>score {r.score.toFixed(3)} · {r.used_as_evidence ? "evidence" : "below threshold"}</span><p>{r.text}</p></div>)}</details>}</div>}
            </div>
          </div>)}
          {busy && <div className="message-row"><div className="message-avatar"><Sparkles size={15} /></div><div className="thinking-bubble"><span /><span /><span /> Searching your knowledge base…</div></div>}
          <div ref={bottomRef} />
        </div>
        <form className="chat-composer" onSubmit={e => { e.preventDefault(); sendQuestion(); }}><div className="composer-wrap"><textarea className="chat-input" rows="1" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask anything about your PDFs…" onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); } }} /><div className="composer-footer"><span><Command size={12} /> Enter to send · Shift + Enter for new line</span><button className="send-button" type="submit" disabled={!question.trim() || busy}><Send size={16} /></button></div></div></form>
        <div className="chat-disclaimer">AI can make mistakes. Verify important information in the cited source pages.</div>
      </section>
      <aside className="chat-side">
        <div className="panel context-panel"><div className="side-panel-title"><BookOpen size={16} /><h3>Search context</h3></div><p>Choose which indexed documents should be used to answer your questions.</p>
          <button className={`scope-option ${!selectedDocs.length ? "selected" : ""}`} onClick={() => setSelectedDocs([])}><span className="radio-dot" /><span><b>All documents</b><small>Search your full library</small></span><span className="scope-count">{docs.length}</span></button>
          <div className="context-divider" />
          {docs.map(doc => <label key={doc.doc_id} className="doc-check-row"><input type="checkbox" checked={selectedDocs.includes(doc.doc_id)} onChange={e => setSelectedDocs(prev => e.target.checked ? [...prev, doc.doc_id] : prev.filter(id => id !== doc.doc_id))} /><span className="mini-pdf"><FileText size={13} /></span><span className="doc-check-name" title={doc.filename}>{doc.filename}</span></label>)}
          {!docs.length && <button className="btn btn-secondary btn-full" onClick={onGoDocs}>Add documents</button>}
        </div>
        <div className="panel grounded-panel"><div className="grounded-icon"><ShieldCheck size={18} /></div><h3>Evidence-first answers</h3><p>Responses are based on retrieved passages. When the documents don't contain enough evidence, the assistant should say so.</p><div className="grounded-feature"><Check size={13} /> Page-level citations</div><div className="grounded-feature"><Check size={13} /> Insufficient-evidence fallback</div><div className="grounded-feature"><Check size={13} /> Inspect retrieved context</div></div>
      </aside>
    </div>
  </div>;
}
function selectedModelLabel(docs) { return docs.length ? "Knowledge base ready" : "Waiting for documents"; }

function Evaluation({ evaluation }) {
  const report = evaluation?.report;
  const summary = report?.summary;
  const gt = summary?.ground_truth_retrieval_metrics || {};
  const abst = summary?.abstention_behaviour || summary?.abstention_metrics || summary?.abstention || {};
  const op = summary?.operational_metrics || {};
  const rows = report?.items || report?.per_question || report?.results || [];
  return <div className="evaluation-page">
    <PageHeading eyebrow="QUALITY & PERFORMANCE" title="Evaluation center" description="Track retrieval quality, abstention behaviour, and operational performance against your labelled test set." action={<a className="btn btn-secondary" href="https://github.com/devendra8861jee-data/knowledgepilot-ai" target="_blank" rel="noreferrer"><Github size={16} /> Repository</a>} />
    {!evaluation?.available ? <div className="panel eval-empty"><div className="empty-icon"><Gauge size={24} /></div><h3>No evaluation report yet</h3><p>{evaluation?.message || "Run the labelled dataset from your existing evaluation workflow to populate this dashboard."}</p></div> : <>
      <div className="eval-run-card"><div className="eval-run-icon"><Activity size={19} /></div><div><b>Latest evaluation report</b><span>Metrics below are read from the saved report, not estimated.</span></div><span className="report-chip"><span className="status-dot" /> Report loaded</span></div>
      <div className="eval-config-strip"><span><b>Embedder</b> {report?.config?.embedder || "—"}</span><span><b>Questions</b> {summary?.n_questions ?? "—"}</span><span><b>Top-k</b> {report?.config?.k ?? 4}</span><span><b>Threshold</b> {report?.config?.min_relevance ?? "—"}</span></div>
      <div className="section-title-row eval-section-title"><div><h2>Retrieval performance</h2><p>How well the right source passages are ranked.</p></div></div>
      <div className="eval-metric-grid">
        <MetricCard title={`Recall@${report?.config?.k ?? 4}`} value={metric(gt, ["recall@4", "recall_at_k"])} sub="Expected pages retrieved" color="green" />
        <MetricCard title={`Hit rate@${report?.config?.k ?? 4}`} value={metric(gt, ["hit_rate@4", "hit_rate_at_k"])} sub="Questions with a target hit" color="blue" />
        <MetricCard title="Precision" value={metric(gt, ["precision@4", "precision_at_k", "precision"])} sub="Relevant passages in results" color="purple" />
        <MetricCard title="MRR" value={metric(gt, ["mrr", "mean_reciprocal_rank"])} sub="Rank of the first relevant hit" color="amber" />
      </div>
      <div className="section-title-row eval-section-title"><div><h2>Reliability & latency</h2><p>Measure when the system answers, abstains, and how long retrieval takes.</p></div></div>
      <div className="overview-grid eval-secondary-grid">
        <section className="panel eval-detail-panel"><div className="panel-header"><div><h3>Abstention behaviour</h3><p>Questions with insufficient evidence</p></div><ShieldCheck size={18} className="green-text" /></div><div className="detail-stat"><span>Correct abstention</span><b>{metric(abst, ["correct_abstention_rate_unanswerable", "correct_abstention_rate", "correct_abstention"])}</b></div><div className="detail-stat"><span>False abstention</span><b>{metric(abst, ["false_abstention_rate_answerable", "false_abstention_rate", "false_abstention"])}</b></div><div className="detail-stat"><span>Keyword coverage</span><b>{metric(summary?.proxy_metrics || summary?.lexical_metrics || summary, ["context_keyword_coverage", "keyword_coverage", "keyword_coverage_in_context"])}</b></div></section>
        <section className="panel eval-detail-panel"><div className="panel-header"><div><h3>Operational metrics</h3><p>Latency and reliability</p></div><Clock3 size={18} className="blue-text" /></div><div className="detail-stat"><span>Mean latency</span><b>{seconds(op, ["mean_latency_s", "mean_latency"])}</b></div><div className="detail-stat"><span>P95 latency</span><b>{seconds(op, ["p95_latency_s", "p95_latency"])}</b></div><div className="detail-stat"><span>Error rate</span><b>{metric(op, ["error_rate"])}</b></div></section>
      </div>
      <section className="panel judge-panel"><div className="judge-icon"><Sparkles size={17} /></div><div><h3>LLM-judged quality metrics</h3><p>Faithfulness, answer relevance, and context relevance require an LLM judge to be configured and measured. They are not inferred by this UI.</p></div><span className="na-pill">Not measured</span></section>
      <section className="panel human-review"><div className="panel-header"><div><h3>Human review checklist</h3><p>Recommended before treating this run as production-ready.</p></div><span className="review-pill">Manual review</span></div><div className="review-list"><div><span>01</span> Compare generated answers against expected answers.</div><div><span>02</span> Check that each citation excerpt supports its claim.</div><div><span>03</span> Inspect false abstentions and unsupported-question cases.</div></div></section>
      <section className="panel question-results"><div className="panel-header"><div><h3>Per-question results</h3><p>{rows.length} labelled questions from the latest evaluation.</p></div><span className="count-pill">{rows.length}</span></div>
        <div className="question-table"><div className="question-table-head"><span>QUESTION</span><span>TYPE</span><span>RESULT</span><span>TOP SCORE</span><span>LATENCY</span></div>
          {rows.map((row, index) => <div className="question-table-row" key={row.id || index}>
            <div className="question-cell"><b>{row.id || `Q${String(index + 1).padStart(2, "0")}`}</b><span>{row.question}</span>{row.answer && <details><summary>View answer and expected answer</summary><p><strong>System:</strong> {row.answer}</p><p><strong>Expected:</strong> {row.expected_answer || "Not included in report"}</p></details>}</div>
            <div><span className={`question-type ${row.answerable ? "type-answerable" : "type-unknown"}`}>{row.answerable ? "Answerable" : "Unanswerable"}</span></div>
            <div><span className={`question-result ${row.abstained ? (row.answerable ? "result-warn" : "result-good") : (row.answerable ? "result-good" : "result-warn")}`}>{row.abstained ? "Abstained" : "Answered"}</span></div>
            <div className="score-cell">{row.retrieved?.[0]?.score != null ? Number(row.retrieved[0].score).toFixed(3) : "—"}</div>
            <div className="score-cell">{row.total_latency_s != null ? fmtTime(row.total_latency_s) : "—"}</div>
          </div>)}
        </div>
      </section>
    </>}
  </div>;
}
function metric(obj, keys) {
  if (!obj) return "—";
  for (const key of keys) if (obj[key] != null && Number.isFinite(Number(obj[key]))) return (Number(obj[key]) * 100).toFixed(1) + "%";
  return "—";
}
function seconds(obj, keys) {
  if (!obj) return "—";
  for (const key of keys) if (obj[key] != null && Number.isFinite(Number(obj[key]))) return Number(obj[key]).toFixed(3) + " s";
  return "—";
}
function MetricCard({ title, value, sub, color }) {
  const numeric = Number.parseFloat(value);
  const width = value !== "—" ? Math.min(100, Math.max(0, numeric)) : 0;
  return <div className="metric-card"><div className="metric-card-head"><span>{title}</span><span className={`metric-dot ${color}`} /></div><div className="metric-number">{value}</div><div className="metric-sub">{sub}</div><div className="metric-track"><span className={color} style={{ width: width + "%" }} /></div></div>;
}

function SettingsPage({ draft, changeDraft, save, busy, config }) {
  if (!draft) return <div className="loading-card"><LoaderCircle className="spin" /> Loading settings…</div>;
  return <div className="settings-page"><PageHeading eyebrow="PREFERENCES & CONFIGURATION" title="Settings" description="Tune retrieval behaviour and configure the model used to generate answers." />
    <form onSubmit={save} className="settings-form">
      <section className="panel settings-section"><div className="settings-section-head"><div className="settings-section-icon violet"><BookOpen size={18} /></div><div><h2>Embeddings</h2><p>Choose how your documents are represented for semantic search.</p></div></div>
        <label className="form-field"><span>Embedding backend</span><select value={draft.embedding_backend} onChange={e => changeDraft("embedding_backend", e.target.value)}><option value="sentence-transformers">Sentence Transformers · semantic</option><option value="hashing">Hashing · offline</option></select><small>Changing the embedder selects a separate vector collection. Re-index documents if needed.</small></label>
        <div className="settings-two-col"><RangeField label="Chunk size (characters)" value={draft.chunk_size} min={200} max={2000} step={50} onChange={v => changeDraft("chunk_size", v)} /><RangeField label="Chunk overlap (characters)" value={draft.chunk_overlap} min={0} max={500} step={10} onChange={v => changeDraft("chunk_overlap", v)} /></div>
      </section>
      <section className="panel settings-section"><div className="settings-section-head"><div className="settings-section-icon blue"><Search size={18} /></div><div><h2>Retrieval</h2><p>Control how many passages are searched and the minimum evidence score.</p></div></div>
        <div className="settings-two-col"><RangeField label="Top-k passages" value={draft.top_k} min={1} max={10} step={1} onChange={v => changeDraft("top_k", v)} /><RangeField label="Minimum relevance score" value={draft.min_relevance} min={0} max={1} step={0.01} onChange={v => changeDraft("min_relevance", v)} /></div>
      </section>
      <section className="panel settings-section"><div className="settings-section-head"><div className="settings-section-icon green"><Bot size={18} /></div><div><h2>Answer generation</h2><p>Connect an OpenAI-compatible provider such as Groq using server-side environment variables.</p></div></div>
        <label className="form-field"><span>LLM provider</span><select value={draft.llm_provider} onChange={e => changeDraft("llm_provider", e.target.value)}><option value="none">None · extractive answers</option><option value="openai_compatible">OpenAI-compatible API</option></select></label>
        <div className="settings-two-col"><label className="form-field"><span>Model ID</span><input value={draft.llm_model || ""} onChange={e => changeDraft("llm_model", e.target.value)} placeholder="openai/gpt-oss-20b" /></label><label className="form-field"><span>Base URL</span><input value={draft.llm_base_url || ""} onChange={e => changeDraft("llm_base_url", e.target.value)} placeholder="https://api.groq.com/openai/v1" /></label></div>
        <div className={`key-status ${config?.has_api_key ? "key-ok" : "key-missing"}`}><ShieldCheck size={16} /><span>{config?.has_api_key ? "API key found in backend environment. Its value is never sent to the browser." : "No API key detected. Add LLM_API_KEY to the root .env file, then restart the API."}</span></div>
      </section>
      <div className="settings-save-row"><span>Changes apply to this API session. Chunk settings affect newly indexed PDFs.</span><button className="btn btn-primary" disabled={busy} type="submit">{busy ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />} Apply settings</button></div>
    </form>
  </div>;
}
function RangeField({ label, value, min, max, step, onChange }) {
  return <label className="range-field"><span>{label}<b>{value}</b></span><input type="range" value={value} min={min} max={max} step={step} onChange={e => onChange(Number(e.target.value))} /><small><span>{min}</span><span>{max}</span></small></label>;
}
