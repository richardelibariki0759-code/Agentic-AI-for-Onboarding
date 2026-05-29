import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const TOPICS = [
  { label: "ONA App", icon: "📱", iconImg: "/icons/ona-app.png", desc: "Forms, data collection & sync" },
  { label: "Data Collection", icon: "🌿", desc: "Field methods & image capture" },
  { label: "Bruno Pushcart", icon: "🤖", iconImg: "/icons/bruno.png", desc: "Device assembly & setup" },
];

const QUICK_REPLIES = {
  awaiting_step_confirmation: ["Yes, done!", "No,I need help","No, I'm stuck 🙋"],
  awaiting_proceed_or_new: ["Next step", "New topic"],
  awaiting_confirmation: ["Yes, I'm familiar", "No, I'm new to this"],
};

// ─── Topic icon — image if available, emoji fallback ─────────────────────────
function TopicIcon({ topic, size = 28 }) {
  if (topic.iconImg) {
    return (
      <img
        src={topic.iconImg}
        alt={topic.label}
        style={{ width: size, height: size, objectFit: "contain", borderRadius: 4 }}
        onError={(e) => { e.target.style.display = "none"; e.target.nextSibling.style.display = "inline"; }}
      />
    );
  }
  return <span style={{ fontSize: size }}>{topic.icon}</span>;
}

// ─── Markdown-lite renderer ──────────────────────────────────────────────────
function MarkdownLine({ text }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") && p.endsWith("**") ? (
          <strong key={i}>{p.slice(2, -2)}</strong>
        ) : (
          <span key={i}>{p}</span>
        )
      )}
    </>
  );
}

// Strip [Image URL: ...] labels AND bare image URLs from display text,
// and collect the URLs so we can render them as <img> tags instead.
function extractAndCleanImages(text) {
  if (!text) return { cleaned: text, urls: [] };
  const urls = [];
  const labelPattern = /\[Image(?:\s+URL)?:\s*(https?:\/\/[^\]\s]+)\]/gi;
  let m;
  while ((m = labelPattern.exec(text)) !== null) {
    urls.push(m[1]);
  }
  const barePattern = /(https?:\/\/[^\s\]]+\.(?:jpg|jpeg|png|webp))/gi;
  while ((m = barePattern.exec(text)) !== null) {
    if (!urls.includes(m[1])) urls.push(m[1]);
  }
  let cleaned = text
    .replace(/\[Image(?:\s+URL)?:\s*https?:\/\/[^\]\s]+\]/gi, "")
    .replace(/(https?:\/\/[^\s\]]+\.(?:jpg|jpeg|png|webp))/gi, "")
    .replace(/\[\s*\]/g, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return { cleaned, urls };
}

// Split content into "before checkpoint" and "after checkpoint"
// Backend sends: \n\n---\n\n✅ **Have you completed this step?**
function splitAtCheckpoint(content) {
  if (!content) return { before: content, after: null };
  const dividerCheckpoint = /\n\n---\n\n[\u2705✅]/;
  let idx = content.search(dividerCheckpoint);
  if (idx !== -1) {
    return {
      before: content.slice(0, idx).trimEnd(),
      after: content.slice(idx).replace(/^\n+/, ""),
    };
  }
  const checkpointPattern = /[\u2705✅]\s*\*{0,2}Have you completed this step\?\*{0,2}/i;
  idx = content.search(checkpointPattern);
  if (idx === -1) return { before: content, after: null };
  return {
    before: content.slice(0, idx).trimEnd(),
    after: content.slice(idx),
  };
}

function MessageContent({ content }) {
  if (!content) return null;
  const lines = content.split("\n");
  const elements = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("### ")) {
      elements.push(
        <h3 key={i} style={{ margin: "0.75rem 0 0.25rem", fontSize: 15, fontWeight: 600, color: "var(--color-text-primary)" }}>
          {line.slice(4)}
        </h3>
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h2 key={i} style={{ margin: "0.75rem 0 0.25rem", fontSize: 17, fontWeight: 600 }}>
          {line.slice(3)}
        </h2>
      );
    } else if (line.trim() === "---") {
      elements.push(
        <hr key={i} style={{ border: "none", borderTop: "0.5px solid var(--color-border-tertiary)", margin: "0.75rem 0" }} />
      );
    } else if (line.startsWith("- ") || line.startsWith("• ")) {
      elements.push(
        <div key={i} style={{ display: "flex", gap: 8, marginBottom: 4 }}>
          <span style={{ color: "var(--color-text-secondary)", flexShrink: 0 }}>•</span>
          <span><MarkdownLine text={line.slice(2)} /></span>
        </div>
      );
    } else if (line.trim() === "") {
      elements.push(<div key={i} style={{ height: 8 }} />);
    } else {
      elements.push(
        <p key={i} style={{ margin: "0 0 4px" }}>
          <MarkdownLine text={line} />
        </p>
      );
    }
    i++;
  }
  return <div style={{ fontSize: 15, lineHeight: 1.65 }}>{elements}</div>;
}

// ─── Step progress ───────────────────────────────────────────────────────────
function StepProgress({ content }) {
  const match = content?.match(/`(\d+)\/(\d+)`/);
  if (!match) return null;
  const current = parseInt(match[1]);
  const total = parseInt(match[2]);
  const pct = Math.round((current / total) * 100);
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 6 }}>
        <span>Step {current} of {total}</span>
        <span>{pct}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 99, background: "var(--color-background-secondary)", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, borderRadius: 99, background: "linear-gradient(90deg, #1D9E75, #5DCAA5)", transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}

// ─── Image gallery (no URLs shown) ───────────────────────────────────────────
function ImageGallery({ urls }) {
  if (!urls || urls.length === 0) return null;
  return (
    <div style={{ marginTop: 12, marginBottom: 12, display: "flex", flexDirection: "column", gap: 10 }}>
      {urls.map((url, i) => (
        <div
          key={i}
          style={{
            borderRadius: 10,
            overflow: "hidden",
            border: "0.5px solid var(--color-border-tertiary)",
            background: "var(--color-background-secondary)",
          }}
        >
          <img
            src={url}
            alt={`Reference image ${i + 1}`}
            style={{ width: "100%", maxWidth: 650, display: "block", borderRadius: 10 }}
            onError={(e) => { e.target.style.display = "none"; }}
          />
        </div>
      ))}
    </div>
  );
}

// ─── Message — images shown BEFORE the ✅ checkpoint line ────────────────────
function Message({ msg }) {
  const isBot = msg.role === "assistant";
  const hasStep = isBot && msg.content?.includes("📍 Step");

  const { cleaned: cleanedContent, urls: inlineUrls } = isBot
    ? extractAndCleanImages(msg.content)
    : { cleaned: msg.content, urls: [] };

  const allImageUrls = isBot
    ? [...new Set([...inlineUrls, ...(msg.images || [])])]
    : [];

  const { before, after } = isBot
    ? splitAtCheckpoint(cleanedContent)
    : { before: cleanedContent, after: null };

  return (
    <div style={{ display: "flex", justifyContent: isBot ? "flex-start" : "flex-end", marginBottom: 16, animation: "fadeSlideIn 0.25s ease" }}>
      {isBot && (
        <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg, #1D9E75, #0F6E56)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, flexShrink: 0, marginRight: 10, marginTop: 2 }}>
          🌱
        </div>
      )}

      <div style={{ maxWidth: "75%" }}>
        {hasStep && <StepProgress content={cleanedContent} />}

        <div style={{
          padding: "12px 16px",
          borderRadius: isBot ? "4px 16px 16px 16px" : "16px 4px 16px 16px",
          background: isBot ? "var(--color-background-primary)" : "linear-gradient(135deg, #1D9E75, #0F6E56)",
          border: isBot ? "0.5px solid var(--color-border-tertiary)" : "none",
          color: isBot ? "var(--color-text-primary)" : "#fff",
          boxShadow: isBot ? "none" : "0 2px 12px rgba(29,158,117,0.3)",
        }}>
          <MessageContent content={before} />

          {isBot && allImageUrls.length > 0 && (
            <ImageGallery urls={allImageUrls} />
          )}

          {after && <MessageContent content={after} />}
        </div>
      </div>
    </div>
  );
}

// ─── Typing indicator ────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
      <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg, #1D9E75, #0F6E56)", display: "flex", alignItems: "center", justifyContent: "center" }}>
        🌱
      </div>
      <div style={{ padding: "14px 18px", borderRadius: "4px 16px 16px 16px", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", display: "flex", gap: 6 }}>
        {[0, 1, 2].map((i) => (
          <div key={i} style={{ width: 8, height: 8, borderRadius: "50%", background: "#1D9E75", animation: `typingBounce 1.4s ease-in-out ${i * 0.2}s infinite` }} />
        ))}
      </div>
    </div>
  );
}

// ─── Quick replies ───────────────────────────────────────────────────────────
function QuickReplies({ state, onSend }) {
  const replies = QUICK_REPLIES[state];
  if (!replies) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, padding: "8px 0" }}>
      {replies.map((r) => (
        <button key={r} onClick={() => onSend(r)} style={{ padding: "8px 16px", borderRadius: 99, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: "var(--color-text-primary)", fontSize: 13, cursor: "pointer" }}>
          {r}
        </button>
      ))}
    </div>
  );
}

// ─── Admin Upload Panel (admin-only) ─────────────────────────────────────────
function AdminUploadPanel() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setResult(null);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setResult(`✅ Uploaded: ${data.filename || file.name} — ${data.chunks_added ?? ""} chunks indexed`);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(`⚠️ ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{
      margin: "16px 20px",
      padding: "14px 16px",
      borderRadius: 12,
      background: "rgba(29,158,117,0.08)",
      border: "0.5px solid rgba(29,158,117,0.3)",
    }}>
      <p style={{ margin: "0 0 10px", fontSize: 11, fontWeight: 700, color: "#1D9E75", letterSpacing: "0.05em", textTransform: "uppercase" }}>
        🔒 Admin — Upload Document
      </p>

      <div
        onClick={() => fileInputRef.current?.click()}
        style={{
          padding: "10px 12px",
          borderRadius: 8,
          border: "0.5px dashed rgba(29,158,117,0.5)",
          background: "var(--color-background-primary)",
          cursor: "pointer",
          fontSize: 13,
          color: file ? "var(--color-text-primary)" : "var(--color-text-secondary)",
          marginBottom: 10,
          textAlign: "center",
        }}
      >
        {file ? `📄 ${file.name}` : "Click to choose a file…"}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        style={{ display: "none" }}
        onChange={(e) => { setFile(e.target.files[0] || null); setResult(null); setError(null); }}
      />

      <button
        onClick={handleUpload}
        disabled={!file || uploading}
        style={{
          width: "100%",
          padding: "9px",
          borderRadius: 8,
          border: "none",
          background: file && !uploading ? "linear-gradient(135deg, #1D9E75, #0F6E56)" : "var(--color-background-secondary)",
          color: file && !uploading ? "#fff" : "var(--color-text-secondary)",
          fontSize: 13,
          fontWeight: 600,
          cursor: file && !uploading ? "pointer" : "not-allowed",
          transition: "all 0.2s",
        }}
      >
        {uploading ? "Uploading…" : "Upload & Index"}
      </button>

      {result && (
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "#1D9E75" }}>{result}</p>
      )}
      {error && (
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "var(--color-text-danger, #D85A30)" }}>{error}</p>
      )}
    </div>
  );
}

// ─── Welcome screen ──────────────────────────────────────────────────────────
function WelcomeScreen({ onTopicSelect }) {
  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "48px 24px 24px", display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div style={{ textAlign: "center", maxWidth: 520, marginBottom: 48 }}>
        <div style={{ width: 72, height: 72, borderRadius: "50%", background: "linear-gradient(135deg, #1D9E75, #0F6E56)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32, margin: "0 auto 24px" }}>
          🌱
        </div>
        <h1 style={{ fontSize: 32, fontWeight: 700, margin: "0 0 12px" }}>What do you want to do today?</h1>
        <p style={{ fontSize: 16, color: "var(--color-text-secondary)", margin: 0, lineHeight: 1.6 }}>
          Your AI training Onboarding for ONA, Bruno & data collection workflows.
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, width: "100%", maxWidth: 560 }}>
        {TOPICS.map((t) => (
          <button key={t.label} onClick={() => onTopicSelect(t.label)} style={{ padding: "20px 16px", borderRadius: 16, border: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)", cursor: "pointer" }}>
            {/* IMAGE icon if available, emoji fallback */}
            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: 36 }}>
              <TopicIcon topic={t} size={50} />
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, marginTop: 8 }}>{t.label}</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 4 }}>{t.desc}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────
function Sidebar({ onNewChat, onTopicSelect, currentState, isAdmin }) {
  const [collapsed, setCollapsed] = useState(false);

  const STATE_LABELS = {
    idle: { label: "Ready", color: "#1D9E75" },
    awaiting_confirmation: { label: "Assessing", color: "#EF9F27" },
    awaiting_goal: { label: "Getting goal", color: "#EF9F27" },
    awaiting_step_confirmation: { label: "Step in progress", color: "#378ADD" },
    awaiting_challenge: { label: "Helping", color: "#D85A30" },
    awaiting_proceed_or_new: { label: "Awaiting next", color: "#7F77DD" },
  };

  const stateInfo = STATE_LABELS[currentState] || STATE_LABELS.idle;

  return (
    <aside style={{
      width: collapsed ? 74 : 240,
      flexShrink: 0,
      transition: "width 0.25s ease",
      borderRight: "0.5px solid var(--color-border-tertiary)",
      background: "var(--color-background-secondary)",
      display: "flex",
      flexDirection: "column",
      padding: "20px 0",
    }}>
      <div style={{ padding: collapsed ? "0 12px 20px" : "0 20px 20px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: collapsed ? "center" : "space-between", marginBottom: 16 }}>
          {!collapsed && (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(135deg, #1D9E75, #0F6E56)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                🌱
              </div>
              <span style={{ fontWeight: 700, fontSize: 15 }}>Onboarding Agent</span>
            </div>
          )}
          <button onClick={() => setCollapsed(!collapsed)} style={{ width: 30, height: 30, borderRadius: 8, border: "none", cursor: "pointer" }}>
            {collapsed ? "→" : "←"}
          </button>
        </div>
        <button onClick={onNewChat} style={{ width: "100%", padding: "10px", borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", cursor: "pointer" }}>
          {collapsed ? "+" : "+ New conversation"}
        </button>
      </div>

      <div style={{ padding: collapsed ? "16px 10px" : "16px 20px" }}>
        {!collapsed && (
          <p style={{ fontSize: 11, fontWeight: 600, marginBottom: 10 }}>QUICK TOPICS</p>
        )}
        {TOPICS.map((t) => (
          <button key={t.label} onClick={() => onTopicSelect(t.label)} style={{ width: "100%", padding: "10px", borderRadius: 8, border: "none", background: "transparent", cursor: "pointer", textAlign: "left", display: "flex", alignItems: "center", gap: 10 }}>
            {/* IMAGE icon if available, emoji fallback */}
            <TopicIcon topic={t} size={20} />
            {!collapsed && <span>{t.label}</span>}
          </button>
        ))}
      </div>

      {/* Admin upload panel — only visible when isAdmin === true */}
      {isAdmin && !collapsed && <AdminUploadPanel />}

      <div style={{ marginTop: "auto", padding: collapsed ? "16px 10px" : "16px 20px" }}>
        {!collapsed && (
          <p style={{ fontSize: 11, fontWeight: 600, marginBottom: 8 }}>SESSION STATE</p>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: stateInfo.color }} />
          {!collapsed && <span style={{ fontSize: 12 }}>{stateInfo.label}</span>}
        </div>

        {/* Admin indicator badge */}
        {isAdmin && !collapsed && (
          <div style={{ marginTop: 8, padding: "3px 8px", borderRadius: 6, background: "rgba(29,158,117,0.12)", border: "0.5px solid rgba(29,158,117,0.3)", fontSize: 11, color: "#1D9E75", fontWeight: 600, display: "inline-block" }}>
            🔒 Admin
          </div>
        )}
      </div>
    </aside>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [state, setState] = useState("idle");
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [error, setError] = useState(null);

  const [isAdmin] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("admin") === "1";
  });

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/session`, { method: "POST" })
      .then((r) => r.json())
      .then((d) => setSessionId(d.session_id))
      .catch(() => setError("Could not connect to the backend."));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const sendMessage = useCallback(
    async (text) => {
      if (!text.trim() || !sessionId || isLoading) return;
      const userMsg = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setInputValue("");
      setIsLoading(true);
      setShowWelcome(false);
      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message: text }),
        });
        const data = await res.json();
        setMessages((prev) => [...prev, { role: "assistant", content: data.reply, images: data.images || [] }]);
        setState(data.state);
      } catch {
        setMessages((prev) => [...prev, { role: "assistant", content: "⚠️ Backend connection error.", images: [] }]);
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, isLoading]
  );

  const handleNewChat = useCallback(() => {
    fetch(`${API_BASE}/session`, { method: "POST" })
      .then((r) => r.json())
      .then((d) => {
        setSessionId(d.session_id);
        setMessages([]);
        setState("idle");
        setShowWelcome(true);
      });
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputValue);
    }
  };

  return (
    <>
      <style>{`
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes typingBounce { 0%,60%,100% { transform: translateY(0); opacity:0.4; } 30% { transform: translateY(-7px); opacity:1; } }
        * { box-sizing: border-box; }
        textarea { resize: none; outline: none; font-family: inherit; }
      `}</style>

      <div style={{ display: "flex", height: "100vh", background: "var(--color-background-tertiary)", fontFamily: "var(--font-sans)" }}>
        <Sidebar
          onNewChat={handleNewChat}
          onTopicSelect={(topic) => sendMessage(`🔹 ${topic}`)}
          currentState={state}
          isAdmin={isAdmin}
        />

        <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Header */}
          <div style={{ padding: "16px 24px", borderBottom: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Field Onboarding Agentic AI</h2>
              <p style={{ margin: 0, fontSize: 12, color: "var(--color-text-secondary)" }}>
                Artemis Onboarding Agentic AI
              </p>
            </div>
            <div style={{
              padding: "4px 12px",
              borderRadius: 99,
              fontSize: 12,
              background: sessionId ? "var(--color-background-success)" : "var(--color-background-danger)",
              color: sessionId ? "var(--color-text-success)" : "var(--color-text-danger)",
              border: `0.5px solid ${sessionId ? "var(--color-border-success)" : "var(--color-border-danger)"}`,
            }}>
              {sessionId ? "● Connected" : "○ Connecting…"}
            </div>
          </div>

          {showWelcome && messages.length === 0 ? (
            <WelcomeScreen onTopicSelect={(topic) => sendMessage(`🔹 ${topic}`)} />
          ) : (
            <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
              {messages.map((msg, i) => <Message key={i} msg={msg} />)}
              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          )}

          <div style={{ padding: "16px 24px", borderTop: "0.5px solid var(--color-border-tertiary)" }}>
            <QuickReplies state={state} onSend={sendMessage} />
            <div style={{ display: "flex", gap: 12, alignItems: "flex-end", background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: 14, padding: "10px 14px" }}>
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question..."
                rows={1}
                disabled={isLoading || !sessionId}
                style={{ flex: 1, border: "none", background: "transparent", fontSize: 14 }}
              />
              <button
                onClick={() => sendMessage(inputValue)}
                disabled={!inputValue.trim() || isLoading}
                style={{ width: 36, height: 36, borderRadius: 10, border: "none", cursor: "pointer" }}
              >
                ↑
              </button>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
