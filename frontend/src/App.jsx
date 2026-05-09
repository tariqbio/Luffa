import React, { useState, useRef, useCallback } from 'react';
import { predictDisease } from './api.js';

const SCAN_STEPS = [
  'Preprocessing — 224×224 resize, normalise to [0,1]',
  'CNN inference — 4-block convolutional network',
  'CNN-ViT Hybrid inference — transformer attention heads',
  'Ensemble soft voting — averaging both model outputs',
  'Generating full disease management report',
];

const TABS = ['Overview', 'Symptoms', 'Treatment', 'Prevention', 'Impact'];

/* ─── Header ──────────────────────────────────────────────── */
function Header() {
  return (
    <header className="header">
      <div className="wrap header__inner">
        <div className="logo">
          <div className="logo__mark">🌿</div>
          <div>
            <div className="logo__name">LuffaGuard</div>
            <div className="logo__tagline">Leaf Disease Detection System</div>
          </div>
        </div>
        <div className="status-badge">
          <span className="status-dot" />
          CNN + CNN-ViT Hybrid · Ensemble Active
        </div>
      </div>
    </header>
  );
}

/* ─── Upload ──────────────────────────────────────────────── */
function UploadView({ onFile }) {
  const [drag, setDrag] = useState(false);
  const inp = useRef();
  const handle = f => f?.type.startsWith('image/') && onFile(f);

  return (
    <div className="wrap">
      {/* Hero */}
      <section className="hero">
        <div className="hero__eyebrow">AI-Powered Plant Diagnostics</div>
        <h1 className="hero__h1 anim-fade-up anim-fade-up--1">
          Detect disease in<br /><em>Luffa aegyptiaca</em><br />in seconds
        </h1>
        <p className="hero__sub anim-fade-up anim-fade-up--2">
          Upload a leaf photograph. The ensemble of your trained CNN and
          CNN-ViT Hybrid models diagnoses the disease and returns a complete
          management report with treatment and prevention protocols.
        </p>
        <div className="hero__pills anim-fade-up anim-fade-up--3">
          <span className="pill pill--accent">Ensemble Model</span>
          <span className="pill">CNN — 99.19% accuracy</span>
          <span className="pill">CNN-ViT Hybrid — 99.51% accuracy</span>
          <span className="pill">6,166 training images</span>
          <span className="pill">2 disease classes</span>
        </div>
      </section>

      {/* Drop zone */}
      <div className="upload-section anim-fade-up anim-fade-up--4">
        <div
          className={`drop-zone ${drag ? 'drop-zone--active' : 'drop-zone--idle'}`}
          onClick={() => inp.current.click()}
          onDragEnter={e => { e.preventDefault(); setDrag(true); }}
          onDragOver={e  => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
        >
          <div className="drop-zone__inner" />
          <div className="drop-zone__glow" />
          <span className="drop-zone__icon">🍃</span>
          <h3 className="drop-zone__title">
            {drag ? 'Release to analyse' : 'Drop a leaf image here'}
          </h3>
          <p className="drop-zone__sub">
            Drag & drop a photograph of a <span>Luffa aegyptiaca</span> leaf<br />
            or browse your files · JPG, PNG, WebP supported
          </p>
          <div className="drop-zone__btn">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2v9M4 7l4-4 4 4" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 13h12" strokeLinecap="round"/>
            </svg>
            Choose Image
          </div>
          <input ref={inp} type="file" accept="image/*" style={{ display:'none' }}
            onChange={e => handle(e.target.files[0])} />
        </div>
      </div>
    </div>
  );
}

/* ─── Scan ────────────────────────────────────────────────── */
function ScanView({ preview, step }) {
  return (
    <div className="wrap scan-view">
      <div className="scan-card">
        <div className="scan-image">
          {preview && <img src={preview} alt="Analysed leaf" />}
          <div className="scan-beam" />
          <div className="scan-cross scan-cross--tl" />
          <div className="scan-cross scan-cross--tr" />
          <div className="scan-cross scan-cross--bl" />
          <div className="scan-cross scan-cross--br" />
        </div>
        <div className="scan-info">
          <div className="scan-label">Analysing specimen</div>
          <div className="scan-title">Running disease detection</div>
          <div className="step-list">
            {SCAN_STEPS.map((txt, i) => {
              const cls = i < step ? 'step--done' : i === step ? 'step--active' : '';
              return (
                <div key={i} className={`step ${cls}`}>
                  <div className="step__dot">{i < step ? '✓' : i + 1}</div>
                  <div className="step__text">{txt}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Results ─────────────────────────────────────────────── */
function ResultView({ data, onReset }) {
  const [tab, setTab] = useState('Overview');
  const di   = data.disease_info;
  const prob = data.probabilities;
  const labs = Object.keys(prob.ensemble);

  const confColor = c => c >= 90 ? 'var(--phosphor)' : c >= 70 ? 'var(--amber)' : 'var(--red)';
  const urgCls    = u => u === 'high' ? 'urgency-tag--high' : u === 'moderate' ? 'urgency-tag--moderate' : 'urgency-tag--low';

  const tabContent = () => {
    switch (tab) {
      case 'Overview': return (
        <div className="info-overview">
          <div className="info-block" style={{ gridColumn: '1/-1' }}>
            <div className="info-block__label">Disease Overview</div>
            <p>{di.overview}</p>
          </div>
          <div className="info-block">
            <div className="info-block__label">Causes & Vectors</div>
            <ul className="info-list">
              {di.causes.map((c,i) => (
                <li key={i}><span className="info-list__icon">→</span><span>{c}</span></li>
              ))}
            </ul>
          </div>
          <div className="info-block">
            <div className="info-block__label">Action Required</div>
            <p style={{ fontFamily:'var(--ff-mono)', fontSize:'13px', color: di.urgency_level === 'high' ? 'var(--red)' : 'var(--amber)', lineHeight:'1.7' }}>
              {di.urgency}
            </p>
          </div>
        </div>
      );
      case 'Symptoms': return (
        <div className="info-block">
          <div className="info-block__label">Observed Symptoms</div>
          <ul className="info-list">
            {di.symptoms.map((s,i) => (
              <li key={i}><span className="info-list__icon">◆</span><span>{s}</span></li>
            ))}
          </ul>
        </div>
      );
      case 'Treatment': return (
        <div className="info-block">
          <div className="info-block__label">Treatment Protocol</div>
          <ul className="info-list">
            {di.treatment.map((t,i) => (
              <li key={i}><span className="info-list__icon info-list__icon--green">✦</span><span>{t}</span></li>
            ))}
          </ul>
        </div>
      );
      case 'Prevention': return (
        <div className="info-block">
          <div className="info-block__label">Prevention Measures</div>
          <ul className="info-list">
            {di.prevention.map((p,i) => (
              <li key={i}><span className="info-list__icon info-list__icon--amber">◉</span><span>{p}</span></li>
            ))}
          </ul>
        </div>
      );
      case 'Impact': return (
        <div className="info-block">
          <div className="info-block__label">Economic Impact</div>
          <div className="impact-box">{di.economic_impact}</div>
        </div>
      );
    }
  };

  return (
    <div className="wrap results">
      <button className="results__back" onClick={onReset}>
        ← Analyse another image
      </button>

      {data.demo_mode && (
        <div className="demo-warning">
          ⚠️ <strong>Demo mode</strong> — Set CNN_DRIVE_ID &amp; HYBRID_DRIVE_ID env vars and redeploy for real predictions.
        </div>
      )}

      {/* Image + Verdict */}
      <div className="result-grid anim-fade-up">
        <div className="result-image">
          <img src={data.image_b64} alt="Analysed leaf" />
          <div className="result-image__label">
            {data.label} · {data.confidence}% ensemble confidence
          </div>
        </div>

        <div className="verdict">
          <div className="verdict__kicker">Detection Result</div>
          <div className="verdict__name">{data.label}</div>
          <div className="verdict__sci">{di.scientific_name}</div>

          <div className="verdict__conf-row">
            <div className="verdict__pct" style={{ color: confColor(data.confidence) }}>
              {data.confidence}%
            </div>
            <div className="verdict__bar-wrap">
              <div className="verdict__bar-label">Ensemble confidence</div>
              <div className="verdict__track">
                <div className="verdict__fill" style={{
                  width: `${data.confidence}%`,
                  background: confColor(data.confidence),
                }} />
              </div>
            </div>
          </div>

          <div className={`urgency-tag ${urgCls(di.urgency_level)}`}>
            ⚡ {di.urgency}
          </div>
        </div>
      </div>

      {/* Model trio */}
      <div className="model-trio anim-fade-up anim-fade-up--1">
        {[
          { label:'CNN Baseline',    key:'cnn',      ensemble:false },
          { label:'CNN-ViT Hybrid',  key:'hybrid',   ensemble:false },
          { label:'Ensemble',        key:'ensemble', ensemble:true  },
        ].map(({ label, key, ensemble }) => (
          <div key={key} className={`model-card ${ensemble ? 'model-card--ensemble' : ''}`}>
            <div className="model-card__name">{label}</div>
            <div className="model-card__val">{prob[key][data.label]}%</div>
            <div className="model-card__bars">
              {labs.map(l => (
                <div key={l} className="mini-bar">
                  <div className="mini-bar__label">
                    <span>{l.split(' ')[0]}</span>
                    <span>{prob[key][l]}%</span>
                  </div>
                  <div className="mini-bar__track">
                    <div
                      className={`mini-bar__fill ${l === data.label ? 'mini-bar__fill--primary' : ''}`}
                      style={{ width: `${prob[key][l]}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Info tabs */}
      <div className="info-panel anim-fade-up anim-fade-up--2">
        <div className="tab-rail">
          {TABS.map(t => (
            <button key={t}
              className={`tab-btn ${tab === t ? 'tab-btn--active' : ''}`}
              onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>
        <div className="tab-pane tab-pane--active">{tabContent()}</div>
      </div>
    </div>
  );
}

/* ─── Footer ──────────────────────────────────────────────── */
function Footer() {
  return (
    <footer className="footer">
      <div className="wrap footer__inner">
        <span>LuffaGuard · Luffa aegyptiaca Disease Detection</span>
        <a href="https://doi.org/10.17632/nym8bw5hr6.3" target="_blank" rel="noreferrer">
          Dataset: 10.17632/nym8bw5hr6.3
        </a>
      </div>
    </footer>
  );
}

/* ─── App ─────────────────────────────────────────────────── */
const VIEW = { UPLOAD:'upload', SCAN:'scan', RESULT:'result' };

export default function App() {
  const [view,    setView]    = useState(VIEW.UPLOAD);
  const [preview, setPreview] = useState(null);
  const [step,    setStep]    = useState(0);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);

  const handleFile = useCallback(async file => {
    setError(null); setStep(0);
    setPreview(URL.createObjectURL(file));
    setView(VIEW.SCAN);
    window.scrollTo({ top: 0 });

    const timer = setInterval(() =>
      setStep(s => s < SCAN_STEPS.length - 1 ? s + 1 : s), 700);

    try {
      const [data] = await Promise.all([
        predictDisease(file),
        new Promise(r => setTimeout(r, 3600)),
      ]);
      clearInterval(timer);
      setStep(SCAN_STEPS.length);
      setResult(data);
      setView(VIEW.RESULT);
    } catch (e) {
      clearInterval(timer);
      setError(e.message);
      setView(VIEW.UPLOAD);
    }
  }, []);

  const handleReset = useCallback(() => {
    setView(VIEW.UPLOAD); setResult(null);
    setPreview(null); setStep(0);
    window.scrollTo({ top: 0 });
  }, []);

  return (
    <>
      <Header />
      {error && (
        <div className="wrap" style={{ marginTop: 20 }}>
          <div className="demo-warning">⚠ {error}</div>
        </div>
      )}
      {view === VIEW.UPLOAD && <UploadView onFile={handleFile} />}
      {view === VIEW.SCAN   && <ScanView preview={preview} step={step} />}
      {view === VIEW.RESULT && <ResultView data={result} onReset={handleReset} />}
      <Footer />
    </>
  );
}
