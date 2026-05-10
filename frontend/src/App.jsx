import React, { useState, useRef, useCallback, useEffect } from 'react';
import { predictDisease } from './api.js';

const STEPS = [
  'Preprocessing — 224×224 resize, normalise to [0,1]',
  'CNN inference — 4-block convolutional network',
  'CNN-ViT Hybrid inference — transformer attention heads',
  'Ensemble soft voting — combining both model outputs',
  'Generating disease management report',
];
const TABS = ['Overview','Symptoms','Treatment','Prevention','Impact'];

/* ── Gauge ─────────────────────────────────────── */
function Gauge({ value, color }) {
  const r = 28, c = 2 * Math.PI * r;
  return (
    <svg width="68" height="68" viewBox="0 0 68 68">
      <circle cx="34" cy="34" r={r} fill="none" stroke="rgba(255,255,255,.06)" strokeWidth="4"/>
      <circle cx="34" cy="34" r={r} fill="none" stroke={color} strokeWidth="4"
        strokeDasharray={c} strokeDashoffset={c - (value/100)*c}
        strokeLinecap="round" transform="rotate(-90 34 34)"
        style={{transition:'stroke-dashoffset .9s cubic-bezier(.16,1,.3,1)'}}/>
      <text x="34" y="39" textAnchor="middle" fill={color} fontSize="12" fontFamily="Inter" fontWeight="700">{value}%</text>
    </svg>
  );
}

/* ── Modal ─────────────────────────────────────── */
function Modal({ title, sub, onClose, children }) {
  useEffect(() => {
    const fn = e => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [onClose]);
  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal__head">
          <div>
            <div className="modal__title">{title}</div>
            {sub && <div className="modal__sub">{sub}</div>}
          </div>
          <button className="modal__close" onClick={onClose}>✕</button>
        </div>
        <div className="modal__body">{children}</div>
      </div>
    </div>
  );
}

/* ── Toast ─────────────────────────────────────── */
function ToastStack({ items }) {
  return (
    <div className="toast-stack">
      {items.map(t => <div key={t.id} className="toast-item"><span>{t.icon}</span>{t.msg}</div>)}
    </div>
  );
}

/* ── Header ────────────────────────────────────── */
function Header() {
  return (
    <header className="header">
      <div className="wrap header__inner">
        <div className="logo">
          <div className="logo__icon">🌿</div>
          <div>
            <div className="logo__name">LuffaGuard</div>
            <div className="logo__tag">Leaf Disease Detection</div>
          </div>
        </div>
        <div className="status-pill">
          <span className="status-dot"/>
          Ensemble Active
        </div>
      </div>
    </header>
  );
}

/* ── Upload ─────────────────────────────────────── */
function UploadView({ onFile }) {
  const [drag, setDrag] = useState(false);
  const inp = useRef(); const cam = useRef();
  const go = f => f?.type.startsWith('image/') && onFile(f);
  return (
    <div className="wrap">
      <section className="hero">
        <div className="hero__eyebrow">AI-Powered Plant Diagnostics</div>
        <h1 className="hero__h1 anim anim-1">
          Detect disease in<br/>
          <em>Luffa aegyptiaca</em><br/>
          instantly
        </h1>
        <p className="hero__sub anim anim-2">
          Upload or photograph a leaf. Your trained CNN and CNN-ViT Hybrid ensemble
          diagnoses the disease and generates a full agronomic management report.
        </p>
        <div className="hero__pills anim anim-3">
          <span className="tag tag--neutral">Ensemble Model</span>
          <span className="tag tag--neutral">CNN · 99.19%</span>
          <span className="tag tag--neutral">CNN-ViT · 99.51%</span>
          <span className="tag tag--neutral">6,166 images</span>
          <span className="tag tag--neutral">2 disease classes</span>
        </div>
      </section>

      <div className="stats-bar anim anim-4">
        {[
          { icon:'🤖', val:'99.51%', lbl:'Best model accuracy' },
          { icon:'🗂',  val:'6,166',  lbl:'Training images' },
          { icon:'🔬', val:'2',       lbl:'Disease classes detected' },
        ].map(s => (
          <div className="stat-card" key={s.lbl}>
            <div className="stat-icon">{s.icon}</div>
            <div>
              <div className="stat-val">{s.val}</div>
              <div className="stat-lbl">{s.lbl}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="upload-wrap">
        <div
          className={`drop-zone ${drag ? 'drop-zone--drag' : ''}`}
          onClick={() => inp.current.click()}
          onDragEnter={e=>{e.preventDefault();setDrag(true)}}
          onDragOver={e=>{e.preventDefault();setDrag(true)}}
          onDragLeave={()=>setDrag(false)}
          onDrop={e=>{e.preventDefault();setDrag(false);go(e.dataTransfer.files[0])}}
        >
          <span className="dz-corner dz-corner--tl"/><span className="dz-corner dz-corner--tr"/>
          <span className="dz-corner dz-corner--bl"/><span className="dz-corner dz-corner--br"/>
          <span className="dz-icon">{drag ? '📂' : '🍃'}</span>
          <h3 className="dz-title">{drag ? 'Release to analyse' : 'Drop a leaf image here'}</h3>
          <p className="dz-sub">Drag & drop a <strong>Luffa aegyptiaca</strong> leaf photo · JPG, PNG, WebP</p>
          <div className="dz-actions">
            <button className="btn btn--primary" onClick={e=>{e.stopPropagation();inp.current.click()}}>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M8 2v9M4 7l4-4 4 4"/><path d="M2 13h12"/></svg>
              Upload Photo
            </button>
            <button className="btn btn--secondary btn--cam" onClick={e=>{e.stopPropagation();cam.current.click()}}>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><path d="M1 5.5A1.5 1.5 0 012.5 4h1l1-2h7l1 2h1A1.5 1.5 0 0115 5.5v7A1.5 1.5 0 0113.5 14h-11A1.5 1.5 0 011 12.5v-7z"/><circle cx="8" cy="9" r="2.5"/></svg>
              Take Photo
            </button>
          </div>
          <p className="dz-hint">Tip: use the rear camera for best image quality</p>
          <input ref={inp} type="file" accept="image/*" style={{display:'none'}} onChange={e=>go(e.target.files[0])}/>
          <input ref={cam} type="file" accept="image/*" capture="environment" style={{display:'none'}} onChange={e=>go(e.target.files[0])}/>
        </div>
      </div>
    </div>
  );
}

/* ── Scan ───────────────────────────────────────── */
function ScanView({ preview, step }) {
  return (
    <div className="wrap scan-wrap">
      <div className="scan-card">
        <div className="scan-img">
          {preview && <img src={preview} alt="leaf"/>}
          <div className="scan-beam"/>
          <div className="scan-grid"/>
          <span className="scan-cross sc-tl"/><span className="scan-cross sc-tr"/>
          <span className="scan-cross sc-bl"/><span className="scan-cross sc-br"/>
        </div>
        <div className="scan-info">
          <div className="scan-label">Analysing specimen</div>
          <div className="scan-title">Running disease detection</div>
          <div className="steps">
            {STEPS.map((txt, i) => {
              const s = i < step ? 'step--done' : i === step ? 'step--active' : '';
              return (
                <div key={i} className={`step ${s}`}>
                  <div className="step__dot">{i < step ? '✓' : i+1}</div>
                  <div className="step__txt">{txt}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Results ────────────────────────────────────── */
function ResultView({ data, onReset, addToast }) {
  const [tab,   setTab]   = useState('Overview');
  const [modal, setModal] = useState(null);
  const di   = data.disease_info;
  const prob = data.probabilities;
  const labs = Object.keys(prob.ensemble);

  const pct = c => c >= 90 ? 'var(--blue-l)' : c >= 75 ? 'var(--amber)' : 'var(--rose-l)';
  const urgCls = u => u === 'high' ? 'tag--rose' : u === 'moderate' ? 'tag--amber' : 'tag--green';
  const stripe = data.label === 'Mosaic Disease' ? 'verdict__stripe--mosaic' : 'verdict__stripe--insect';

  const download = () => {
    const txt = [
      'LuffaGuard Disease Report','',
      `Detection: ${data.label}`,`Confidence: ${data.confidence}%`,
      `Scientific Name: ${di.scientific_name}`,`Urgency: ${di.urgency}`,'',
      `CNN:      ${prob.cnn[data.label]}%`,
      `Hybrid:   ${prob.hybrid[data.label]}%`,
      `Ensemble: ${prob.ensemble[data.label]}%`,'',
      'OVERVIEW', di.overview,'',
      'SYMPTOMS', ...di.symptoms,'',
      'TREATMENT', ...di.treatment,'',
      'PREVENTION', ...di.prevention,'',
      'ECONOMIC IMPACT', di.economic_impact,
    ].join('\n');
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob([txt],{type:'text/plain'})),
      download: `LuffaGuard_${data.label.replace(' ','_')}.txt`,
    });
    a.click();
    addToast({ icon:'📄', msg:'Report downloaded' });
  };

  const tabBody = () => {
    switch(tab) {
      case 'Overview': return <>
        <div className="sec-title">Disease Overview</div>
        <p className="prose">{di.overview}</p>
        <div className="sec-title">Causes & Vectors</div>
        <ul className="tlist">{di.causes.map((c,i)=><li key={i}><span className="bullet bullet--blue"/><span>{c}</span></li>)}</ul>
      </>;
      case 'Symptoms': return <>
        <div className="sec-title">Observed Symptoms</div>
        <ul className="tlist">{di.symptoms.map((s,i)=><li key={i}><span className="bullet bullet--amber"/><span>{s}</span></li>)}</ul>
      </>;
      case 'Treatment': return <>
        <div className="sec-title">Treatment Protocol</div>
        <ul className="tlist">{di.treatment.map((t,i)=><li key={i}><span className="bullet bullet--green"/><span>{t}</span></li>)}</ul>
      </>;
      case 'Prevention': return <>
        <div className="sec-title">Prevention Measures</div>
        <ul className="tlist">{di.prevention.map((p,i)=><li key={i}><span className="bullet"/><span>{p}</span></li>)}</ul>
      </>;
      case 'Impact': return <>
        <div className="sec-title">Economic Impact</div>
        <div className="callout">{di.economic_impact}</div>
      </>;
    }
  };

  return (
    <div className="wrap results-wrap">
      <div className="results-bar">
        <button className="btn btn--secondary btn--sm" onClick={onReset}>← New Analysis</button>
        <button className="btn btn--secondary btn--sm" onClick={download}>
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M8 2v9M4 11l4 4 4-4"/><path d="M2 14h12"/></svg>
          Download Report
        </button>
      </div>

      {data.demo_mode && <div className="demo-warn">⚠ Demo mode — set CNN_DRIVE_ID &amp; HYBRID_DRIVE_ID env vars for real predictions.</div>}

      {/* Verdict */}
      <div className="verdict anim">
        <div className={`verdict__stripe ${stripe}`}/>
        <img className="verdict__leaf" src={data.image_b64} alt="leaf"/>
        <div className="verdict__body">
          <div className="verdict__kicker">Detection Result</div>
          <div className="verdict__name">{data.label}</div>
          <div className="verdict__sci">{di.scientific_name}</div>
          <div className="verdict__tags">
            <span className={`tag ${urgCls(di.urgency_level)}`}>⚡ {di.urgency}</span>
            <span className="tag tag--neutral">{data.label === 'Mosaic Disease' ? 'Viral' : 'Pest'} damage</span>
            <span className="tag tag--neutral">{data.confidence >= 95 ? 'High' : data.confidence >= 80 ? 'Moderate' : 'Low'} confidence</span>
          </div>
        </div>
        <div className="verdict__conf">
          <div className="verdict__pct" style={{color:pct(data.confidence)}}>{data.confidence}%</div>
          <div className="verdict__clbl">Ensemble<br/>confidence</div>
        </div>
      </div>

      {/* Image + model cards */}
      <div className="res-grid anim anim-1">
        <div className="res-img-card">
          <img src={data.image_b64} alt="leaf"/>
          <div className="res-img-card__footer">Analysed · {new Date().toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'})}</div>
        </div>
        <div className="model-trio">
          {[
            {label:'CNN Baseline',   key:'cnn',      color:'var(--cyan)',     best:false},
            {label:'CNN-ViT Hybrid', key:'hybrid',   color:'var(--violet)',   best:false},
            {label:'Ensemble',       key:'ensemble', color:'var(--blue-l)',   best:true},
          ].map(({label,key,color,best})=>(
            <div key={key} className={`mc${best?' mc--best':''}`}>
              <Gauge value={Math.round(prob[key][data.label])} color={color}/>
              <div style={{flex:1}}>
                <div className="mc__name">{label}{best&&<span className="best-badge">BEST</span>}</div>
                <div className="mc__bars">
                  {labs.map(l=>(
                    <div key={l} className="mbar">
                      <div className="mbar__lbl">{l}</div>
                      <div className="mbar__track"><div className={`mbar__fill ${l===data.label?'mbar__fill--hi':'mbar__fill--lo'}`} style={{width:`${prob[key][l]}%`}}/></div>
                      <div className="mbar__val">{prob[key][l]}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Info cards */}
      <div className="info-cards anim anim-2">
        <div className="ic ic--ov">
          <div className="ic__head"><div className="ic__title"><span className="ic__emoji">🔎</span>Overview</div></div>
          <div className="ic__body">{di.overview.slice(0,165)}…</div>
        </div>
        <div className="ic ic--sy">
          <div className="ic__head"><div className="ic__title"><span className="ic__emoji">🌡</span>Key Symptoms</div><button className="view-all" onClick={()=>setModal('symptoms')}>View all →</button></div>
          <ul className="ic__list">{di.symptoms.slice(0,3).map((s,i)=><li key={i}>{s}</li>)}</ul>
        </div>
        <div className="ic ic--tr">
          <div className="ic__head"><div className="ic__title"><span className="ic__emoji">💊</span>Treatment</div><button className="view-all" onClick={()=>setModal('treatment')}>Full protocol →</button></div>
          <ul className="ic__list">{di.treatment.slice(0,3).map((t,i)=><li key={i}>{t}</li>)}</ul>
        </div>
        <div className="ic ic--im">
          <div className="ic__head"><div className="ic__title"><span className="ic__emoji">📉</span>Economic Impact</div></div>
          <div className="ic__body">{di.economic_impact.slice(0,165)}…</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="detail-panel anim anim-3">
        <div className="tab-rail">
          {TABS.map(t=><button key={t} className={`tab-btn${tab===t?' tab-btn--active':''}`} onClick={()=>setTab(t)}>{t}</button>)}
        </div>
        <div className="tab-pane tab-pane--active">{tabBody()}</div>
      </div>

      {/* Modals */}
      {modal==='symptoms'&&(
        <Modal title="All Symptoms" sub={`${data.label} — ${di.scientific_name}`} onClose={()=>setModal(null)}>
          <div className="sec-title">Symptoms ({di.symptoms.length})</div>
          {di.symptoms.map((s,i)=><div key={i} className="modal__row"><div className="modal__num">{i+1}</div><div className="modal__txt">{s}</div></div>)}
        </Modal>
      )}
      {modal==='treatment'&&(
        <Modal title="Full Treatment Protocol" sub={`Urgency: ${di.urgency}`} onClose={()=>setModal(null)}>
          <div className="sec-title">Treatment Steps</div>
          {di.treatment.map((t,i)=><div key={i} className="modal__row"><div className="modal__num">{i+1}</div><div className="modal__txt">{t}</div></div>)}
          <div className="sec-title" style={{marginTop:20}}>Prevention</div>
          {di.prevention.slice(0,4).map((p,i)=><div key={i} className="modal__row"><div className="modal__num" style={{background:'rgba(34,197,94,.08)',borderColor:'rgba(34,197,94,.2)',color:'var(--green)'}}>{i+1}</div><div className="modal__txt">{p}</div></div>)}
        </Modal>
      )}
    </div>
  );
}

/* ── App ────────────────────────────────────────── */
const VIEW = {UPLOAD:'upload',SCAN:'scan',RESULT:'result'};
export default function App() {
  const [view,    setView]    = useState(VIEW.UPLOAD);
  const [preview, setPreview] = useState(null);
  const [step,    setStep]    = useState(0);
  const [result,  setResult]  = useState(null);
  const [toasts,  setToasts]  = useState([]);

  const addToast = useCallback(({icon='✅',msg})=>{
    const id = Date.now();
    setToasts(t=>[...t,{id,icon,msg}]);
    setTimeout(()=>setToasts(t=>t.filter(x=>x.id!==id)),3500);
  },[]);

  const handleFile = useCallback(async file => {
    setStep(0); setPreview(URL.createObjectURL(file)); setView(VIEW.SCAN);
    window.scrollTo({top:0,behavior:'smooth'});
    const timer = setInterval(()=>setStep(s=>s<STEPS.length-1?s+1:s),700);
    try {
      const [data] = await Promise.all([predictDisease(file), new Promise(r=>setTimeout(r,3600))]);
      clearInterval(timer); setStep(STEPS.length); setResult(data); setView(VIEW.RESULT);
      addToast({icon:data.demo_mode?'⚠️':'✅',msg:data.demo_mode?'Demo mode — simulated result':`Detected: ${data.label} (${data.confidence}%)`});
    } catch(e) {
      clearInterval(timer); setView(VIEW.UPLOAD);
      addToast({icon:'❌',msg:e.message});
    }
  },[addToast]);

  const handleReset = useCallback(()=>{
    setView(VIEW.UPLOAD); setResult(null); setPreview(null); setStep(0);
    window.scrollTo({top:0,behavior:'smooth'});
  },[]);

  return (
    <>
      <Header/>
      {view===VIEW.UPLOAD && <UploadView onFile={handleFile}/>}
      {view===VIEW.SCAN   && <ScanView preview={preview} step={step}/>}
      {view===VIEW.RESULT && <ResultView data={result} onReset={handleReset} addToast={addToast}/>}
      <footer className="footer">
        <div className="wrap footer__inner">
          <span>LuffaGuard · Luffa aegyptiaca Disease Detection</span>
          <a href="https://doi.org/10.17632/nym8bw5hr6.3" target="_blank" rel="noreferrer">Dataset: 10.17632/nym8bw5hr6.3</a>
        </div>
      </footer>
      <ToastStack items={toasts}/>
    </>
  );
}
