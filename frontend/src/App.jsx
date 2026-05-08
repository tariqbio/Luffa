import React, { useState, useRef, useCallback } from 'react';
import { predictDisease } from './api.js';

/* ─── tiny helpers ───────────────────────────────────────────────────── */
const s = (obj) => Object.entries(obj).map(([k,v])=>`${k}:${v}`).join(';');

const SCAN_STEPS = [
  'Preprocessing — 224×224 resize + normalise',
  'CNN inference — 4-block convolutional network',
  'CNN-ViT Hybrid inference — transformer attention',
  'Ensemble soft voting — averaging probabilities',
  'Building disease report',
];

const TABS = ['Overview','Symptoms','Treatment','Prevention','Impact'];

/* ─── sub-components ─────────────────────────────────────────────────── */

function Header() {
  return (
    <header style={s({
      borderBottom:'1px solid rgba(74,222,128,.10)',
      padding:'18px 0',
    })}>
      <div style={s({maxWidth:'960px',margin:'0 auto',padding:'0 24px',
                      display:'flex',alignItems:'center',justifyContent:'space-between'})}>
        <div style={s({display:'flex',alignItems:'center',gap:'12px'})}>
          <div style={s({width:'36px',height:'36px',borderRadius:'10px',
                          background:'var(--bg3)',border:'1px solid var(--border2)',
                          display:'flex',alignItems:'center',justifyContent:'center',fontSize:'20px'})}>
            🌿
          </div>
          <div>
            <div style={s({fontFamily:'var(--serif)',fontSize:'20px',color:'var(--green)',letterSpacing:'-0.3px'})}>
              LuffaGuard
            </div>
            <div style={s({fontSize:'10px',color:'var(--text3)',textTransform:'uppercase',letterSpacing:'.08em'})}>
              Leaf Disease Detection
            </div>
          </div>
        </div>
        <div style={s({
          fontFamily:'var(--mono)',fontSize:'11px',color:'var(--text3)',
          border:'1px solid var(--border)',borderRadius:'20px',padding:'5px 12px',
          display:'flex',alignItems:'center',gap:'6px'
        })}>
          <span style={s({width:'6px',height:'6px',borderRadius:'50%',
                           background:'var(--green-d)',display:'inline-block',
                           animation:'pulse 2s infinite'})} />
          CNN + CNN-ViT Hybrid
          <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}`}</style>
        </div>
      </div>
    </header>
  );
}

function UploadZone({ onFile }) {
  const [drag, setDrag] = useState(false);
  const inp = useRef();
  const handle = f => { if (f && f.type.startsWith('image/')) onFile(f); };

  return (
    <div style={s({maxWidth:'960px',margin:'0 auto',padding:'0 24px'})}>
      {/* hero */}
      <div style={s({padding:'52px 0 44px',textAlign:'center'})}>
        <h1 style={s({fontFamily:'var(--serif)',fontSize:'clamp(32px,5vw,54px)',lineHeight:'1.1'})}>
          Diagnose <em style={s({color:'var(--green)',fontStyle:'italic'})}>Luffa aegyptiaca</em>
          <br />leaf disease instantly
        </h1>
        <p style={s({marginTop:'14px',color:'var(--text2)',fontSize:'15px',fontWeight:'300',
                      maxWidth:'480px',margin:'14px auto 0',lineHeight:'1.75'})}>
          Upload a leaf photograph. The ensemble of your trained CNN and CNN-ViT Hybrid
          models analyses it and returns a full disease management report.
        </p>
        <div style={s({display:'flex',gap:'8px',justifyContent:'center',flexWrap:'wrap',marginTop:'22px'})}>
          {['CNN 99.19%','CNN-ViT 99.51%','Ensemble · Soft Voting','6,166 images'].map(t=>(
            <span key={t} style={s({
              fontFamily:'var(--mono)',fontSize:'11px',padding:'5px 14px',
              borderRadius:'20px',border:'1px solid var(--border2)',
              color:'var(--green-d)',background:'var(--bg3)'
            })}>{t}</span>
          ))}
        </div>
      </div>

      {/* drop zone */}
      <div
        onClick={() => inp.current.click()}
        onDragEnter={e=>{e.preventDefault();setDrag(true)}}
        onDragOver={e=>{e.preventDefault();setDrag(true)}}
        onDragLeave={()=>setDrag(false)}
        onDrop={e=>{e.preventDefault();setDrag(false);handle(e.dataTransfer.files[0])}}
        style={s({
          border:`1.5px dashed ${drag?'var(--green-d)':'var(--border2)'}`,
          borderRadius:'20px',padding:'60px 32px',textAlign:'center',cursor:'pointer',
          background:drag?'var(--bg3)':'var(--bg2)',transition:'all .2s',
          position:'relative',overflow:'hidden',marginBottom:'64px',
        })}
      >
        <div style={s({fontSize:'52px',marginBottom:'14px'})}>🍃</div>
        <h3 style={s({fontFamily:'var(--serif)',fontSize:'22px',color:'var(--text)',marginBottom:'8px'})}>
          Drop a Luffa leaf image here
        </h3>
        <p style={s({fontSize:'14px',color:'var(--text3)'})}> Supports JPG, PNG, WebP</p>
        <div style={s({
          display:'inline-block',marginTop:'18px',
          background:'var(--bg3)',border:'1px solid var(--border2)',borderRadius:'10px',
          padding:'10px 24px',fontSize:'14px',fontWeight:'500',color:'var(--green)',
        })}>Choose Image</div>
        <input ref={inp} type="file" accept="image/*" style={s({display:'none'})}
          onChange={e=>handle(e.target.files[0])} />
      </div>
    </div>
  );
}

function ScanView({ preview, step }) {
  return (
    <div style={s({maxWidth:'960px',margin:'0 auto',padding:'0 24px 64px'})}>
      <div style={s({
        background:'var(--bg4)',border:'1px solid var(--border)',borderRadius:'20px',
        overflow:'hidden',display:'flex',minHeight:'300px',
      })}>
        {/* image */}
        <div style={s({width:'260px',flexShrink:'0',position:'relative',background:'#050905'})}>
          {preview && <img src={preview} alt="" style={s({width:'100%',height:'100%',objectFit:'cover',display:'block'})} />}
          {/* scan beam */}
          <div style={s({
            position:'absolute',left:'0',right:'0',height:'3px',
            background:'linear-gradient(90deg,transparent,var(--green),transparent)',
            boxShadow:'0 0 18px var(--green)',
            top:`${(step/SCAN_STEPS.length)*100}%`,
            transition:'top .6s ease',opacity:step<SCAN_STEPS.length?1:0,
          })} />
        </div>

        {/* steps */}
        <div style={s({flex:'1',padding:'32px',display:'flex',flexDirection:'column',justifyContent:'center'})}>
          <div style={s({fontFamily:'var(--mono)',fontSize:'12px',color:'var(--text3)',marginBottom:'6px'})}>
            ANALYSING IMAGE …
          </div>
          <div style={s({fontFamily:'var(--serif)',fontSize:'22px',marginBottom:'22px'})}>
            Running disease detection
          </div>
          {SCAN_STEPS.map((txt, i) => {
            const done   = i < step;
            const active = i === step;
            return (
              <div key={i} style={s({
                display:'flex',alignItems:'center',gap:'12px',
                fontSize:'14px',marginBottom:'12px',
                color: done?'var(--text2)': active?'var(--green)':'var(--text3)',
                transition:'color .3s',
              })}>
                <div style={s({
                  width:'22px',height:'22px',borderRadius:'50%',flexShrink:'0',
                  border:`1.5px solid ${done?'var(--green-d)':active?'var(--green)':'var(--text3)'}`,
                  display:'flex',alignItems:'center',justifyContent:'center',
                  fontSize:'10px',fontFamily:'var(--mono)',
                  background: done?'var(--bg3)':'transparent',
                  color: done?'var(--green-d)':'inherit',
                })}>
                  {done ? '✓' : i+1}
                </div>
                {txt}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ResultView({ data, onReset }) {
  const [tab, setTab] = useState('Overview');
  const di = data.disease_info;
  const probs = data.probabilities;
  const labels = Object.keys(probs.ensemble);

  const confColor = c => c >= 90 ? 'var(--green)' : c >= 70 ? 'var(--amber)' : 'var(--red)';

  const tabContent = () => {
    switch (tab) {
      case 'Overview': return (
        <div>
          <Label>Disease Overview</Label>
          <p style={s({fontSize:'14px',color:'var(--text2)',lineHeight:'1.8',marginBottom:'20px'})}>{di.overview}</p>
          <Label>Causes / Vectors</Label>
          <List items={di.causes} icon="→" />
        </div>
      );
      case 'Symptoms': return (
        <div><Label>Observed Symptoms</Label><List items={di.symptoms} icon="→" /></div>
      );
      case 'Treatment': return (
        <div><Label>Treatment Protocol</Label><List items={di.treatment} icon="✦" iconColor="var(--green-d)" /></div>
      );
      case 'Prevention': return (
        <div><Label>Prevention Measures</Label><List items={di.prevention} icon="◉" iconColor="var(--amber)" /></div>
      );
      case 'Impact': return (
        <div>
          <Label>Economic Impact</Label>
          <div style={s({
            background:'var(--bg3)',border:'1px solid var(--border)',borderLeft:'3px solid var(--green-d)',
            borderRadius:'12px',padding:'16px 20px',fontSize:'14px',color:'var(--text2)',lineHeight:'1.75',
          })}>{di.economic_impact}</div>
        </div>
      );
    }
  };

  return (
    <div style={s({maxWidth:'960px',margin:'0 auto',padding:'0 24px 72px'})}>
      {data.demo_mode && (
        <div style={s({
          background:'rgba(251,191,36,.07)',border:'1px solid rgba(251,191,36,.25)',
          borderRadius:'12px',padding:'12px 18px',marginBottom:'20px',
          fontSize:'13px',color:'#d4a010',display:'flex',alignItems:'center',gap:'8px',
        })}>
          ⚠️ <strong>Demo mode</strong> — set CNN_DRIVE_ID and HYBRID_DRIVE_ID env vars and redeploy for real predictions.
        </div>
      )}

      <button onClick={onReset} style={s({
        display:'inline-flex',alignItems:'center',gap:'8px',
        padding:'11px 22px',borderRadius:'12px',fontSize:'14px',fontWeight:'500',
        background:'var(--bg3)',border:'1px solid var(--border2)',color:'var(--green)',
        marginBottom:'22px',transition:'all .15s',cursor:'pointer',
      })}>← Analyse another image</button>

      {/* top row */}
      <div style={s({display:'grid',gridTemplateColumns:'260px 1fr',gap:'18px',marginBottom:'18px'})}>

        {/* image */}
        <div style={s({background:'var(--bg4)',border:'1px solid var(--border)',borderRadius:'16px',overflow:'hidden'})}>
          <img src={data.image_b64} alt="Analysed leaf" style={s({width:'100%',display:'block'})} />
          <div style={s({padding:'10px 14px',fontFamily:'var(--mono)',fontSize:'11px',
                          color:'var(--text3)',textAlign:'center',borderTop:'1px solid var(--border)'})}>
            {data.label} · {data.confidence}% confidence
          </div>
        </div>

        {/* verdict */}
        <div style={s({background:'var(--bg4)',border:'1px solid var(--border)',borderRadius:'16px',padding:'26px'})}>
          <Label>Detection Result</Label>
          <div style={s({fontFamily:'var(--serif)',fontSize:'30px',lineHeight:'1.2',marginBottom:'4px'})}>
            {data.label}
          </div>
          <div style={s({fontSize:'13px',color:'var(--text3)',fontStyle:'italic',marginBottom:'18px'})}>
            {di.scientific_name}
          </div>

          {/* confidence */}
          <div style={s({display:'flex',alignItems:'center',gap:'16px',marginBottom:'18px'})}>
            <div style={s({fontFamily:'var(--mono)',fontSize:'38px',fontWeight:'500',
                            color:confColor(data.confidence)})}>
              {data.confidence}%
            </div>
            <div style={s({flex:'1'})}>
              <div style={s({fontSize:'11px',color:'var(--text3)',marginBottom:'6px'})}>Ensemble confidence</div>
              <div style={s({height:'6px',background:'var(--bg3)',borderRadius:'3px',overflow:'hidden'})}>
                <div style={s({
                  height:'100%',borderRadius:'3px',
                  width:`${data.confidence}%`,
                  background:confColor(data.confidence),
                  transition:'width .8s cubic-bezier(.23,1,.32,1)',
                })} />
              </div>
            </div>
          </div>

          <div style={s({
            display:'inline-flex',alignItems:'center',gap:'6px',
            padding:'6px 14px',borderRadius:'8px',fontFamily:'var(--mono)',fontSize:'12px',
            background:'rgba(251,191,36,.09)',color:'var(--amber)',
            border:'1px solid rgba(251,191,36,.22)',
          })}>⚡ {di.urgency}</div>
        </div>
      </div>

      {/* model trio */}
      <div style={s({display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'12px',marginBottom:'18px'})}>
        {[
          {label:'CNN Baseline',   key:'cnn',    highlight:false},
          {label:'CNN-ViT Hybrid', key:'hybrid', highlight:false},
          {label:'Ensemble ✦',     key:'ensemble',highlight:true},
        ].map(({label:ml,key,highlight})=>(
          <div key={key} style={s({
            background:'var(--bg3)',border:`1px solid ${highlight?'var(--border2)':'var(--border)'}`,
            borderRadius:'12px',padding:'14px',
          })}>
            <div style={s({fontFamily:'var(--mono)',fontSize:'10px',color:'var(--text3)',
                            textTransform:'uppercase',letterSpacing:'.08em',marginBottom:'6px'})}>
              {ml}
            </div>
            <div style={s({fontFamily:'var(--mono)',fontSize:'24px',fontWeight:'500',
                            color:highlight?'var(--green)':'var(--text)'})}>
              {probs[key][data.label]}%
            </div>
            <div style={s({display:'flex',justifyContent:'space-between',marginTop:'4px'})}>
              {labels.map(l=>(
                <span key={l} style={s({fontFamily:'var(--mono)',fontSize:'10px',color:'var(--text3)'})}>
                  {l.split(' ')[0]}: {probs[key][l]}%
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* info tabs */}
      <div style={s({background:'var(--bg4)',border:'1px solid var(--border)',borderRadius:'16px',overflow:'hidden'})}>
        <div style={s({display:'flex',borderBottom:'1px solid var(--border)',overflowX:'auto'})}>
          {TABS.map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={s({
              padding:'13px 20px',fontSize:'13px',fontWeight:'500',
              color:tab===t?'var(--green)':'var(--text3)',
              background:'none',border:'none',borderBottom:`2px solid ${tab===t?'var(--green)':'transparent'}`,
              cursor:'pointer',whiteSpace:'nowrap',marginBottom:'-1px',transition:'color .15s',
            })}>{t}</button>
          ))}
        </div>
        <div style={s({padding:'26px'})}>{tabContent()}</div>
      </div>
    </div>
  );
}

/* tiny shared components */
function Label({ children }) {
  return (
    <div style={s({fontFamily:'var(--mono)',fontSize:'11px',color:'var(--text3)',
                    textTransform:'uppercase',letterSpacing:'.1em',marginBottom:'12px'})}>
      {children}
    </div>
  );
}
function List({ items, icon='→', iconColor='var(--text3)' }) {
  return (
    <ul style={s({listStyle:'none',display:'flex',flexDirection:'column',gap:'8px'})}>
      {items.map((it,i)=>(
        <li key={i} style={s({fontSize:'14px',color:'var(--text2)',lineHeight:'1.65',
                                paddingLeft:'20px',position:'relative'})}>
          <span style={s({position:'absolute',left:'0',top:'1px',color:iconColor,fontSize:'11px'})}>
            {icon}
          </span>
          {it}
        </li>
      ))}
    </ul>
  );
}

/* ─── main App ───────────────────────────────────────────────────────── */
const VIEW = { UPLOAD: 'upload', SCAN: 'scan', RESULT: 'result' };

export default function App() {
  const [view,    setView]    = useState(VIEW.UPLOAD);
  const [preview, setPreview] = useState(null);
  const [step,    setStep]    = useState(0);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);

  const handleFile = useCallback(async (file) => {
    setError(null);
    setStep(0);
    setPreview(URL.createObjectURL(file));
    setView(VIEW.SCAN);

    // Animate steps
    const stepInterval = setInterval(() => {
      setStep(s => s < SCAN_STEPS.length - 1 ? s + 1 : s);
    }, 700);

    try {
      const data = await predictDisease(file);
      // Wait for animation to finish (at least 3.5s of steps)
      await new Promise(r => setTimeout(r, Math.max(0, 3500)));
      clearInterval(stepInterval);
      setStep(SCAN_STEPS.length);
      setResult(data);
      setView(VIEW.RESULT);
    } catch (e) {
      clearInterval(stepInterval);
      setError(e.message);
      setView(VIEW.UPLOAD);
    }
  }, []);

  const handleReset = useCallback(() => {
    setView(VIEW.UPLOAD);
    setResult(null);
    setPreview(null);
    setStep(0);
  }, []);

  return (
    <div style={s({minHeight:'100vh',display:'flex',flexDirection:'column'})}>
      <Header />

      {error && (
        <div style={s({
          maxWidth:'960px',margin:'16px auto 0',padding:'0 24px',width:'100%',
        })}>
          <div style={s({
            background:'rgba(248,113,113,.08)',border:'1px solid rgba(248,113,113,.25)',
            borderRadius:'12px',padding:'12px 18px',fontSize:'13px',color:'var(--red)',
          })}>⚠ {error}</div>
        </div>
      )}

      {view === VIEW.UPLOAD && <UploadZone onFile={handleFile} />}
      {view === VIEW.SCAN   && <ScanView preview={preview} step={step} />}
      {view === VIEW.RESULT && <ResultView data={result} onReset={handleReset} />}

      <footer style={s({
        borderTop:'1px solid var(--border)',padding:'22px 0',marginTop:'auto',
      })}>
        <div style={s({
          maxWidth:'960px',margin:'0 auto',padding:'0 24px',
          display:'flex',alignItems:'center',justifyContent:'space-between',
          fontSize:'12px',color:'var(--text3)',
        })}>
          <span>LuffaGuard · <em>Luffa aegyptiaca</em> Disease Detection</span>
          <a href="https://doi.org/10.17632/nym8bw5hr6.3" target="_blank"
             style={s({color:'var(--text3)'})} rel="noreferrer">
            Dataset: 10.17632/nym8bw5hr6.3
          </a>
        </div>
      </footer>
    </div>
  );
}
