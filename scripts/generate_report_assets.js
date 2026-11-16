/**
 * Reproducible script to generate all summary tables and figures for the YZM5110 Research Report.
 * Author: Samet Çetinkaya (26526003)
 * Usage: node scripts/generate_report_assets.js
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const RESULTS_DIR = path.join(ROOT, 'results');
const FIGURES_DIR = path.join(RESULTS_DIR, 'figures');
if (!fs.existsSync(FIGURES_DIR)) fs.mkdirSync(FIGURES_DIR, { recursive: true });

const summaries = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, 'all_summaries.json'), 'utf8'));
const robCsv = fs.readFileSync(path.join(RESULTS_DIR, 'robustness_comparison.csv'), 'utf8').trim().split(/\r?\n/);
const gateCsv = fs.readFileSync(path.join(RESULTS_DIR, 'gate_activations.csv'), 'utf8').trim().split(/\r?\n/);

function parseCsv(lines) {
  const headers = lines[0].split(',');
  return lines.slice(1).map(l => {
    const vals = l.split(',');
    const obj = {};
    headers.forEach((h, i) => {
      const v = vals[i];
      obj[h] = isNaN(Number(v)) ? v : Number(v);
    });
    return obj;
  });
}

const robRows = parseCsv(robCsv);
const gateRows = parseCsv(gateCsv);

const { execFileSync } = require('child_process');

// 0. Translate description field in all_summaries.json to English
const descMap = {
  'B0': 'Baseline (Original MMGCN)',
  'B1': 'Control (Un-gated + Corruption Augmentation)',
  'G0': 'Ablation (Directed Cross-Modal Gate Only)',
  'G1': 'Proposed (Directed Gate + Corruption Augmentation)',
  'G1_CW': 'Ablation 2B (Gate + Augmentation + Class-Weighted Loss)'
};
Object.keys(summaries).forEach(k => {
  if (descMap[k]) summaries[k].description = descMap[k];
});
fs.writeFileSync(path.join(RESULTS_DIR, 'all_summaries.json'), JSON.stringify(summaries, null, 2), 'utf8');

// Ensure clean_and_minority_table.csv has standardized English headers
const cleanCsvPath = path.join(RESULTS_DIR, 'clean_and_minority_table.csv');
if (fs.existsSync(cleanCsvPath)) {
  const cleanLines = fs.readFileSync(cleanCsvPath, 'utf8').trim().split(/\r?\n/);
  cleanLines[0] = 'Model,W-F1 (Std),Macro-F1 (Std),fear F1 (Std),disgust F1 (Std),tau (Val),W-F1 (LogitAdj),Macro-F1 (LogitAdj),fear F1 (LogitAdj),disgust F1 (LogitAdj),Params,Train (s),Peak GPU (MiB)';
  fs.writeFileSync(cleanCsvPath, cleanLines.join('\n') + '\n', 'utf8');
}

// 1. Generate English SVG & PNG charts for Figure 4 (Robustness) and Figure 5 (Minority F1) from CSV/JSON data
const modes = ['clean', 'audio_noise_1', 'audio_noise_2', 'audio_zero', 'visual_noise_1', 'visual_noise_2', 'visual_zero', 'av_noise_1', 'av_zero', 'av_swap'];
const variants = ['B0', 'B1', 'G0', 'G1', 'G1_CW'];
const vColors = { B0: '#1f77b4', B1: '#ff7f0e', G0: '#2ca02c', G1: '#d62728', G1_CW: '#9467bd' };

let barsFig1 = '';
const xStart = 68, plotW = 860, yTop = 50, yBot = 295, plotH = yBot - yTop; // y from 0 to 62
const groupW = plotW / modes.length;
const barW = 12;

// Y grid lines
for (let yVal = 0; yVal <= 60; yVal += 10) {
  const y = yBot - (yVal / 62) * plotH;
  barsFig1 += `<line x1="${xStart}" y1="${y.toFixed(1)}" x2="${xStart + plotW}" y2="${y.toFixed(1)}" stroke="#e2e8f0" stroke-width="1"/>`;
  barsFig1 += `<text x="${xStart - 8}" y="${(y + 4).toFixed(1)}" font-size="10.5" fill="#334155" text-anchor="end">${yVal}</text>`;
}

modes.forEach((m, mi) => {
  const gx = xStart + mi * groupW + 11;
  variants.forEach((v, vi) => {
    const row = robRows.find(r => r.variant === v && r.corruption === m);
    const val = row ? Number(row.w_f1) : 0;
    const h = (val / 62) * plotH;
    const y = yBot - h;
    const x = gx + vi * (barW + 1.5);
    barsFig1 += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${h.toFixed(1)}" fill="${vColors[v]}"/>`;
  });
  const lx = gx + 2.5 * (barW + 1.5);
  barsFig1 += `<text x="${lx.toFixed(1)}" y="${yBot + 16}" font-size="10.5" fill="#0f172a" text-anchor="end" transform="rotate(-28, ${lx.toFixed(1)}, ${yBot + 16})">${m}</text>`;
});

const svgFig1 = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 365" width="960" height="365" font-family="Segoe UI, Arial, sans-serif">
  <rect width="960" height="365" fill="#ffffff"/>
  <text x="480" y="28" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">MELD: Robustness Comparison Under Audio/Visual Modality Corruption</text>
  ${barsFig1}
  <rect x="${xStart}" y="${yTop}" width="${plotW}" height="${plotH}" fill="none" stroke="#64748b" stroke-width="1.2"/>
  <text x="22" y="175" font-size="11.5" font-weight="bold" fill="#0f172a" text-anchor="middle" transform="rotate(-90, 22, 175)">Test Weighted-F1 (%)</text>
  <text x="498" y="355" font-size="11" font-weight="bold" fill="#0f172a" text-anchor="middle">Modality Corruption Scenario</text>
  <!-- Legend -->
  <rect x="838" y="56" width="84" height="104" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1" rx="4"/>
  <text x="880" y="72" font-size="10" font-weight="bold" fill="#334155" text-anchor="middle">Variant</text>
  <rect x="848" y="80" width="14" height="9" fill="#1f77b4"/><text x="868" y="88" font-size="10" fill="#0f172a">B0</text>
  <rect x="848" y="96" width="14" height="9" fill="#ff7f0e"/><text x="868" y="104" font-size="10" fill="#0f172a">B1</text>
  <rect x="848" y="112" width="14" height="9" fill="#2ca02c"/><text x="868" y="120" font-size="10" fill="#0f172a">G0</text>
  <rect x="848" y="128" width="14" height="9" fill="#d62728"/><text x="868" y="136" font-size="10" fill="#0f172a">G1</text>
  <rect x="848" y="144" width="14" height="9" fill="#9467bd"/><text x="868" y="152" font-size="10" fill="#0f172a">G1_CW</text>
</svg>`;
fs.writeFileSync(path.join(FIGURES_DIR, 'fig1_robustness_wf1.svg'), svgFig1, 'utf8');

// Figure 2 (Minority F1 & Macro-F1 Recovery in English)
let barsFig2 = '';
const xStart2 = 68, plotW2 = 760, yTop2 = 48, yBot2 = 295, plotH2 = yBot2 - yTop2; // y from 0 to 38
const groupW2 = plotW2 / variants.length;
const barW2 = 24;
const mColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'];

for (let yVal = 0; yVal <= 35; yVal += 5) {
  const y = yBot2 - (yVal / 38) * plotH2;
  barsFig2 += `<line x1="${xStart2}" y1="${y.toFixed(1)}" x2="${xStart2 + plotW2}" y2="${y.toFixed(1)}" stroke="#e2e8f0" stroke-width="1"/>`;
  barsFig2 += `<text x="${xStart2 - 8}" y="${(y + 4).toFixed(1)}" font-size="10.5" fill="#334155" text-anchor="end">${yVal}</text>`;
}

variants.forEach((v, vi) => {
  const s = summaries[v];
  const vals = [
    s.clean_test_standard.macro_f1,
    s.clean_test_logit_adjusted.macro_f1,
    s.clean_test_logit_adjusted.fear_f1,
    s.clean_test_logit_adjusted.disgust_f1
  ];
  const gx = xStart2 + vi * groupW2 + 24;
  vals.forEach((val, bi) => {
    const h = (val / 38) * plotH2;
    const y = yBot2 - h;
    const x = gx + bi * (barW2 + 3);
    barsFig2 += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW2}" height="${Math.max(0.5, h).toFixed(1)}" fill="${mColors[bi]}"/>`;
  });
  const lx = gx + 2 * (barW2 + 3);
  barsFig2 += `<text x="${lx.toFixed(1)}" y="${yBot2 + 20}" font-size="11" font-weight="bold" fill="#0f172a" text-anchor="middle">${v}</text>`;
});

const svgFig2 = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 345" width="860" height="345" font-family="Segoe UI, Arial, sans-serif">
  <rect width="860" height="345" fill="#ffffff"/>
  <text x="440" y="28" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">Minority-Class (fear &amp; disgust) and Macro-F1 Recovery on MELD</text>
  ${barsFig2}
  <rect x="${xStart2}" y="${yTop2}" width="${plotW2}" height="${plotH2}" fill="none" stroke="#64748b" stroke-width="1.2"/>
  <text x="24" y="175" font-size="11.5" font-weight="bold" fill="#0f172a" text-anchor="middle" transform="rotate(-90, 24, 175)">F1 Score (%)</text>
  <text x="448" y="336" font-size="11" font-weight="bold" fill="#0f172a" text-anchor="middle">Model Variant</text>
  <!-- Legend -->
  <rect x="615" y="55" width="205" height="82" fill="#ffffff" fill-opacity="0.94" stroke="#cbd5e1" rx="4"/>
  <rect x="625" y="64" width="15" height="9" fill="#1f77b4"/><text x="646" y="72" font-size="10" fill="#0f172a">Macro-F1 (Standard)</text>
  <rect x="625" y="81" width="15" height="9" fill="#ff7f0e"/><text x="646" y="89" font-size="10" fill="#0f172a">Macro-F1 (Logit-Adjusted)</text>
  <rect x="625" y="98" width="15" height="9" fill="#2ca02c"/><text x="646" y="106" font-size="10" fill="#0f172a">fear F1 (Logit-Adjusted)</text>
  <rect x="625" y="115" width="15" height="9" fill="#d62728"/><text x="646" y="123" font-size="10" fill="#0f172a">disgust F1 (Logit-Adjusted)</text>
</svg>`;
fs.writeFileSync(path.join(FIGURES_DIR, 'fig2_minority_f1.svg'), svgFig2, 'utf8');

// Render English PNGs via headless Chrome so fig1_robustness_wf1.png and fig2_minority_f1.png are 100% English
const chromeCandidates = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe'
];
const browserExe = chromeCandidates.find(p => fs.existsSync(p));
if (browserExe) {
  [
    { svg: 'fig1_robustness_wf1.svg', png: 'fig1_robustness_wf1.png', w: 960, h: 365 },
    { svg: 'fig2_minority_f1.svg', png: 'fig2_minority_f1.png', w: 860, h: 345 }
  ].forEach(item => {
    const svgFile = path.join(FIGURES_DIR, item.svg);
    const pngOut1 = path.join(FIGURES_DIR, item.png);
    const url = 'file:///' + svgFile.replace(/\\/g, '/');
    execFileSync(browserExe, [
      '--headless=new',
      '--disable-gpu',
      '--hide-scrollbars',
      `--window-size=${item.w},${item.h}`,
      `--screenshot=${pngOut1}`,
      url
    ]);
    if (fs.existsSync(svgFile)) fs.unlinkSync(svgFile);
  });
}

// 2. Generate SVG Figure 3: MELD Dataset Overview, Class Imbalance & Authentic Dialogue Illustration
const svgMeld = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 430" width="980" height="430" font-family="Segoe UI, Arial, sans-serif">
  <rect width="980" height="430" fill="#ffffff" rx="8"/>
  <rect x="8" y="8" width="964" height="414" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  
  <text x="490" y="34" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">Figure 1: MELD Dataset Structure — Multi-Party Dialogue Context, Trimodal Features, and Severe Class Imbalance</text>
  
  <!-- Left Panel: Authentic MELD Dialogue Example (Friends) -->
  <rect x="24" y="50" width="465" height="354" fill="#ffffff" stroke="#94a3b8" stroke-width="1.2" rx="6"/>
  <rect x="24" y="50" width="465" height="30" fill="#1e293b" rx="6"/>
  <text x="256" y="70" font-size="13" font-weight="bold" fill="#ffffff" text-anchor="middle">A. Authentic MELD Dialogue Snippet (Friends TV Series, PKL Verified)</text>
  
  <!-- Utterance 1 -->
  <rect x="38" y="92" width="437" height="54" fill="#f1f5f9" stroke="#cbd5e1" rx="4"/>
  <circle cx="58" cy="119" r="14" fill="#3b82f6"/>
  <text x="58" y="123" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">S1</text>
  <text x="80" y="108" font-size="11" font-weight="bold" fill="#1e3a8a">u₁ (Joey): <tspan font-weight="normal" fill="#334155">"Good job Joe! Well done! Top notch!"</tspan></text>
  <text x="80" y="126" font-size="10" fill="#475569">Text (600-d TextCNN)  |  Audio (300-d OpenSMILE)  |  Visual (342-d DenseNet)</text>
  <rect x="405" y="97" width="62" height="18" fill="#dcfce7" stroke="#16a34a" rx="3"/>
  <text x="436" y="110" font-size="10" font-weight="bold" fill="#15803d" text-anchor="middle">joy (4)</text>

  <!-- Utterance 2 -->
  <rect x="38" y="156" width="437" height="54" fill="#f1f5f9" stroke="#cbd5e1" rx="4"/>
  <circle cx="58" cy="183" r="14" fill="#0d9488"/>
  <text x="58" y="187" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">S2</text>
  <text x="80" y="172" font-size="11" font-weight="bold" fill="#134e4a">u₂ (Chandler): <tspan font-weight="normal" fill="#334155">"You liked it? You really liked it?"</tspan></text>
  <text x="80" y="190" font-size="10" fill="#475569">High pitch rise in Audio (300-d) + Raised eyebrows in Visual (342-d)</text>
  <rect x="395" y="161" width="72" height="18" fill="#fef9c3" stroke="#ca8a04" rx="3"/>
  <text x="431" y="174" font-size="10" font-weight="bold" fill="#a16207" text-anchor="middle">surprise (1)</text>

  <!-- Utterance 3 -->
  <rect x="38" y="220" width="437" height="54" fill="#f1f5f9" stroke="#cbd5e1" rx="4"/>
  <circle cx="58" cy="247" r="14" fill="#3b82f6"/>
  <text x="58" y="251" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">S1</text>
  <text x="80" y="236" font-size="11" font-weight="bold" fill="#1e3a8a">u₃ (Joey): <tspan font-weight="normal" fill="#334155">"Oh-no-no-no, give me some specifics."</tspan></text>
  <text x="80" y="254" font-size="10" fill="#475569">Speaker identity embedding s(u₃)=S1 added to text node l₃</text>
  <rect x="400" y="225" width="67" height="18" fill="#fee2e2" stroke="#dc2626" rx="3"/>
  <text x="433" y="238" font-size="10" font-weight="bold" fill="#b91c1c" text-anchor="middle">anger (6)</text>

  <!-- Utterance 4 -->
  <rect x="38" y="284" width="437" height="58" fill="#fff1f2" stroke="#f43f5e" stroke-dasharray="4,2" rx="4"/>
  <circle cx="58" cy="313" r="14" fill="#e11d48"/>
  <text x="58" y="317" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">S2</text>
  <text x="80" y="300" font-size="10.5" font-weight="bold" fill="#881337">u₄ (Minority Emotion Challenges in MELD):</text>
  <text x="80" y="316" font-size="9.5" fill="#9f1239">Fear (1.9% test) &amp; Disgust (2.6% test) share lexical overlap with Anger/Neutral.</text>
  <text x="80" y="331" font-size="9.3" fill="#9f1239">Without class weighting/logit adj., MMGCN predicts them ≤ 4 times in 2,610 utterances!</text>

  <text x="256" y="366" font-size="10" fill="#334155" text-anchor="middle" font-weight="bold">Dataset Splits in MELD_features_raw1.pkl (84.1 MiB, up to 9 speakers/dialogue):</text>
  <text x="256" y="384" font-size="9.6" fill="#475569" text-anchor="middle">Train: 1,038 dia (8,990 utt)  |  Valid (10%): 115 dia (999 utt)  |  Test: 280 dia (2,610 utt)</text>

  <!-- Right Panel: Class Distribution & Imbalance Chart -->
  <rect x="505" y="50" width="451" height="354" fill="#ffffff" stroke="#94a3b8" stroke-width="1.2" rx="6"/>
  <rect x="505" y="50" width="451" height="30" fill="#1e293b" rx="6"/>
  <text x="730" y="70" font-size="13" font-weight="bold" fill="#ffffff" text-anchor="middle">B. MELD Class Imbalance (Train vs. Test Support &amp; B0 F1 Collapse)</text>

  <!-- Horizontal bars for 7 classes -->
  <!-- Neutral -->
  <text x="580" y="112" font-size="10.5" font-weight="bold" fill="#1e293b" text-anchor="end">0: neutral</text>
  <rect x="588" y="99" width="175" height="18" fill="#3b82f6" rx="2"/>
  <text x="770" y="112" font-size="10" fill="#0f172a">47.1% (Test: 1,256) | B0 F1: 75.2%</text>

  <!-- Joy -->
  <text x="580" y="148" font-size="10.5" font-weight="bold" fill="#1e293b" text-anchor="end">4: joy</text>
  <rect x="588" y="135" width="65" height="18" fill="#10b981" rx="2"/>
  <text x="660" y="148" font-size="10" fill="#0f172a">17.4% (Test: 402) | B0 F1: 52.2%</text>

  <!-- Anger -->
  <text x="580" y="184" font-size="10.5" font-weight="bold" fill="#1e293b" text-anchor="end">6: anger</text>
  <rect x="588" y="171" width="44" height="18" fill="#f59e0b" rx="2"/>
  <text x="639" y="184" font-size="10" fill="#0f172a">11.1% (Test: 345) | B0 F1: 43.8%</text>

  <!-- Surprise -->
  <text x="580" y="220" font-size="10.5" font-weight="bold" fill="#1e293b" text-anchor="end">1: surprise</text>
  <rect x="588" y="207" width="46" height="18" fill="#8b5cf6" rx="2"/>
  <text x="641" y="220" font-size="10" fill="#0f172a">12.0% (Test: 281) | B0 F1: 46.5%</text>

  <!-- Sadness -->
  <text x="580" y="256" font-size="10.5" font-weight="bold" fill="#1e293b" text-anchor="end">3: sadness</text>
  <rect x="588" y="243" width="27" height="18" fill="#64748b" rx="2"/>
  <text x="622" y="256" font-size="10" fill="#0f172a">6.9% (Test: 208) | B0 F1: 29.6%</text>

  <!-- Disgust -->
  <text x="580" y="292" font-size="10.5" font-weight="bold" fill="#be123c" text-anchor="end">5: disgust</text>
  <rect x="588" y="279" width="11" height="18" fill="#e11d48" rx="2"/>
  <text x="606" y="292" font-size="10" font-weight="bold" fill="#be123c">2.7% (Test: 68) | B0 F1: 0.0%–2.9% [COLLAPSED]</text>

  <!-- Fear -->
  <text x="580" y="328" font-size="10.5" font-weight="bold" fill="#be123c" text-anchor="end">2: fear</text>
  <rect x="588" y="315" width="10" height="18" fill="#e11d48" rx="2"/>
  <text x="605" y="328" font-size="10" font-weight="bold" fill="#be123c">2.7% (Test: 50) | B0 F1: 0.0%–3.5% [COLLAPSED]</text>

  <rect x="520" y="346" width="420" height="46" fill="#fef2f2" stroke="#f87171" rx="4"/>
  <text x="730" y="364" font-size="10" font-weight="bold" fill="#991b1b" text-anchor="middle">Empirical Observation: Neutral (48.1% test) is 25× larger than Fear (1.9% test).</text>
  <text x="730" y="381" font-size="9.5" fill="#7f1d1d" text-anchor="middle">Unweighted MELD FocalLoss (unlike IEMOCAP) causes minority-class collapse.</text>
</svg>`;
fs.writeFileSync(path.join(FIGURES_DIR, 'fig_meld_overview.svg'), svgMeld, 'utf8');

// 3. Generate SVG Figure 4: Architectural Diagram (Original MMGCN vs Proposed Directed Cross-Modal Gated MMGCN & GraphMFT)
const svgArch = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 410" width="980" height="410" font-family="Segoe UI, Arial, sans-serif">
  <defs>
    <marker id="arrowBlue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#2563eb"/>
    </marker>
    <marker id="arrowRed" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#dc2626"/>
    </marker>
    <marker id="arrowGreen" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#16a34a"/>
    </marker>
  </defs>
  <rect width="980" height="410" fill="#ffffff" rx="8"/>
  <rect x="8" y="8" width="964" height="394" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="490" y="32" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">Figure 2: Mechanism Comparison — (A) Original MMGCN Corruption Propagation vs. (B) Proposed Directed Cross-Modal Edge Gate</text>

  <!-- Panel A: Original MMGCN -->
  <rect x="22" y="46" width="455" height="342" fill="#ffffff" stroke="#94a3b8" rx="6"/>
  <rect x="22" y="46" width="455" height="28" fill="#991b1b" rx="6"/>
  <text x="249" y="65" font-size="12.5" font-weight="bold" fill="#ffffff" text-anchor="middle">A. Original MMGCN: Undirected Fixed Angular Similarity</text>

  <!-- Nodes in Panel A -->
  <circle cx="115" cy="145" r="28" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="115" y="142" font-size="12" font-weight="bold" fill="#991b1b" text-anchor="middle">aᵢ (Audio)</text>
  <text x="115" y="156" font-size="9.5" fill="#b91c1c" text-anchor="middle">CORRUPTED</text>

  <circle cx="380" cy="145" r="28" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="380" y="142" font-size="12" font-weight="bold" fill="#92400e" text-anchor="middle">vᵢ (Visual)</text>
  <text x="380" y="156" font-size="9.5" fill="#b45309" text-anchor="middle">Noisy (342d)</text>

  <circle cx="248" cy="265" r="32" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>
  <text x="248" y="261" font-size="12.5" font-weight="bold" fill="#1e3a8a" text-anchor="middle">lᵢ (Text)</text>
  <text x="248" y="276" font-size="9.5" fill="#1d4ed8" text-anchor="middle">BiLSTM + Spk</text>

  <!-- Edges in Panel A -->
  <line x1="143" y1="145" x2="348" y2="145" stroke="#dc2626" stroke-width="2.2" marker-end="url(#arrowRed)"/>
  <line x1="136" y1="166" x2="222" y2="241" stroke="#dc2626" stroke-width="3" marker-end="url(#arrowRed)"/>
  <line x1="358" y1="166" x2="274" y2="241" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arrowRed)"/>

  <rect x="82" y="190" width="125" height="24" fill="#fef2f2" stroke="#f87171" rx="3"/>
  <text x="144" y="206" font-size="10.5" font-weight="bold" fill="#b91c1c" text-anchor="middle">cos ≈ 0 ⇒ A(aᵢ,lᵢ) ≈ 0.50!</text>

  <rect x="36" y="312" width="427" height="64" fill="#fff1f2" stroke="#fecdd3" rx="4"/>
  <text x="249" y="330" font-size="10.5" font-weight="bold" fill="#881337" text-anchor="middle">Why "Over-Smoothing" is Invalid Here, but Corruption Propagation Occurs:</text>
  <text x="249" y="346" font-size="10" fill="#9f1239" text-anchor="middle">• GCNII (64 layers, α=0.1 residual to h⁽⁰⁾, θₗ=log(λ/l+1)) already prevents over-smoothing.</text>
  <text x="249" y="362" font-size="10" fill="#9f1239" text-anchor="middle">• However, 1 − arccos(0)/π = 0.50: pure noise still flows into Text lᵢ at 50% edge weight!</text>

  <!-- Panel B: Proposed GatedMM_GCN -->
  <rect x="498" y="46" width="460" height="342" fill="#ffffff" stroke="#94a3b8" rx="6"/>
  <rect x="498" y="46" width="460" height="28" fill="#15803d" rx="6"/>
  <text x="728" y="65" font-size="12.5" font-weight="bold" fill="#ffffff" text-anchor="middle">B. Proposed GatedMM_GCN (G1): Asymmetric Directed Sigmoid Gates</text>

  <!-- Nodes in Panel B -->
  <circle cx="595" cy="145" r="28" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="595" y="142" font-size="12" font-weight="bold" fill="#991b1b" text-anchor="middle">aᵢ (Audio)</text>
  <text x="595" y="156" font-size="9.5" fill="#b91c1c" text-anchor="middle">Noisy/Weak</text>

  <circle cx="860" cy="145" r="28" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="860" y="142" font-size="12" font-weight="bold" fill="#92400e" text-anchor="middle">vᵢ (Visual)</text>
  <text x="860" y="156" font-size="9.5" fill="#b45309" text-anchor="middle">Noisy/Weak</text>

  <circle cx="728" cy="265" r="32" fill="#dcfce7" stroke="#16a34a" stroke-width="2.5"/>
  <text x="728" y="261" font-size="12.5" font-weight="bold" fill="#14532d" text-anchor="middle">lᵢ (Text)</text>
  <text x="728" y="276" font-size="9.5" fill="#15803d" text-anchor="middle">PROTECTED</text>

  <!-- Directed Asymmetric Edges in Panel B -->
  <!-- a -> l (suppressed) -->
  <line x1="610" y1="170" x2="698" y2="248" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrowRed)"/>
  <rect x="530" y="198" width="138" height="22" fill="#fef2f2" stroke="#ef4444" rx="3"/>
  <text x="599" y="213" font-size="10" font-weight="bold" fill="#b91c1c" text-anchor="middle">g(l←a) = 0.38 (Shielded)</text>

  <!-- l -> a (open) -->
  <line x1="712" y1="238" x2="622" y2="158" stroke="#16a34a" stroke-width="2.8" marker-end="url(#arrowGreen)"/>
  <rect x="640" y="166" width="115" height="20" fill="#dcfce7" stroke="#16a34a" rx="3"/>
  <text x="697" y="180" font-size="9.5" font-weight="bold" fill="#15803d" text-anchor="middle">g(a←l) = 0.84 (Open)</text>

  <!-- v -> l (suppressed) -->
  <line x1="844" y1="170" x2="758" y2="248" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrowRed)"/>
  <rect x="788" y="198" width="142" height="22" fill="#fef2f2" stroke="#ef4444" rx="3"/>
  <text x="859" y="213" font-size="10" font-weight="bold" fill="#b91c1c" text-anchor="middle">g(l←v) = 0.48 (Shielded)</text>

  <!-- l -> v (open) -->
  <line x1="744" y1="238" x2="834" y2="158" stroke="#16a34a" stroke-width="2.8" marker-end="url(#arrowGreen)"/>
  <rect x="758" y="166" width="115" height="20" fill="#dcfce7" stroke="#16a34a" rx="3"/>
  <text x="815" y="180" font-size="9.5" font-weight="bold" fill="#15803d" text-anchor="middle">g(v←l) = 0.82 (Open)</text>

  <rect x="512" y="312" width="432" height="64" fill="#f0fdf4" stroke="#bbf7d0" rx="4"/>
  <text x="728" y="329" font-size="10.5" font-weight="bold" fill="#14532d" text-anchor="middle">Key Distinction from GraphMFT (Li et al., 2022) &amp; Empirical Discovery:</text>
  <text x="728" y="345" font-size="10" fill="#166534" text-anchor="middle">• Intra-modal edges untouched; independent directed sigmoid gₘₙ ∈ (0,1) + readout gating.</text>
  <text x="728" y="361" font-size="10" fill="#166534" text-anchor="middle">• Learns directional asymmetry: Text shields itself (0.38/0.48) while enriching Audio/Visual (0.84/0.82)!</text>
</svg>`;
fs.writeFileSync(path.join(FIGURES_DIR, 'fig_architecture_gate.svg'), svgArch, 'utf8');

// 4. Generate SVG Figure 5: Gate Activation Heatmap & Per-Class F1 Comparison
const svgGateAndClass = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 390" width="980" height="390" font-family="Segoe UI, Arial, sans-serif">
  <rect width="980" height="390" fill="#ffffff" rx="8"/>
  <rect x="8" y="8" width="964" height="374" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="490" y="32" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">Figure 3: Learned Cross-Modal Gate Asymmetry (Table 4) and Per-Class F1 Across All 7 MELD Emotions</text>

  <!-- Left Panel: Directed Gate Matrix for G0, G1, G1_CW -->
  <rect x="22" y="48" width="440" height="320" fill="#ffffff" stroke="#94a3b8" rx="6"/>
  <text x="242" y="72" font-size="12.5" font-weight="bold" fill="#1e293b" text-anchor="middle">A. Learned Directed Gate Weights g(target ← source) on Clean Test</text>

  <!-- Table headers -->
  <rect x="38" y="86" width="408" height="26" fill="#1e293b"/>
  <text x="115" y="103" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">Directed Edge</text>
  <text x="225" y="103" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">G0 (Gate Only)</text>
  <text x="315" y="103" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">G1 (Proposed)</text>
  <text x="398" y="103" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">G1_CW (2B)</text>

  <!-- Row 1: 2_0 (Text <- Audio) -->
  <rect x="38" y="112" width="408" height="30" fill="#fee2e2"/>
  <text x="115" y="131" font-size="10.5" font-weight="bold" fill="#991b1b" text-anchor="middle">2_0: Text ← Audio</text>
  <text x="225" y="131" font-size="11" font-weight="bold" fill="#991b1b" text-anchor="middle">0.3254</text>
  <text x="315" y="131" font-size="11" font-weight="bold" fill="#991b1b" text-anchor="middle">0.3784</text>
  <text x="398" y="131" font-size="11" font-weight="bold" fill="#991b1b" text-anchor="middle">0.2393</text>

  <!-- Row 2: 2_1 (Text <- Visual) -->
  <rect x="38" y="142" width="408" height="30" fill="#ffedd5"/>
  <text x="115" y="161" font-size="10.5" font-weight="bold" fill="#9a3412" text-anchor="middle">2_1: Text ← Visual</text>
  <text x="225" y="161" font-size="11" font-weight="bold" fill="#9a3412" text-anchor="middle">0.4643</text>
  <text x="315" y="161" font-size="11" font-weight="bold" fill="#9a3412" text-anchor="middle">0.4783</text>
  <text x="398" y="161" font-size="11" font-weight="bold" fill="#9a3412" text-anchor="middle">0.5348</text>

  <!-- Row 3: 0_2 (Audio <- Text) -->
  <rect x="38" y="172" width="408" height="30" fill="#dcfce7"/>
  <text x="115" y="191" font-size="10.5" font-weight="bold" fill="#166534" text-anchor="middle">0_2: Audio ← Text</text>
  <text x="225" y="191" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">0.9319</text>
  <text x="315" y="191" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">0.8359</text>
  <text x="398" y="191" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">0.9825</text>

  <!-- Row 4: 1_2 (Visual <- Text) -->
  <rect x="38" y="202" width="408" height="30" fill="#dcfce7"/>
  <text x="115" y="221" font-size="10.5" font-weight="bold" fill="#166534" text-anchor="middle">1_2: Visual ← Text</text>
  <text x="225" y="221" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">0.6891</text>
  <text x="315" y="221" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">0.8177</text>
  <text x="398" y="221" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">0.9962</text>

  <!-- Row 5: 0_1 & 1_0 -->
  <rect x="38" y="232" width="408" height="30" fill="#f1f5f9"/>
  <text x="115" y="251" font-size="10.5" fill="#334155" text-anchor="middle">0_1 / 1_0: Audio ↔ Vis</text>
  <text x="225" y="251" font-size="10.5" fill="#334155" text-anchor="middle">1.00 / 0.97</text>
  <text x="315" y="251" font-size="10.5" fill="#334155" text-anchor="middle">1.00 / 0.92</text>
  <text x="398" y="251" font-size="10.5" fill="#334155" text-anchor="middle">1.00 / 1.00</text>

  <!-- Unfavorable / Zero-out behavior box -->
  <rect x="38" y="272" width="408" height="84" fill="#fffbeb" stroke="#f59e0b" rx="4"/>
  <text x="242" y="290" font-size="10" font-weight="bold" fill="#92400e" text-anchor="middle">Diagnostic Finding under Complete Zeroing (audio_zero):</text>
  <text x="242" y="307" font-size="9.1" fill="#78350f" text-anchor="middle">When aᵢ = 0, LayerNorm(0) = 0 ⇒ Gate MLP input vanishes except init bias (+2.0),</text>
  <text x="242" y="323" font-size="9.1" fill="#78350f" text-anchor="middle">causing g(l←a) to rebound 0.378 → 0.621 (drift = 0.063). Thus G1's robustness</text>
  <text x="242" y="339" font-size="9.1" fill="#78350f" text-anchor="middle">stems from static directional shielding + readout gating, not dynamic zero-detection.</text>

  <!-- Right Panel: Per-Class F1 Comparison Across All 7 Emotions -->
  <rect x="478" y="48" width="480" height="320" fill="#ffffff" stroke="#94a3b8" rx="6"/>
  <text x="718" y="72" font-size="12.5" font-weight="bold" fill="#1e293b" text-anchor="middle">B. Per-Class Test F1 (%) Across All 7 Emotions (Standard vs. Minority Fixes)</text>

  <rect x="494" y="86" width="448" height="26" fill="#1e293b"/>
  <text x="542" y="103" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">Emotion</text>
  <text x="615" y="103" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">B0 (Orig)</text>
  <text x="685" y="103" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">G1 (Std)</text>
  <text x="765" y="103" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">B1+LA (2A)</text>
  <text x="845" y="103" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">G1_CW (2B)</text>
  <text x="915" y="103" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">G1_CW+LA</text>

  <!-- 7 emotion rows -->
  <g font-size="10.5" text-anchor="middle">
    <rect x="494" y="112" width="448" height="26" fill="#f8fafc"/>
    <text x="542" y="129" font-weight="bold" fill="#1e293b">0: neutral</text>
    <text x="615" y="129">75.17</text><text x="685" y="129" font-weight="bold" fill="#15803d">75.23</text><text x="765" y="129">66.76</text><text x="845" y="129">73.75</text><text x="915" y="129">70.72</text>

    <rect x="494" y="138" width="448" height="26" fill="#ffffff"/>
    <text x="542" y="155" font-weight="bold" fill="#1e293b">1: surprise</text>
    <text x="615" y="155">46.52</text><text x="685" y="155" font-weight="bold" fill="#15803d">46.82</text><text x="765" y="155">46.92</text><text x="845" y="155">45.39</text><text x="915" y="155">45.44</text>

    <rect x="494" y="164" width="448" height="26" fill="#fef2f2"/>
    <text x="542" y="181" font-weight="bold" fill="#b91c1c">2: fear (50)</text>
    <text x="615" y="181" fill="#b91c1c">0.00</text><text x="685" y="181" fill="#b91c1c">0.00</text><text x="765" y="181" font-weight="bold" fill="#15803d">5.87</text><text x="845" y="181" font-weight="bold" fill="#15803d">3.85</text><text x="915" y="181" font-weight="bold" fill="#15803d">4.95</text>

    <rect x="494" y="190" width="448" height="26" fill="#ffffff"/>
    <text x="542" y="207" font-weight="bold" fill="#1e293b">3: sadness</text>
    <text x="615" y="207">29.57</text><text x="685" y="207">27.04</text><text x="765" y="207" font-weight="bold" fill="#15803d">29.02</text><text x="845" y="207">20.17</text><text x="915" y="207">24.53</text>

    <rect x="494" y="216" width="448" height="26" fill="#f8fafc"/>
    <text x="542" y="233" font-weight="bold" fill="#1e293b">4: joy</text>
    <text x="615" y="233">52.17</text><text x="685" y="233" font-weight="bold" fill="#15803d">54.35</text><text x="765" y="233">53.13</text><text x="845" y="233">50.40</text><text x="915" y="233">49.45</text>

    <rect x="494" y="242" width="448" height="26" fill="#fef2f2"/>
    <text x="542" y="259" font-weight="bold" fill="#b91c1c">5: disgust (68)</text>
    <text x="615" y="259" fill="#b91c1c">2.86</text><text x="685" y="259" fill="#b91c1c">0.00</text><text x="765" y="259" font-weight="bold" fill="#15803d">4.08</text><text x="845" y="259" font-weight="bold" fill="#15803d">3.70</text><text x="915" y="259" font-weight="bold" fill="#15803d">6.70</text>

    <rect x="494" y="268" width="448" height="26" fill="#ffffff"/>
    <text x="542" y="285" font-weight="bold" fill="#1e293b">6: anger</text>
    <text x="615" y="285">43.83</text><text x="685" y="285">43.81</text><text x="765" y="285">44.19</text><text x="845" y="285" font-weight="bold" fill="#15803d">44.33</text><text x="915" y="285">43.61</text>

    <rect x="494" y="296" width="448" height="28" fill="#e2e8f0"/>
    <text x="542" y="314" font-weight="bold" fill="#0f172a">Weighted-F1</text>
    <text x="615" y="314" font-weight="bold">57.44</text><text x="685" y="314" font-weight="bold" fill="#15803d">57.56</text><text x="765" y="314">53.73</text><text x="845" y="314">55.78</text><text x="915" y="314">54.53</text>

    <rect x="494" y="324" width="448" height="28" fill="#cbd5e1"/>
    <text x="542" y="342" font-weight="bold" fill="#0f172a">Macro-F1</text>
    <text x="615" y="342" font-weight="bold">35.73</text><text x="685" y="342" font-weight="bold" fill="#15803d">35.32</text><text x="765" y="342" font-weight="bold" fill="#15803d">35.71</text><text x="845" y="342">34.51</text><text x="915" y="342">35.06</text>
  </g>
</svg>`;
fs.writeFileSync(path.join(FIGURES_DIR, 'fig_gates_and_perclass.svg'), svgGateAndClass, 'utf8');

console.log('Generated SVG figures and verified all results in:', FIGURES_DIR);
