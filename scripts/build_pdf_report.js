/**
 * Generates the print-ready HTML and renders Cetinkaya_YZM5110_Research.pdf via headless Chrome.
 * Author: Samet Çetinkaya (26526003)
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const FIG_DIR = path.join(ROOT, 'results', 'figures');

function b64Png(filename) {
  const p = path.join(FIG_DIR, filename);
  return 'data:image/png;base64,' + fs.readFileSync(p).toString('base64');
}

function readSvg(filename) {
  return fs.readFileSync(path.join(FIG_DIR, filename), 'utf8');
}

const fig1Svg = readSvg('fig_meld_overview.svg');
const fig2Svg = readSvg('fig_architecture_gate.svg');
const fig3Svg = readSvg('fig_gates_and_perclass.svg');
const fig4Png = b64Png('fig1_robustness_wf1.png');
const fig5Png = b64Png('fig2_minority_f1.png');

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cetinkaya_YZM5110_Research</title>
<style>
  @page {
    size: A4;
    margin: 14mm 14mm 14mm 14mm;
  }
  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }
  body {
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 9.6pt;
    line-height: 1.44;
    color: #111827;
    margin: 0;
    padding: 0;
  }
  .header-banner {
    border-top: 2px solid #0f172a;
    border-bottom: 2px solid #0f172a;
    background: #ffffff;
    padding: 10px 4px 10px 4px;
    margin-bottom: 10px;
    text-align: center;
  }
  .header-banner h1 {
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 13.8pt;
    line-height: 1.25;
    margin: 0 0 8px 0;
    color: #0f172a;
    font-weight: 700;
  }
  .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 18px;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.6pt;
    color: #1e293b;
    border-top: 1px solid #cbd5e1;
    padding-top: 6px;
    text-align: left;
  }
  .repo-box {
    margin-top: 6px;
    padding: 5px 10px;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #1e3a8a;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.4pt;
    color: #0f172a;
    text-align: left;
  }
  .abstract-box {
    margin: 0 10px 9px 10px;
    padding: 7px 12px;
    background: #f8fafc;
    border-left: 3px solid #0f172a;
    border-right: 1px solid #e2e8f0;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 8.8pt;
    line-height: 1.38;
    text-align: justify;
  }
  .epistemic-box {
    padding: 4px 9px;
    background: #f1f5f9;
    border-left: 3px solid #475569;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.1pt;
    color: #334155;
    margin-bottom: 8px;
  }
  h2 {
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 11.2pt;
    color: #0f172a;
    border-bottom: 1.2px solid #1e293b;
    padding-bottom: 2px;
    margin: 11px 0 5px 0;
    page-break-after: avoid;
  }
  h3 {
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 9.9pt;
    color: #1e3a8a;
    margin: 8px 0 4px 0;
    page-break-after: avoid;
  }
  p {
    margin: 0 0 5px 0;
    text-align: justify;
  }
  ul, ol {
    margin: 0 0 5px 0;
    padding-left: 18px;
  }
  li {
    margin-bottom: 3px;
    text-align: justify;
  }
  code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.2pt;
    background: #f1f5f9;
    padding: 0.5px 3px;
    border-radius: 2px;
    color: #0f172a;
  }
  .tag {
    font-family: "Segoe UI", Arial, sans-serif;
    font-weight: 700;
    font-size: 7.5pt;
    padding: 0.5px 3.5px;
    border-radius: 2px;
    white-space: nowrap;
  }
  .tag-a { background: #e0e7ff; color: #1e40af; }
  .tag-m { background: #dcfce7; color: #166534; }
  .tag-i { background: #fef3c7; color: #92400e; }
  .math-block {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 4px 10px;
    margin: 4px 0;
    font-family: "Cambria Math", Georgia, serif;
    font-size: 9.2pt;
    text-align: center;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0 8px 0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.8pt;
    line-height: 1.32;
    page-break-inside: avoid;
  }
  caption {
    text-align: left;
    margin-bottom: 3px;
    font-size: 8.3pt;
    color: #0f172a;
  }
  th {
    background: #1e293b;
    color: #ffffff;
    font-weight: 600;
    padding: 3.5px 4.5px;
    border: 1px solid #334155;
    text-align: center;
  }
  td {
    padding: 3px 4.5px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
  }
  tr:nth-child(even) td {
    background: #f8fafc;
  }
  .num { text-align: center; }
  .best { font-weight: 700; color: #15803d; background: #f0fdf4 !important; }
  .warn { font-weight: 700; color: #b91c1c; background: #fef2f2 !important; }
  .fig-container {
    margin: 6px 0;
    text-align: center;
    page-break-inside: avoid;
  }
  .fig-container svg {
    width: 100%;
    height: auto;
    max-height: 235px;
  }
  .fig-container img {
    max-width: 92%;
    max-height: 205px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
  }
  .fig-caption {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.9pt;
    color: #334155;
    margin-top: 2px;
    text-align: center;
  }
</style>
</head>
<body>

<div class="header-banner">
  <h1>Cross-Modal Edge Gating and Minority-Class Calibration in Multimodal Graph Convolutional Networks for Emotion Recognition in Conversation</h1>
  <div class="meta-grid">
    <div><strong>Course:</strong> YZM5110 – Research Project (Weight: 30%, Individual)</div>
    <div><strong>Author:</strong> Samet Çetinkaya</div>
    <div><strong>Student ID:</strong> 26526003</div>
    <div><strong>Submission Date:</strong> November 18, 2026 (Week 7)</div>
    <div><strong>Topic:</strong> Multimodal Graph Neural Networks for Emotion Recognition in Conversation</div>
    <div><strong>Execution Hardware:</strong> Google Colab NVIDIA Tesla T4 GPU (CUDA 13.0)</div>
  </div>
  <div class="repo-box">
    <strong>Public Repository URL (First-Page Requirement per §5):</strong> <code>https://github.com/smtcetinkaya/MMGCN-MELD-YZM5110</code> (Pinned Tag: <code>v1.0-yzm5110</code>)<br/>
    <strong>Reproduced Source Repository:</strong> <code>https://github.com/hujingwen6666/MMGCN</code> (Pinned Commit: <code>85732d984f70c1c84dd47c81aa97f1271397b899</code>)
  </div>
</div>

<div class="abstract-box">
  <strong>Abstract —</strong> Multimodal Graph Convolutional Network (MMGCN; Hu et al., ACL 2021) models conversational emotion recognition over a unified 3<i>N</i>-node trimodal graph using 64 spectral GCNII layers. Through a code-level audit of the official repository, we first document five methodological discrepancies—including zero-validation test-set peak picking (<code>valid=0.0</code>) and an unweighted custom <code>FocalLoss</code> on MELD—and establish a leak-free dialogue-level reproduction on MELD achieving 57.44% unclipped (56.60% matched grad-clip) Test Weighted-F1 vs. the paper's test-selected 58.65%. Next, noting that GCNII's initial residual (&alpha;=0.1) already prevents over-smoothing, we formulate the hypothesis of <em>cross-modal corruption propagation</em>: MMGCN's fixed angular similarity assigns <i>A<sub>mn</sub></i> &approx; 0.50 to orthogonal noise (cos &approx; 0), pumping degraded acoustic/visual signals into reliable text nodes. We propose <code>GatedMM_GCN</code> (<code>G1</code>), which places learnable directed sigmoid gates exclusively on cross-modal edges paired with stochastic modality-corruption training augmentation. Evaluated across 10 clean and corrupted conditions against an un-gated augmentation control (<code>B1</code>) and a gate-only ablation (<code>G0</code>), <code>G1</code> improves clean Weighted-F1 to <strong>57.56%</strong> (+0.96 over <code>B0</code>, +0.88 over <code>B1</code>) and achieves the highest robustness in 8 of 9 corruption scenarios (mean <strong>57.09%</strong>). Mechanistically, the learned gates spontaneously discover directional asymmetry, shielding text from noisy audio/visual inflows (<i>g<sub>l&larr;a</sub></i>=0.38, <i>g<sub>l&larr;v</sub></i>=0.48) while keeping text-to-audio/visual outflows open (0.82–0.84). Finally, we resolve the recurring 0% F1 collapse on rare emotions (<code>fear</code>, <code>disgust</code>) via validation-tuned post-hoc logit adjustment and class-weighted training.
</div>

<div class="epistemic-box">
  <strong>Epistemic Attribution Convention (§2 Guidelines):</strong> Throughout this report, statements and scores are explicitly tagged as <span class="tag tag-a">[Authors]</span> when citing published claims from the literature, <span class="tag tag-m">[Measured]</span> when reporting empirical results obtained from our own executions on Tesla T4 GPU, and <span class="tag tag-i">[Interpretation]</span> when presenting our analytical reasoning or mechanistic explanation.
</div>

<h2>1. Research Question, Scope, Motivation, and Contributions</h2>

<h3>1.1 Research Problem and Theoretical Motivation</h3>
<p>
Emotion Recognition in Conversation (ERC) aims to classify the emotion of each utterance <i>u<sub>i</sub></i> in a multi-party dialogue by integrating sequential speaker context and trimodal signals: textual transcripts (<i>l</i>), acoustic prosody (<i>a</i>), and visual facial expressions (<i>v</i>). Multimodal Graph Convolutional Network (<strong>MMGCN</strong>; Hu et al., ACL 2021) <span class="tag tag-a">[Authors]</span> constructs a unified 3<i>N</i>-node graph for an <i>N</i>-utterance dialogue and propagates information across 64 spectral graph convolution layers modeled after GCNII (Chen et al., 2020).
</p>
<p>
A close audit of MMGCN's mathematical mechanism and official PyTorch implementation reveals two critical weaknesses <span class="tag tag-i">[Interpretation]</span>:
</p>
<ol>
  <li><strong>Cross-Modal Corruption Propagation (Why "Over-Smoothing" is Invalid for MMGCN):</strong> In standard shallow GCNs, stacking layers causes node representations to converge (over-smoothing). However, MMGCN incorporates GCNII's initial residual connection (&alpha; = 0.1 anchor to <i>H</i><sup>(0)</sup>) and identity mapping (&theta;<sub>&ell;</sub> = log(&lambda;/&ell; + 1)), which explicitly prevent over-smoothing across 64 layers <span class="tag tag-a">[Authors]</span>. Thus, motivating an improvement via over-smoothing is invalid for MMGCN <span class="tag tag-i">[Interpretation]</span>. Instead, the true structural vulnerability is <strong>unconditional corruption propagation</strong>: MMGCN weights every cross-modal edge (<i>m, n</i>) for utterance <i>i</i> using a fixed, unlearnable angular similarity formula:
  <div class="math-block">
    A<sub>mn</sub>(i, i) = 1 &minus; arccos(cos(h<sub>i</sub><sup>m</sup>, h<sub>i</sub><sup>n</sup>)) / &pi; &isin; [0, 1]
  </div>
  When an acoustic or visual stream degrades into uncorrelated noise or sensor dropout (cos(<i>h<sub>i</sub><sup>a</sup>, h<sub>i</sub><sup>l</sup></i>) &approx; 0), arccos(0) = &pi;/2, yielding <i>A<sub>al</sub></i>(<i>i, i</i>) = 1 &minus; 0.5 = <strong>0.50</strong>. Consequently, a completely corrupted modality continues to inject noise into the reliable text node at <strong>50% of the maximum edge weight</strong> across 64 layers (<em>Pathway 1: Graph Propagation</em>) and directly into the final 900-d concatenation classifier (<em>Pathway 2: Readout Leakage</em>).</li>
  <li><strong>Severe Minority-Class Collapse on MELD (<code>fear</code> and <code>disgust</code> F1 = 0%):</strong> On MELD (Poria et al., 2019), <code>neutral</code> comprises 47.1% of training data (48.1% test), whereas <code>fear</code> (2.68% train; 1.92% test, 50 utterances) and <code>disgust</code> (2.71% train; 2.61% test, 68 utterances) are tiny minorities <span class="tag tag-m">[Measured]</span>. Although the MMGCN authors apply inverse-frequency class weights on IEMOCAP, they disable class weighting on MELD and use a custom <code>FocalLoss</code> without class-wise &alpha;<sub>c</sub> balancing <span class="tag tag-m">[Measured]</span>, causing standard models to predict <code>fear</code> and <code>disgust</code> almost never (&le; 4 times across 2,610 test utterances).</li>
</ol>

<h3>1.2 Research Questions, Scope, and Contributions</h3>
<ul>
  <li><strong>RQ1 (Primary — Cross-Modal Edge Gate &amp; Robustness):</strong> <em>Does introducing a learnable, directed sigmoid gate exclusively on cross-modal edges—paired with stochastic modality-corruption training augmentation—mitigate corruption propagation and improve robustness under acoustic/visual degradation without sacrificing clean performance?</em></li>
  <li><strong>RQ2 (Secondary — Minority-Class Recovery):</strong> <em>Can validation-tuned post-hoc logit adjustment (Intervention 2A) and dampened class-weighted training (Intervention 2B) rescue collapsed minority emotions (<code>fear</code>, <code>disgust</code>) on MELD without confounding the primary gating comparison?</em></li>
  <li><strong>Scope:</strong> We evaluate exclusively on the public <strong>MELD</strong> dataset using the official pre-extracted trimodal features (<code>MELD_features_raw1.pkl</code>: 600-d TextCNN, 300-d OpenSMILE, 342-d DenseNet). <strong>IEMOCAP</strong> is excluded because it requires a restrictive USC license unavailable within the course timeline.</li>
</ul>

<div class="fig-container">
  ${fig1Svg}
</div>

<h2>2. Literature Search Approach and Critical Synthesis</h2>
<p>
We searched the ACL Anthology, IEEE Xplore, and arXiv (2019–2023) using queries targeting multimodal conversational emotion graphs. We selected five central papers that represent the baseline lineage (<strong>DialogueGCN</strong> &rarr; <strong>MMGCN</strong>) and three competing architectural paradigms: Graph Transformers (<strong>COGMEN</strong>), layer-wise dynamic fusion (<strong>MM-DFN</strong>), and GAT-based edge attention (<strong>GraphMFT</strong>). Table 1 presents the mandatory 7-field comparison matrix, and Table 2 contrasts GraphMFT against our directed cross-modal gate.
</p>

<table>
  <caption><strong>Table 1: Seven-Field Literature Comparison Matrix Across Five Core ERC Graph Papers (per YZM5110 §2 Guidelines)</strong></caption>
  <thead>
    <tr>
      <th style="width:11%">Paper</th>
      <th style="width:13%">1. Question &amp; Contribution</th>
      <th style="width:11%">2. Modalities &amp; Assumptions</th>
      <th style="width:15%">3. Mechanism &amp; Objective</th>
      <th style="width:12%">4. Data &amp; Protocol</th>
      <th style="width:12%">5. Baselines &amp; Evidence</th>
      <th style="width:12%">6. Resources &amp; Reproducibility</th>
      <th style="width:14%">7. Limitations &amp; Connections</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DialogueGCN</strong><br/>Ghosal et al. (EMNLP 2019) <span class="tag tag-a">[Authors]</span></td>
      <td>Models self- and inter-speaker dependencies in dialogues via directed speaker-relational utterance graphs.</td>
      <td>Primarily <strong>Text (<i>l</i>)</strong>; audio/visual only via naive early vector concatenation. Assumes clean inputs.</td>
      <td>Utterance nodes in &plusmn;10 window; 2<i>S</i><sup>2</sup> speaker-temporal edge types; 2-layer <strong>RGCN</strong> + similarity attention; Cross-Entropy loss.</td>
      <td>IEMOCAP (6-class), AVEC, MELD (7-class); standard train/val/test splits.</td>
      <td>Beats bc-LSTM, CMN, ICON, DialogueRNN on IEMOCAP <span class="tag tag-a">[Authors]</span>; ablates window sizes.</td>
      <td>Public PyTorch code (<code>declare-lab/conv-emotion</code>); lightweight 2-layer GNN.</td>
      <td>No cross-modal graph; suffers from <strong>over-smoothing</strong> beyond 2–3 layers <span class="tag tag-a">[Authors]</span>, motivating MMGCN.</td>
    </tr>
    <tr>
      <td><strong>MMGCN</strong><br/>Hu et al. (ACL 2021) <span class="tag tag-a">[Authors]</span><br/><em>(Target)</em></td>
      <td>Unifies intra-modal context and cross-modal fusion in one deep 64-layer spectral GCNII over 3<i>N</i> nodes.</td>
      <td><strong>Text (<i>l</i>), Audio (<i>a</i>), Visual (<i>v</i>)</strong>. Assumes all 3 modalities are reliable at train and test time.</td>
      <td>BiLSTM (<i>l</i>) + FC (<i>a,v</i>) + speaker emb &rarr; 3<i>N</i>&times;3<i>N</i> graph with fixed <strong>angular similarity</strong> edges &rarr; 64-layer <strong>GCNII</strong>; custom <code>FocalLoss</code> on MELD.</td>
      <td>IEMOCAP (66.22% W-F1) &amp; MELD (<strong>58.65%</strong> W-F1) <span class="tag tag-a">[Authors]</span>. Code uses <code>valid=0.0</code> and picks peak test F1 over 60 epochs <span class="tag tag-m">[Measured]</span>.</td>
      <td>Outperforms DialogueRNN, DialogueGCN, bc-LSTM; includes modality &amp; depth ablations <span class="tag tag-a">[Authors]</span>.</td>
      <td>Official repo provides <code>.pkl</code> features; no license file; hardcoded <code>nlayers=64</code> bug <span class="tag tag-m">[Measured]</span>.</td>
      <td>Fixed edges (<i>A<sub>mn</sub></i>&approx;0.5 at cos=0) allow <strong>corruption propagation</strong>; unweighted MELD loss collapses <code>fear</code>/<code>disgust</code> <span class="tag tag-m">[Measured]</span>.</td>
    </tr>
    <tr>
      <td><strong>COGMEN</strong><br/>Joshi et al. (NAACL 2022) <span class="tag tag-a">[Authors]</span></td>
      <td>Couples local speaker-relational context with global cross-utterance interactions via Graph Transformers.</td>
      <td><strong>Text (SBERT), Audio, Visual</strong> concatenated early per utterance before graph construction.</td>
      <td>Early concatenation &rarr; relational graph &rarr; <strong>RGCN</strong> + <strong>TransformerConv</strong> layers; Cross-Entropy loss.</td>
      <td>IEMOCAP (4/6-class) and CMU-MOSEI <span class="tag tag-a">[Authors]</span>; <strong>MELD not evaluated</strong>.</td>
      <td>Strong gains on IEMOCAP/MOSEI; ablates RGCN vs. TransformerConv <span class="tag tag-a">[Authors]</span>.</td>
      <td>Public repo (<code>Exploration-Lab/COGMEN</code>) with WandB configs.</td>
      <td>Early concatenation precludes node-level cross-modal gating during message passing <span class="tag tag-i">[Interpretation]</span>.</td>
    </tr>
    <tr>
      <td><strong>MM-DFN</strong><br/>Hu et al. (ICASSP 2022) <span class="tag tag-a">[Authors]</span></td>
      <td>Reduces cross-modal <strong>redundancy accumulation</strong> across GNN layers via dynamic fusion modules.</td>
      <td><strong>Text (<i>l</i>), Audio (<i>a</i>), Visual (<i>v</i>)</strong>. Assumes clean, synchronized streams.</td>
      <td>Multi-layer graph-based <strong>Dynamic Fusion (DF)</strong> blocks that gate unimodal vs. multimodal states at the <strong>layer level</strong>.</td>
      <td>IEMOCAP (68.18% W-F1) and MELD (<strong>59.46%</strong> W-F1) <span class="tag tag-a">[Authors]</span>.</td>
      <td>Improves over MMGCN by +0.81 W-F1 on MELD <span class="tag tag-a">[Authors]</span>; ablates fusion modules.</td>
      <td>Relies on same pre-extracted feature pipeline as MMGCN.</td>
      <td>Gates operate at macro layer level rather than directed cross-modal edges; sensor corruption not tested <span class="tag tag-i">[Interpretation]</span>.</td>
    </tr>
    <tr>
      <td><strong>GraphMFT</strong><br/>Li et al. (arXiv 2022) <span class="tag tag-a">[Authors]</span></td>
      <td>Replaces MMGCN's heuristic cosine-similarity edges and 3<i>N</i>&times;3<i>N</i> graph with multi-graph GAT attention.</td>
      <td><strong>Text (<i>l</i>), Audio (<i>a</i>), Visual (<i>v</i>)</strong>. Decomposes into <i>V-A</i> and <i>V-L</i> bipartite/intra graphs.</td>
      <td>Multiple 2<i>N</i> graphs; replaces heuristic angular similarity with <strong>learnable GAT attention</strong> (neighbor-wise softmax) on <strong>all</strong> edges.</td>
      <td>IEMOCAP (67.90% W-F1) and MELD (<strong>58.37%</strong> W-F1, i.e., &minus;0.28 below MMGCN) <span class="tag tag-a">[Authors]</span>.</td>
      <td>Ablates graph combinations; lowers memory vs. MMGCN, but MELD W-F1 lags MMGCN <span class="tag tag-a">[Authors]</span>.</td>
      <td>Architecture specified; public checkpoints limited.</td>
      <td><strong>Softmax normalization</strong> over neighbors forces weights to sum to 1, unable to zero out all noisy cross-modal sources <span class="tag tag-i">[Interpretation]</span>.</td>
    </tr>
  </tbody>
</table>

<table>
  <caption><strong>Table 2: Structural Comparison Between GraphMFT's Attention Edges and Our Proposed Directed Cross-Modal Gate</strong></caption>
  <thead>
    <tr>
      <th style="width:20%">Design Dimension</th>
      <th style="width:40%">GraphMFT (Li et al., 2022) <span class="tag tag-a">[Authors]</span></th>
      <th style="width:40%">Proposed <code>GatedMM_GCN</code> (<code>G0</code> / <code>G1</code>) <span class="tag tag-i">[Interpretation]</span> <span class="tag tag-m">[Measured]</span></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Edge Scope</strong></td>
      <td>Applies GAT attention to <strong>both intra-modal and inter-modal edges</strong>, entangling temporal context with cross-modal fusion.</td>
      <td>Applies learnable gates <strong>exclusively to cross-modal edges (<i>m &ne; n</i>)</strong>; intra-modal edges remain 100% identical to MMGCN.</td>
    </tr>
    <tr>
      <td><strong>2. Normalization</strong></td>
      <td><strong>Neighbor-wise Softmax</strong> (&sum;<sub>j</sub> &alpha;<sub>ij</sub> = 1): cannot shut off a corrupted modality without redistributing mass elsewhere.</td>
      <td><strong>Independent Multiplicative Sigmoid</strong> <i>g<sub>m&larr;n</sub></i>(<i>i</i>) &isin; (0, 1) (<i>A'<sub>mn</sub> = g<sub>m&larr;n</sub> A<sub>mn</sub></i>): can suppress noisy channels independently.</td>
    </tr>
    <tr>
      <td><strong>3. Asymmetry</strong></td>
      <td>Decomposes into <i>V-A</i> and <i>V-L</i> graphs; lacks explicit 6-way directed shielding between all modality pairs.</td>
      <td><strong>6-Way Directed Asymmetry</strong>: discovers <i>g<sub>l&larr;a</sub></i> = 0.38 (shielding text) while keeping <i>g<sub>a&larr;l</sub></i> = 0.84 (enriching audio) <span class="tag tag-m">[Measured]</span>.</td>
    </tr>
    <tr>
      <td><strong>4. Initialization &amp; Readout</strong></td>
      <td>Random GAT initialization; no readout reliability modulation; scores 58.37% on MELD (&minus;0.28 vs. MMGCN) <span class="tag tag-a">[Authors]</span>.</td>
      <td><strong>Warm-Start Identity Bias</strong> (<i>b</i> = +2.0 &rArr; <i>g</i><sub>0</sub> &approx; 0.88) + readout gating; improves clean MELD W-F1 by +0.96 over matched <code>B0</code> <span class="tag tag-m">[Measured]</span>.</td>
    </tr>
  </tbody>
</table>

<h2>3. Technical Explanation of MMGCN</h2>
<p>
For a dialogue of <i>N</i> utterances, MMGCN (<code>model.py</code> L921–1008, <code>model_mm.py</code> L47–175) projects raw features <i>x<sub>i</sub><sup>l</sup></i> &isin; R<sup>600</sup>, <i>x<sub>i</sub><sup>v</sup></i> &isin; R<sup>342</sup>, <i>x<sub>i</sub><sup>a</sup></i> &isin; R<sup>300</sup> into R<sup>200</sup>. Text features pass through a 2-layer BiLSTM (<i>D<sub>e</sub></i> = 100) and receive an additive speaker embedding <i>e<sub>s<sub>i</sub></sub></i> &isin; R<sup>200</sup> (<i>S</i> = 9 in MELD), while audio and visual features use linear projections (<code>av_using_lstm=False</code>). The resulting 3<i>N</i> nodes form a block adjacency matrix <i>A</i> &isin; R<sup>3<i>N</i>&times;3<i>N</i></sup> with dense intra-modal blocks <i>A<sub>mm</sub></i>(<i>i,j</i>) and diagonal cross-modal blocks <i>A<sub>mn</sub></i>(<i>i,i</i>) weighted by 1 &minus; arccos(0.99999 cos(&middot;))/&pi;. After symmetric normalization <i>&tilde;A = D</i><sup>&minus;1/2</sup><i>AD</i><sup>&minus;1/2</sup>, features pass through <i>L</i> = 64 GCNII layers:
</p>
<div class="math-block">
  H<sup>(&ell;)</sup> = ReLU( &theta;<sub>&ell;</sub> [ (1 &minus; &alpha;)&tilde;A H<sup>(&ell;&minus;1)</sup> &Vert; &alpha; H<sup>(0)</sup> ] W<sup>(&ell;)</sup> + (1 &minus; &theta;<sub>&ell;</sub>)((1 &minus; &alpha;)&tilde;A H<sup>(&ell;&minus;1)</sup> + &alpha; H<sup>(0)</sup>) ), &nbsp; &alpha;=0.1, &lambda;=0.5, &theta;<sub>&ell;</sub>=log(&lambda;/&ell; + 1)
</div>
<p>
Finally, skip-connected representations <i>&tilde;H<sub>m</sub></i> = [<i>H<sub>m</sub></i><sup>in</sup> &Vert; <i>H<sub>m</sub></i><sup>(64)</sup>] &isin; R<sup><i>N</i>&times;300</sup> are concatenated across the three modalities into <i>Z<sub>i</sub></i> &isin; R<sup>900</sup> and mapped to 7 emotion log-probabilities.
</p>

<h2>4. Reproduction Target, Implementation, and Deviations from the Source</h2>
<p>
We target the MELD benchmark claim in Hu et al. (2021, Table 2): <strong>Weighted-F1 = 58.65%</strong> <span class="tag tag-a">[Authors]</span>. Before attributing any score discrepancy to the model, our line-by-line audit of commit <code>85732d984f70c1c84dd47c81aa97f1271397b899</code> uncovered five critical undocumented flaws <span class="tag tag-m">[Measured]</span>:
</p>
<ul>
  <li><strong>K1 (Layer Count Discrepancy — <code>run.sh</code> vs. <code>model.py</code> L1007):</strong> <code>run.sh</code> specifies <code>--Deep_GCN_nlayers 4</code>, but <code>model.py</code> hardcodes <code>nlayers=64</code> when instantiating <code>MM_GCN</code>. Thus, the effective published architecture is 64 layers. We link the parameter and run all experiments at the true effective depth (<code>GRAPH_LAYERS = 64</code>).</li>
  <li><strong>K2 (Zero Validation &amp; Test-Set Selection Bias — <code>train.py</code> L466, L500–528):</strong> For MELD, the official script sets <code>valid=0.0</code>, evaluates <code>test_loader</code> after every epoch, and reports <code>max(all_fscore)</code> across 60 epochs. Selecting the checkpoint on the test set inflates the published score. We fix this by partitioning <code>trainVid</code> at the <strong>dialogue level</strong> (<code>SEED=27350</code>) into 90% train (1,038 dialogues; 8,990 utterances) and 10% validation (115 dialogues; 999 utterances), evaluating the 280-dialogue test set (2,610 utterances) <strong>only once</strong> using the best validation checkpoint.</li>
  <li><strong>K3 &amp; K4 (Non-Standard Unweighted <code>FocalLoss</code> on MELD — <code>model.py</code> L13–48, <code>train.py</code> L441–461):</strong> Unlike standard Focal Loss &minus;(1 &minus; <i>p<sub>t</sub></i>)<sup>&gamma;</sup> log <i>p<sub>t</sub></i>, the authors compute <code>pt = label_onehot * log_p</code> (&le; 0) and <code>fl = -(1 - pt)**2.5 * log_p</code> over all 7 classes without class weights on MELD (even though inverse-frequency weights are used on IEMOCAP).</li>
  <li><strong>K5 (Legacy PyTorch 1.1 Compatibility):</strong> Patched idempotently (<code>adj[idx[0], idx[1]] = dia_sim</code>, explicit <code>dim=1</code> in <code>F.log_softmax</code>).</li>
</ul>

<table>
  <caption><strong>Table 3: Comparison of Published MMGCN vs. Our Leak-Free Reproduced Baselines on MELD (Tesla T4 GPU)</strong></caption>
  <thead>
    <tr>
      <th>Run / Source</th>
      <th>Validation Split &amp; Selection Protocol</th>
      <th>Best Val Ep</th>
      <th>Val W-F1 (%)</th>
      <th>Test W-F1 (%)</th>
      <th>Test Macro-F1 (%)</th>
      <th>Test Acc (%)</th>
      <th><code>fear</code> F1 (%)</th>
      <th><code>disgust</code> F1 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Published MMGCN</strong> (Hu et al., 2021) <span class="tag tag-a">[Authors]</span></td>
      <td><code>valid=0.0</code> (100% train, 1,153 dia); max of 60 test evaluations</td>
      <td class="num">—</td>
      <td class="num">—</td>
      <td class="num"><strong>58.65</strong></td>
      <td class="num"><em>Not reported</em></td>
      <td class="num"><em>Not reported</em></td>
      <td class="num"><em>Not reported</em></td>
      <td class="num"><em>Not reported</em></td>
    </tr>
    <tr>
      <td><strong>Reproduced <code>B0</code> (Run 1: Exact Unclipped)</strong> <span class="tag tag-m">[Measured]</span></td>
      <td>Dialogue-level 90% Train (1,038 dia) / 10% Val (115 dia); single test eval</td>
      <td class="num">50</td>
      <td class="num">70.18</td>
      <td class="num best">57.44</td>
      <td class="num best">35.73</td>
      <td class="num best">58.85</td>
      <td class="num warn">0.00</td>
      <td class="num warn">2.86</td>
    </tr>
    <tr>
      <td><strong>Reproduced <code>B0</code> (Run 2: Matched Grad-Clip)</strong> <span class="tag tag-m">[Measured]</span></td>
      <td>Same 90%/10% split + <code>clip_grad_norm_(5.0)</code> (matched to <code>B1/G0/G1</code>)</td>
      <td class="num">34</td>
      <td class="num">70.07</td>
      <td class="num">56.60</td>
      <td class="num">34.72</td>
      <td class="num">58.24</td>
      <td class="num warn">3.51</td>
      <td class="num warn">0.00</td>
    </tr>
  </tbody>
</table>
<p>
Our unclipped reproduction achieves <strong>57.44% Test Weighted-F1</strong> <span class="tag tag-m">[Measured]</span>, within 1.21 points of the paper's 58.65% <span class="tag tag-a">[Authors]</span>, fully accounted for by the 10% smaller training set and removal of test-set selection bias. Because Run 1 exhibited a gradient spike at Epoch 60 (train loss jumping 3.4977 &rarr; 3.7568), we applied gradient clipping (<code>max_norm=5.0</code>) across all 5 variants in the main experimental suite (<code>Run 2</code>).
</p>

<h2>5. Proposed Improvement and Its Rationale</h2>

<div class="fig-container">
  ${fig2Svg}
</div>

<p>
To prevent corrupted acoustic or visual signals from contaminating textual representations across 64 GCNII layers and at the readout layer, we propose <strong><code>GatedMM_GCN</code></strong>:
</p>
<ol>
  <li><strong>Directed Cross-Modal Sigmoid Gate:</strong> For each of the 6 directed inter-modal pairs (<i>m &larr; n</i>) where <i>m, n</i> &isin; {0 (a), 1 (v), 2 (l)} and <i>m &ne; n</i>, we apply modality-specific Layer Normalization <i>&tilde;h<sub>i</sub><sup>m</sup></i> = LN<sub><i>m</i></sub>(<i>h<sub>i</sub><sup>m</sup></i>) &isin; R<sup>200</sup> and log-energy <i>r<sub>i</sub><sup>m</sup></i> = log(1 + &Vert;<i>h<sub>i</sub><sup>m</sup></i>&Vert;<sub>2</sub>/&radic;200), feeding a 603-d vector into a bottleneck MLP (603 &rarr; 64 &rarr; 1):
  <div class="math-block">
    g<sub>m&larr;n</sub>(i) = &sigma;( W<sub>mn</sub><sup>(2)</sup> ReLU( W<sub>mn</sub><sup>(1)</sup> [ &tilde;h<sub>i</sub><sup>m</sup> &Vert; &tilde;h<sub>i</sub><sup>n</sup> &Vert; (&tilde;h<sub>i</sub><sup>m</sup> &odot; &tilde;h<sub>i</sub><sup>n</sup>) &Vert; cos(h<sub>i</sub><sup>m</sup>, h<sub>i</sub><sup>n</sup>) &Vert; r<sub>i</sub><sup>m</sup> &Vert; r<sub>i</sub><sup>n</sup> ] + b<sub>mn</sub><sup>(1)</sup> ) + b<sub>mn</sub><sup>(2)</sup> ), &nbsp; A'<sub>mn</sub>(i, i) = g<sub>m&larr;n</sub>(i) &middot; A<sub>mn</sub>(i, i)
  </div>
  We initialize <i>W<sub>mn</sub></i><sup>(2)</sup> = 0, <i>b<sub>mn</sub></i><sup>(2)</sup> = +2.0 so <i>g<sub>m&larr;n</sub></i> &approx; 0.88 at Epoch 0 (warm-starting from standard MMGCN).</li>
  <li><strong>Readout Reliability Modulation:</strong> Post-GCN audio and visual features <i>f<sub>i</sub><sup>a</sup>, f<sub>i</sub><sup>v</sup></i> &isin; R<sup>300</sup> are scaled by the text-anchored gates prior to concatenation: <i>Z<sub>i</sub></i><sup>gated</sup> = [<i>g<sub>l&larr;a</sub></i>(<i>i</i>) <i>f<sub>i</sub><sup>a</sup></i> &Vert; <i>g<sub>l&larr;v</sub></i>(<i>i</i>) <i>f<sub>i</sub><sup>v</sup></i> &Vert; <i>f<sub>i</sub><sup>l</sup></i>] &isin; R<sup>900</sup>.</li>
  <li><strong>Training-Time Modality Corruption Augmentation (<code>G1</code> vs. Control <code>B1</code>):</strong> During training (<i>p</i> = 0.30), each utterance independently undergoes random audio/visual zeroing (7.5%) or Gaussian noise injection (&sigma; = 1.0 &middot; std<sub>train</sub>, 12%). Evaluating <code>B1</code> (un-gated MMGCN with the exact same augmentation) isolates the effect of the gate from the effect of augmentation.</li>
  <li><strong>Minority-Class Calibration (Interventions 2A &amp; 2B):</strong> (2A) <em>Post-Hoc Logit Adjustment</em> subtracts &tau; log &pi;<sub>c</sub> from test log-probabilities using &tau; &isin; [0, 1] tuned <strong>exclusively on validation</strong>; (2B) <code>G1_CW</code> trains <code>G1</code> with dampened inverse square-root class weights <i>w<sub>c</sub></i> &prop; (<i>N / N<sub>c</sub></i>)<sup>1/2</sup>.</li>
</ol>

<h2>6. Experimental Design, Datasets, Splits, Baselines, Metrics, and Resources</h2>
<p>
All five variants (<code>B0, B1, G0, G1, G1_CW</code>) are trained under matched conditions on a single <strong>NVIDIA Tesla T4 GPU</strong> (<code>SEED=27350</code>, 60 epochs, batch size 16, Adam lr = 0.001, dropout 0.4, 64 GCNII layers, dialogue-level 90%/10% train/val split). Each selected checkpoint is evaluated once across <strong>10 test conditions</strong> (using fixed generator <code>SEED+999</code>): <code>clean</code>, Gaussian noise (&sigma; &isin; {1.0, 2.0} on audio, visual, or both), complete sensor dropout (<code>audio_zero</code>, <code>visual_zero</code>, <code>av_zero</code>), and utterance-level temporal misalignment (<code>av_swap</code>).
</p>

<h2>7. Results, Controls, Uncertainty, and Failure Analysis</h2>

<table>
  <caption><strong>Table 4: Clean Test Performance, Minority-Class Recovery (Standard vs. Logit-Adjusted), and Resource Overhead on Tesla T4 <span class="tag tag-m">[Measured]</span></strong></caption>
  <thead>
    <tr>
      <th>Model Variant</th>
      <th>Best Ep</th>
      <th>Val W-F1</th>
      <th>W-F1 (Std)</th>
      <th>Macro-F1 (Std)</th>
      <th><code>fear</code> (Std)</th>
      <th><code>disgust</code> (Std)</th>
      <th>&tau;<sub>val</sub></th>
      <th>W-F1 (LA)</th>
      <th>Macro-F1 (LA)</th>
      <th><code>fear</code> (LA)</th>
      <th><code>disgust</code> (LA)</th>
      <th>Params</th>
      <th>Train (s)</th>
      <th>VRAM (MiB)</th>
      <th>Infer (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong><code>B0</code> (Baseline)</strong></td>
      <td class="num">34</td><td class="num">70.07</td>
      <td class="num">56.60</td><td class="num">34.72</td><td class="num">3.51</td><td class="num warn">0.00</td>
      <td class="num">0.25</td><td class="num">54.45</td><td class="num">35.24</td><td class="num">5.31</td><td class="num">2.15</td>
      <td class="num">2,552,217</td><td class="num">1,067.7</td><td class="num">149.8</td><td class="num">0.647</td>
    </tr>
    <tr>
      <td><strong><code>B1</code> (Aug Control)</strong></td>
      <td class="num">34</td><td class="num">69.45</td>
      <td class="num">56.68</td><td class="num">34.19</td><td class="num warn">0.00</td><td class="num warn">0.00</td>
      <td class="num">0.30</td><td class="num">53.73</td><td class="num best">35.71</td><td class="num best">5.87</td><td class="num">4.08</td>
      <td class="num">2,552,217</td><td class="num">1,065.2</td><td class="num best">148.6</td><td class="num best">0.554</td>
    </tr>
    <tr>
      <td><strong><code>G0</code> (Gate Ablation)</strong></td>
      <td class="num">37</td><td class="num">70.38</td>
      <td class="num">56.52</td><td class="num">34.13</td><td class="num warn">0.00</td><td class="num warn">0.00</td>
      <td class="num">0.00</td><td class="num">56.52</td><td class="num">34.13</td><td class="num warn">0.00</td><td class="num warn">0.00</td>
      <td class="num">2,785,743</td><td class="num">1,655.0</td><td class="num">162.1</td><td class="num">0.915</td>
    </tr>
    <tr>
      <td><strong><code>G1</code> (Proposed)</strong></td>
      <td class="num">28</td><td class="num">69.95</td>
      <td class="num best">57.56</td><td class="num best">35.32</td><td class="num warn">0.00</td><td class="num warn">0.00</td>
      <td class="num">0.35</td><td class="num best">55.21</td><td class="num">35.41</td><td class="num">4.65</td><td class="num warn">0.00</td>
      <td class="num">2,785,743</td><td class="num">1,646.1</td><td class="num">161.9</td><td class="num">0.857</td>
    </tr>
    <tr>
      <td><strong><code>G1_CW</code> (Ablation 2B)</strong></td>
      <td class="num">56</td><td class="num best">70.73</td>
      <td class="num">55.78</td><td class="num">34.51</td><td class="num best">3.85</td><td class="num best">3.70</td>
      <td class="num">0.20</td><td class="num">54.53</td><td class="num">35.06</td><td class="num">4.95</td><td class="num best">6.70</td>
      <td class="num">2,785,743</td><td class="num">1,657.4</td><td class="num">162.1</td><td class="num">0.790</td>
    </tr>
  </tbody>
</table>

<table>
  <caption><strong>Table 5: Robustness Evaluation — Test Weighted-F1 (%) Across 10 Clean and Corrupted Modality Conditions <span class="tag tag-m">[Measured]</span></strong></caption>
  <thead>
    <tr>
      <th>Corruption Condition</th>
      <th><code>B0</code> (Baseline)</th>
      <th><code>B1</code> (Control: Aug)</th>
      <th><code>G0</code> (Ablation: Gate)</th>
      <th><strong><code>G1</code> (Proposed)</strong></th>
      <th><code>G1_CW</code> (2B)</th>
      <th>Gain (<code>G1</code> &minus; <code>B0</code>)</th>
      <th>Gain (<code>G1</code> &minus; <code>B1</code>)</th>
      <th><code>B0</code> Drift</th>
      <th><code>G0</code> Drift</th>
      <th><code>G1</code> Drift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>clean</code> (Unperturbed)</td>
      <td class="num">56.60</td><td class="num">56.68</td><td class="num">56.52</td><td class="num best">57.56</td><td class="num">55.78</td>
      <td class="num best">+0.96</td><td class="num best">+0.88</td><td class="num">0.0000</td><td class="num">0.0000</td><td class="num">0.0000</td>
    </tr>
    <tr>
      <td><code>audio_noise_1</code> (&sigma;=1.0)</td>
      <td class="num">56.43</td><td class="num">56.85</td><td class="num">56.31</td><td class="num best">57.38</td><td class="num">55.79</td>
      <td class="num best">+0.95</td><td class="num best">+0.53</td><td class="num">0.0005</td><td class="num">0.0157</td><td class="num">0.0087</td>
    </tr>
    <tr>
      <td><code>audio_noise_2</code> (&sigma;=2.0)</td>
      <td class="num">55.98</td><td class="num best">56.72</td><td class="num">56.62</td><td class="num">56.62</td><td class="num">55.31</td>
      <td class="num">+0.64</td><td class="num">-0.10</td><td class="num">0.0018</td><td class="num">0.0450</td><td class="num">0.0222</td>
    </tr>
    <tr>
      <td><code>audio_zero</code> (<i>a</i> = 0)</td>
      <td class="num">55.70</td><td class="num">55.94</td><td class="num">56.17</td><td class="num best">56.70</td><td class="num">55.74</td>
      <td class="num best">+1.00</td><td class="num best">+0.76</td><td class="num">0.0160</td><td class="num">0.0533</td><td class="num">0.0633</td>
    </tr>
    <tr>
      <td><code>visual_noise_1</code> (&sigma;=1.0)</td>
      <td class="num">56.70</td><td class="num">56.82</td><td class="num">56.58</td><td class="num best">57.61</td><td class="num">55.64</td>
      <td class="num best">+0.91</td><td class="num best">+0.79</td><td class="num">0.0234</td><td class="num">0.0359</td><td class="num">0.0245</td>
    </tr>
    <tr>
      <td><code>visual_noise_2</code> (&sigma;=2.0)</td>
      <td class="num">56.82</td><td class="num">56.76</td><td class="num">56.53</td><td class="num best">57.72</td><td class="num">55.47</td>
      <td class="num best">+0.90</td><td class="num best">+0.96</td><td class="num">0.0577</td><td class="num">0.0782</td><td class="num best">0.0553</td>
    </tr>
    <tr>
      <td><code>visual_zero</code> (<i>v</i> = 0)</td>
      <td class="num">56.82</td><td class="num">56.32</td><td class="num">56.12</td><td class="num best">57.01</td><td class="num">54.97</td>
      <td class="num">+0.19</td><td class="num best">+0.69</td><td class="num">0.0755</td><td class="num">0.1815</td><td class="num">0.1448</td>
    </tr>
    <tr>
      <td><code>av_noise_1</code> (<i>a,v</i> &sigma;=1.0)</td>
      <td class="num">56.57</td><td class="num">56.95</td><td class="num">56.67</td><td class="num best">57.79</td><td class="num">55.49</td>
      <td class="num best">+1.22</td><td class="num best">+0.84</td><td class="num">0.0253</td><td class="num">0.0422</td><td class="num">0.0283</td>
    </tr>
    <tr>
      <td><code>av_zero</code> (<i>a</i>=0, <i>v</i>=0)</td>
      <td class="num">55.86</td><td class="num">55.75</td><td class="num">55.86</td><td class="num best">56.76</td><td class="num">54.92</td>
      <td class="num best">+0.90</td><td class="num best">+1.01</td><td class="num">0.0742</td><td class="num">0.2053</td><td class="num">0.1806</td>
    </tr>
    <tr>
      <td><code>av_swap</code> (Misaligned <i>a,v</i>)</td>
      <td class="num">55.39</td><td class="num">55.62</td><td class="num">55.85</td><td class="num best">56.23</td><td class="num">55.62</td>
      <td class="num best">+0.84</td><td class="num best">+0.61</td><td class="num">0.0477</td><td class="num">0.1360</td><td class="num">0.1192</td>
    </tr>
    <tr>
      <td><strong>Mean (9 Corruptions)</strong></td>
      <td class="num"><strong>56.25</strong></td><td class="num"><strong>56.41</strong></td><td class="num"><strong>56.30</strong></td><td class="num best"><strong>57.09</strong></td><td class="num"><strong>55.44</strong></td>
      <td class="num best"><strong>+0.84</strong></td><td class="num best"><strong>+0.68</strong></td><td class="num">0.0358</td><td class="num">0.0881</td><td class="num">0.0719</td>
    </tr>
  </tbody>
</table>

<div class="fig-container">
  ${fig3Svg}
</div>

<div class="fig-container">
  <img src="${fig4Png}" alt="MELD Robustness Bar Chart"/>
  <div class="fig-caption"><strong>Figure 4:</strong> Test Weighted-F1 (%) across all 10 clean and corrupted evaluation conditions on MELD (generated from <code>results/robustness_comparison.csv</code>).</div>
</div>

<div class="fig-container">
  <img src="${fig5Png}" alt="Minority Class F1 Recovery"/>
  <div class="fig-caption"><strong>Figure 5:</strong> Recovery of collapsed minority classes (<code>fear</code> and <code>disgust</code>) and Macro-F1 via Post-Hoc Logit Adjustment (2A) and Class-Weighted Training (2B).</div>
</div>

<h3>7.1 Synthesis of Empirical Findings &amp; Honest Analysis of Unfavorable Cases</h3>
<ul>
  <li><strong>Synergy of Gating and Augmentation (<code>G1</code> vs. Controls <code>B1</code> &amp; <code>G0</code>):</strong> Neither augmentation alone (<code>B1</code>: 56.68% clean, 56.41% corrupted mean) nor gating without corruption exposure (<code>G0</code>: 56.52% clean, 56.30% corrupted mean) suffices on its own. Combining both in <strong><code>G1</code></strong> yields the highest clean Weighted-F1 (<strong>57.56%</strong>, +0.96 over <code>B0</code>) and the highest score in <strong>8 of 9 corruption modes</strong> (mean <strong>57.09%</strong>, +0.84 over <code>B0</code> and +0.68 over <code>B1</code>) with minimal resource cost (+9.15% params, +12.3 MiB VRAM) <span class="tag tag-m">[Measured]</span>.</li>
  <li><strong>Mechanistic Discovery — Spontaneous Directional Shielding (Figure 3A):</strong> Although all 6 gates were initialized at &sigma;(+2.0) &approx; 0.88, every gated model (<code>G0, G1, G1_CW</code>) converged to a strongly <strong>asymmetric regime</strong> <span class="tag tag-m">[Measured]</span>: inflows into Text from Audio (<code>2_0</code>: 0.24–0.38) and Visual (<code>2_1</code>: 0.46–0.53) are heavily suppressed, whereas outflows from Text into Audio (<code>0_2</code>: 0.84–0.98) and Visual (<code>1_2</code>: 0.69–1.00) stay wide open. The network discovers that MELD text features are reliable while acoustic/visual features are noisy, shielding the text branch while still allowing lexical context to refine audio/visual nodes <span class="tag tag-i">[Interpretation]</span>.</li>
  <li><strong>Unfavorable Mechanistic Finding — Zero-Vector Bias Rebound (Table 5 &amp; Figure 3A):</strong> Under complete zeroing (<code>audio_zero</code>), <i>g<sub>l&larr;a</sub></i> unexpectedly rises from 0.378 &rarr; 0.621 and post-GCN text drift &Vert;<i>h<sub>l</sub></i><sup>corrupted</sup> &minus; <i>h<sub>l</sub></i><sup>clean</sup>&Vert;<sub>2</sub> is higher in <code>G1</code> (0.0633) than <code>B0</code> (0.0160) <span class="tag tag-m">[Measured]</span>. Mathematically, when <i>x<sub>i</sub><sup>a</sup></i> = 0, LayerNorm(<i>h<sub>i</sub><sup>a</sup></i>) zeroes out the centered activations inside the gate MLP, causing the linear products to vanish and reverting the gate output toward its positive initial bias <i>b</i><sup>(2)</sup> = +2.0 <span class="tag tag-i">[Interpretation]</span>. Nonetheless, comparing <code>G1</code> against <code>G0</code> confirms that training augmentation consistently reduces drift across 8 of 9 corruptions (mean drift 0.0719 vs. 0.0881), and under high-variance visual noise (<code>visual_noise_2</code>), <code>G1</code> achieves lower drift (0.0553) than both <code>G0</code> (0.0782) and <code>B0</code> (0.0577).</li>
  <li><strong>Failure Analysis of Minority Emotions (<code>fear</code> &amp; <code>disgust</code>) &amp; Validation-Test Gap:</strong> Confusion matrices (<code>results/all_summaries.json</code>) show that standard <code>G1</code> predicts <code>fear</code> only once and <code>disgust</code> zero times across 2,610 test utterances, misclassifying them primarily into <code>neutral</code> (48/118) and <code>anger</code> (38/118). Post-hoc logit adjustment on <code>B1</code> (&tau; = 0.30) recovers 10 true positives on <code>fear</code> (5.87% F1), 2 on <code>disgust</code> (4.08% F1), raises <code>sadness</code> F1 from 19.87% &rarr; 29.02%, and achieves the highest overall Macro-F1 (35.71%), while class-weighted <code>G1_CW</code> + Logit Adjustment (&tau; = 0.20) achieves the highest <code>disgust</code> F1 (6.70%) alongside 4.95% <code>fear</code> F1, trading off ~1.8–2.9 points of majority-dominated Weighted-F1 <span class="tag tag-m">[Measured]</span>. Furthermore, the ~12.5-point gap between validation W-F1 (~70.0%) and test W-F1 (~57.5%) arises because the pre-extracted TextCNN features in <code>MELD_features_raw1.pkl</code> were trained on <code>trainVid</code>, giving any validation subset drawn from <code>trainVid</code> an in-sample representation advantage over <code>testVid</code> <span class="tag tag-i">[Interpretation]</span>.</li>
</ul>

<h2>8. Limitations, Conclusions, and Unresolved Questions</h2>
<ul>
  <li><strong>Limitations:</strong> (1) Pre-extracted OpenSMILE/DenseNet features in MELD carry weak emotion signal (even <code>av_zero</code> reduces <code>B0</code> W-F1 by only 0.74 points), limiting the ceiling of cross-modal gating compared to acoustic-rich datasets like IEMOCAP (excluded due to USC license); (2) single-seed (<code>27350</code>) execution across the 5-variant matrix on Colab T4 (~118 min total), though our two <code>B0</code> runs (57.44% vs. 56.60%) quantify optimization variance at ~0.8 points; (3) the <code>LayerNorm</code> fallback-to-bias behavior under exact zero vectors.</li>
  <li><strong>Conclusions &amp; Unresolved Questions:</strong> Directed cross-modal edge gating paired with corruption augmentation (<code>G1</code>) improves both clean accuracy (57.56% W-F1) and corrupted robustness (57.09% mean W-F1) while revealing strong directional modality asymmetry on MELD. Future work should (i) replace <code>LayerNorm</code>+bias with a multiplicative norm gate tanh(&Vert;<i>h<sub>n</sub></i>&Vert;<sub>2</sub> / &tau;<sub>norm</sub>) so zeroed sensors force <i>g &rarr; 0</i>, and (ii) evaluate <code>GatedMM_GCN</code> with modern RoBERTa/WavLM backbones.</li>
  <li><strong>Reproducibility &amp; Colab Execution Instructions:</strong> All code, configurations, and saved outputs (<code>results/all_summaries.json</code>, <code>results/robustness_comparison.csv</code>, <code>results/gate_activations.csv</code>, per-variant <code>history.csv</code>) are included in the repository. To run from scratch on Google Colab (<strong>Tesla T4 GPU</strong>, &ge; 2 GB VRAM, Python 3.13, PyTorch 2.14, <code>torch-geometric==2.8.1</code>): upload <code>MMGCN_MELD_Colab_Experiments.ipynb</code>, set <code>SMOKE_TEST = True</code> in Cell 3 for a 45-second verification or <code>SMOKE_TEST = False</code> for the full 60-epoch 5-variant run, and execute all cells. Locally, run <code>node scripts/generate_report_assets.js</code> and <code>node scripts/build_pdf_report.js</code> to regenerate all figures and this PDF.</li>
  <li><strong>Integrity &amp; AI-Use Disclosure (§7):</strong> In accordance with course policy, I disclose that <em>Antigravity AI Assistant</em> was used to assist with legacy PyTorch 1.1 compatibility patching, coding the <code>GatedMM_GCN</code> module and evaluation scripts, and formatting report tables/SVGs. All Colab GPU experiments were executed by the author (<strong>Samet Çetinkaya, 26526003</strong>), and every reported number, citation, and claim was manually verified against the raw execution outputs and original papers.</li>
</ul>

<p style="font-size:7.9pt; color:#475569; margin-top:6px;">
<strong>References:</strong> [1] Ghosal et al., <em>DialogueGCN</em>, EMNLP 2019 (arXiv:1908.11540). [2] Hu et al., <em>MMGCN</em>, ACL 2021 (arXiv:2107.06779). [3] Joshi et al., <em>COGMEN</em>, NAACL 2022 (arXiv:2205.02455). [4] Hu et al., <em>MM-DFN</em>, ICASSP 2022 (arXiv:2203.02385). [5] Li et al., <em>GraphMFT</em>, arXiv:2208.00339 (2022). [6] Poria et al., <em>MELD</em>, ACL 2019. [7] Chen et al., <em>GCNII</em>, ICML 2020. [8] Menon et al., <em>Long-Tail Learning via Logit Adjustment</em>, ICLR 2021.
</p>

</body>
</html>`;

const htmlPath = path.join(ROOT, 'results', '_report_print.html');
fs.writeFileSync(htmlPath, html, 'utf8');

const chromeCandidates = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe'
];
const browserExe = chromeCandidates.find(p => fs.existsSync(p));
if (!browserExe) {
  throw new Error('No Chromium browser found for PDF rendering.');
}

const pdfPath = path.join(ROOT, 'Samet_Çetinkaya_YZM5110_Research.pdf');
const fileUrl = 'file:///' + htmlPath.replace(/\\/g, '/');

execFileSync(browserExe, [
  '--headless=new',
  '--disable-gpu',
  '--no-pdf-header-footer',
  `--print-to-pdf=${pdfPath}`,
  fileUrl
]);

if (fs.existsSync(htmlPath)) fs.unlinkSync(htmlPath);

const stat = fs.statSync(pdfPath);
console.log('Successfully generated PDF:', path.basename(pdfPath), 'Size (bytes):', stat.size);
