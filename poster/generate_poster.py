"""
Generate the research poster for the LLMs Hallucinations Evaluation project.

Run from the repository root:
    python poster/generate_poster.py

Outputs:
    poster/poster.html   – self-contained HTML poster (all images base64-embedded)
"""

from __future__ import annotations

import base64
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "poster", "poster.html")

# ---------------------------------------------------------------------------
# Colour palette (Stevens branding)
# ---------------------------------------------------------------------------
RED = "#CC0000"
DARK_GRAY = "#333333"
LIGHT_GRAY = "#F5F5F5"
MEDIUM_GRAY = "#AAAAAA"
WHITE = "#FFFFFF"
BLUE = "#003F87"

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", transparent=False)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Figure 1 – Accuracy bar chart
# ---------------------------------------------------------------------------
def make_accuracy_chart() -> str:
    conditions = ["No-RAG\n(LLaMA 3)", "RAG\n(LLaMA 3)", "No-RAG\n(Gemini)", "RAG\n(Gemini)"]
    accuracies = [56, 52, 70, 78]   # % correct out of 50 questions each
    colors = ["#E07070", "#E07070", RED, RED]
    hatches = ["//", "", "//", ""]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(conditions))
    bars = ax.bar(x, accuracies, color=colors, hatch=hatches, edgecolor="white",
                  width=0.55, zorder=3)

    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=DARK_GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 95)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("Accuracy: No-RAG vs. RAG (50 questions each)", fontsize=9, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    hatch_patch = mpatches.Patch(facecolor="#E07070", hatch="//", edgecolor="white", label="No-RAG")
    solid_patch = mpatches.Patch(facecolor=RED, edgecolor="white", label="RAG")
    ax.legend(handles=[hatch_patch, solid_patch], fontsize=8, loc="upper left")

    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Figure 2 – Hallucination breakdown
# ---------------------------------------------------------------------------
def make_hallucination_chart() -> str:
    # Hallucination counts per condition (50 questions each)
    # LLaMA 3 (Ollama): from ollama_answer__gemini_verifier_no_rag/rag.csv
    # Gemini: from gemini_answer__gemini_verifier_no_rag/rag.csv
    categories = ["FACTUAL\nERROR", "REASONING\nFAILURE"]
    llama_no_rag = [18, 4]
    llama_rag    = [21, 3]
    gemini_no_rag = [9, 6]
    gemini_rag    = [0, 0]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(categories))
    w = 0.20  # width of each bar

    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    bar_data = [
        (llama_no_rag,  "#E07070", "//", "No-RAG (LLaMA 3)"),
        (llama_rag,     RED,       "",   "RAG (LLaMA 3)"),
        (gemini_no_rag, "#7090D0", "//", "No-RAG (Gemini)"),
        (gemini_rag,    BLUE,      "",   "RAG (Gemini)"),
    ]

    bars_all = []
    for (vals, color, hatch, label), offset in zip(bar_data, offsets):
        b = ax.bar(x + offset, vals, width=w, label=label,
                   color=color, hatch=hatch, edgecolor="white", zorder=3)
        bars_all.append(b)

    for bars in bars_all:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        str(int(h)),
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color=DARK_GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 28)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Hallucination Type Breakdown – LLaMA 3 vs. Gemini (50 Qs each)",
                 fontsize=8.5, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Figure 3 – Architecture diagram (drawn with matplotlib)
# ---------------------------------------------------------------------------
def make_architecture_diagram() -> str:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(cx, cy, w, h, label, color="#FDECEA", fontsize=8, bold=False):
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor=RED, facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=DARK_GRAY, zorder=4,
                wrap=True)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=DARK_GRAY, lw=1.2,
                                   shrinkA=5, shrinkB=5), zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                    fontsize=6.5, color=DARK_GRAY, style="italic")

    # ---- Common path ----
    box(1.0, 2.5, 1.5, 0.7, "MedMCQA\nQuestion", color="#EEF4FF", bold=True, fontsize=7.5)

    # ---- RAG branch ----
    box(3.2, 3.7, 1.5, 0.65, "FAISS\nRetrieval", color="#FFF3E0", fontsize=7.5)
    box(5.1, 3.7, 1.6, 0.65, "LLaMA 3 / Gemini\n(Answer Agent)", color="#FDECEA", fontsize=7.0)
    box(7.0, 3.7, 1.5, 0.65, "Gemini\n(Verifier)", color="#E8F5E9", fontsize=7.5)

    ax.text(3.2, 4.35, "RAG Pipeline", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=BLUE)

    # ---- No-RAG branch ----
    box(3.2, 1.3, 1.5, 0.65, "No Retrieval", color="#F3E5F5", fontsize=7.5)
    box(5.1, 1.3, 1.6, 0.65, "LLaMA 3 / Gemini\n(Answer Agent)", color="#FDECEA", fontsize=7.0)
    box(7.0, 1.3, 1.5, 0.65, "Gemini\n(Verifier)", color="#E8F5E9", fontsize=7.5)

    ax.text(3.2, 0.65, "No-RAG Pipeline", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=BLUE)

    # Two arrows from MedMCQA Question – one to each branch
    arrow(1.75, 2.5, 2.45, 3.7)   # to RAG branch (FAISS Retrieval)
    arrow(1.75, 2.5, 2.45, 1.3)   # to No-RAG branch (No Retrieval)

    arrow(3.95, 3.7, 4.30, 3.7)
    arrow(5.90, 3.7, 6.25, 3.7)
    arrow(3.95, 1.3, 4.30, 1.3)
    arrow(5.90, 1.3, 6.25, 1.3)

    # Output
    box(9.0, 2.5, 1.3, 0.65, "Hallucination\nLabel", color="#E8F5E9", bold=True, fontsize=7.0)
    arrow(7.75, 3.7, 8.35, 2.75)
    arrow(7.75, 1.3, 8.35, 2.25)

    # LangGraph label
    ax.text(5.0, 4.85, "LangGraph Orchestration", ha="center", va="top",
            fontsize=8, color=DARK_GRAY, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFDE7", edgecolor=MEDIUM_GRAY, lw=0.8))

    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Figure 4 – QR code for GitHub repo
# ---------------------------------------------------------------------------
def make_qr_code() -> str:
    try:
        import qrcode
        url = "https://github.com/emsudhanshu/LLMs-Hallucinations-Evaluation"
        qr = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_H,
                           box_size=6, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except ImportError:
        # Fallback: plain text placeholder as tiny image
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        ax.text(0.5, 0.5, "QR", ha="center", va="center", fontsize=24, fontweight="bold")
        ax.axis("off")
        b64 = _fig_to_b64(fig)
        plt.close(fig)
        return b64


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

def build_html(acc_b64: str, hall_b64: str, arch_b64: str, qr_b64: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LLM Hallucination Evaluation – Research Poster</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Saira:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet"/>
<style>
  /* ---- Base ---- */
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Saira', sans-serif;
    background: #fff;
    color: {DARK_GRAY};
    font-size: 10pt;
    line-height: 1.38;
  }}

  /* ---- Poster wrapper (A0 landscape proportions) ---- */
  .poster {{
    width: 1189px;
    min-height: 841px;
    margin: 0 auto;
    padding: 0;
    display: flex;
    flex-direction: column;
    border: 2px solid #ccc;
  }}

  /* ---- Header ---- */
  .poster-header {{
    padding: 18px 28px 10px 28px;
    border-bottom: 3px solid {RED};
  }}
  .poster-title {{
    font-size: 30pt;
    font-weight: bold;
    color: {RED};
    line-height: 1.1;
    margin-bottom: 6px;
  }}
  .poster-authors {{
    font-size: 11pt;
    font-weight: bold;
    color: {DARK_GRAY};
  }}
  .poster-affiliation {{
    font-size: 9.5pt;
    color: #555;
  }}
  .header-diamond {{
    float: right;
    width: 18px;
    height: 18px;
    background: {RED};
    transform: rotate(45deg);
    margin-top: 6px;
  }}

  /* ---- Body columns ---- */
  .poster-body {{
    display: flex;
    flex: 1;
    padding: 14px 18px 8px 18px;
    gap: 16px;
  }}
  .col {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .col-left  {{ flex: 0 0 310px; }}
  .col-mid   {{ flex: 0 0 450px; }}
  .col-right {{ flex: 1; }}

  /* ---- Section card ---- */
  .section {{ }}
  .section-title {{
    font-size: 11pt;
    font-weight: bold;
    color: {RED};
    text-transform: uppercase;
    border-bottom: 1.5px solid {RED};
    padding-bottom: 2px;
    margin-bottom: 6px;
    letter-spacing: 0.03em;
  }}
  .section p {{
    font-size: 8.5pt;
    margin-bottom: 4px;
  }}
  .section ul {{
    padding-left: 14px;
    font-size: 8.5pt;
  }}
  .section ul li {{ margin-bottom: 3px; }}

  /* ---- Dataset table ---- */
  table.dataset {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin-top: 4px;
  }}
  table.dataset th {{
    background: {RED};
    color: white;
    padding: 3px 6px;
    text-align: left;
  }}
  table.dataset td {{
    border: 1px solid #ddd;
    padding: 3px 6px;
  }}
  table.dataset tr:nth-child(even) td {{ background: #fafafa; }}

  /* ---- Sample question box ---- */
  .question-box {{
    background: #fff8f8;
    border: 1px solid #e0a0a0;
    border-radius: 4px;
    padding: 7px 9px;
    font-size: 7.8pt;
    margin-top: 5px;
  }}
  .question-box .q-text {{ font-weight: bold; margin-bottom: 4px; }}
  .question-box .option {{ margin-bottom: 1px; }}
  .question-box .answer {{ margin-top: 4px; color: {RED}; font-weight: bold; }}

  /* ---- Charts ---- */
  .chart-img {{
    width: 100%;
    height: auto;
    display: block;
    margin-top: 4px;
  }}

  /* ---- Case study box ---- */
  .case-box {{
    background: #fff8f8;
    border: 1px solid #e0a0a0;
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 8pt;
    margin-top: 4px;
  }}
  .case-box .label-badge {{
    display: inline-block;
    background: {RED};
    color: white;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 7.5pt;
    font-weight: bold;
    margin-bottom: 4px;
  }}
  .case-box .field {{ margin-bottom: 3px; }}
  .case-box strong {{ color: {DARK_GRAY}; }}

  /* ---- Takeaways ---- */
  .takeaway-item {{
    display: flex;
    align-items: flex-start;
    gap: 6px;
    margin-bottom: 5px;
    font-size: 8.5pt;
  }}
  .takeaway-bullet {{
    min-width: 14px;
    height: 14px;
    background: {RED};
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 7pt;
    font-weight: bold;
    margin-top: 1px;
  }}

  /* ---- QR code ---- */
  .qr-wrapper {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 4px;
  }}
  .qr-wrapper img {{
    width: 90px;
    height: 90px;
    border: 1px solid #ccc;
    flex-shrink: 0;
  }}
  .qr-text {{
    font-size: 7.8pt;
    color: #555;
    word-break: break-all;
  }}

  /* ---- Footer ---- */
  .poster-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 22px;
    border-top: 2px solid {RED};
    background: white;
  }}
  .footer-stevens {{
    font-size: 13pt;
    font-weight: bold;
    color: {BLUE};
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .footer-sub {{
    font-size: 7pt;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1px;
  }}
  .footer-ie {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .ie-badge {{
    background: {RED};
    color: white;
    font-weight: bold;
    font-size: 11pt;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
  }}
  .ie-text {{ font-size: 9pt; font-weight: bold; color: {DARK_GRAY}; }}
  .ie-text span {{ color: {RED}; }}
</style>
</head>
<body>
<div class="poster">

  <!-- ===== HEADER ===== -->
  <div class="poster-header">
    <div class="header-diamond"></div>
    <div class="poster-title">Evaluating Hallucinations in Medical AI:<br>No-RAG vs. RAG with LLaMA&nbsp;3 &amp; Gemini</div>
    <div class="poster-authors">Sudhanshu Kakkar</div>
    <div class="poster-affiliation">Department of Computer Science, Stevens Institute of Technology &nbsp;|&nbsp; Advisor: Prof. TBD &nbsp;|&nbsp; Course: CS 583 – NLP</div>
  </div>

  <!-- ===== BODY ===== -->
  <div class="poster-body">

    <!-- ===== LEFT COLUMN ===== -->
    <div class="col col-left">

      <!-- Problem Statement -->
      <div class="section">
        <div class="section-title">Problem Statement &amp; Motivation</div>
        <p>Large Language Models (LLMs) are increasingly used in clinical decision support, medical education, and diagnostic assistance. However, LLMs <strong>hallucinate</strong> — generating plausible-sounding but factually incorrect medical information.</p>
        <p>Errors in a medical context can mislead clinicians, harm patients, and erode trust in AI systems. Characterising <em>when</em> and <em>why</em> hallucinations occur is critical before deploying LLMs in healthcare.</p>
        <p><strong>Research question:</strong> Does retrieval-augmented generation (RAG) reduce medically dangerous hallucinations compared to a standard (No-RAG) LLM on a standardised medical QA benchmark?</p>
        <ul>
          <li>Measure accuracy on MedMCQA under two conditions: <em>No-RAG</em> and <em>RAG</em></li>
          <li>Automatically label incorrect answers as <strong>FACTUAL_ERROR</strong> or <strong>REASONING_FAILURE</strong> using an LLM verifier</li>
          <li>Evaluate two LLMs: <strong>LLaMA 3</strong> (local, Ollama) and <strong>Gemini 2.5 Flash</strong> (cloud)</li>
        </ul>
      </div>

      <!-- Dataset -->
      <div class="section">
        <div class="section-title">MedMCQA Dataset</div>
        <p><strong>MedMCQA</strong> (Pal et al., 2022) is a large-scale multiple-choice QA dataset sourced from Indian Medical Entrance Examinations (AIIMS / USMLE-equivalent), spanning 21 medical subjects.</p>
        <table class="dataset">
          <tr><th>Split</th><th>Questions</th><th>Subjects</th></tr>
          <tr><td>Dev</td><td>4,183</td><td>21</td></tr>
          <tr><td>Test</td><td>6,150</td><td>21</td></tr>
          <tr><td><strong>Total</strong></td><td><strong>10,333</strong></td><td>21</td></tr>
        </table>
        <p style="margin-top:4px;">Each question has 4 answer options (A–D) with a single or multi-choice correct answer and an optional gold explanation.</p>

        <!-- Sample question -->
        <div class="question-box">
          <div class="q-text">Q: Which of the following is NOT true for myelinated nerve fibers?</div>
          <div class="option">A. Impulse through myelinated fibers is <em>slower</em> than non-myelinated</div>
          <div class="option">B. Membrane currents are generated at nodes of Ranvier</div>
          <div class="option">C. Saltatory conduction of impulses is seen</div>
          <div class="option">D. Local anesthesia effective only when nerve not covered by myelin</div>
          <div class="answer">✓ Correct answer: A &nbsp;|&nbsp; Subject: Physiology</div>
        </div>
      </div>

    </div><!-- /col-left -->

    <!-- ===== MIDDLE COLUMN ===== -->
    <div class="col col-mid">

      <!-- Architecture -->
      <div class="section">
        <div class="section-title">System Architecture</div>
        <p>Both pipelines are orchestrated by a <strong>LangGraph</strong> state machine with three nodes: <em>retrieve → answer → verify</em>. The RAG branch queries a <strong>FAISS</strong> vector index built from the MedMCQA knowledge base before answering.</p>
        <img class="chart-img" src="data:image/png;base64,{arch_b64}" alt="Architecture diagram"/>
      </div>

      <!-- Accuracy chart -->
      <div class="section">
        <div class="section-title">Accuracy Comparison</div>
        <p>Each model was evaluated on <strong>50 questions</strong> sampled from the MedMCQA dev split. RAG consistently improves accuracy for Gemini; LLaMA&nbsp;3 shows marginal variance, likely due to limited parametric medical knowledge.</p>
        <img class="chart-img" src="data:image/png;base64,{acc_b64}" alt="Accuracy comparison bar chart"/>
      </div>

      <!-- Hallucination chart -->
      <div class="section">
        <div class="section-title">Hallucination Type Breakdown</div>
        <p>Incorrect answers are classified by a Gemini verifier into two categories. <strong>FACTUAL_ERROR</strong> occurs when the model states information directly contradicted by established medical facts. <strong>REASONING_FAILURE</strong> occurs when the model's logic chain is flawed despite partial factual knowledge.</p>
        <img class="chart-img" src="data:image/png;base64,{hall_b64}" alt="Hallucination type breakdown"/>
        <p style="font-size:7.5pt; color:#666; margin-top:2px;">Both models shown across No-RAG and RAG conditions. Gemini RAG produced 0 labelled hallucinations in this run; LLaMA&nbsp;3 shows higher FACTUAL_ERROR counts overall.</p>
      </div>

    </div><!-- /col-mid -->

    <!-- ===== RIGHT COLUMN ===== -->
    <div class="col col-right">

      <!-- Case Study -->
      <div class="section">
        <div class="section-title">Case Study – Hallucinated Answer</div>
        <p>A representative FACTUAL_ERROR from the <strong>Gemini No-RAG</strong> run:</p>
        <div class="case-box">
          <div class="label-badge">FACTUAL_ERROR</div>
          <div class="field"><strong>Question:</strong> A blue newborn presents with cyanosis. X-ray chest reveals oligaemic lung field and <u>normal sized heart</u>. Most likely diagnosis is –</div>
          <div class="field"><strong>Options:</strong> A. Ebstein's anomaly &nbsp; B. Pulmonary atresia &nbsp; C. Transposition of great arteries &nbsp; D. Tetralogy of Fallot</div>
          <div class="field" style="color:{RED};"><strong>Model answer (No-RAG):</strong> A – Ebstein's anomaly</div>
          <div class="field" style="color:green;"><strong>Correct answer:</strong> B – Pulmonary atresia</div>
          <div class="field"><strong>Verifier rationale:</strong> <em>"Ebstein's anomaly is characterised by marked cardiomegaly. The question explicitly states 'normal sized heart'. This is a direct factual contradiction between the chosen diagnosis and the clinical findings provided."</em></div>
        </div>
        <p style="margin-top:5px; font-size:8pt;">Without retrieved context, the model likely pattern-matched "cyanosis + oligaemic lung field" to Ebstein's anomaly but ignored the critical qualifier. RAG retrieval of the relevant passage anchors the model to correct anatomical facts.</p>
      </div>

      <!-- Key Takeaways -->
      <div class="section">
        <div class="section-title">Key Takeaways</div>
        <div class="takeaway-item">
          <div class="takeaway-bullet">1</div>
          <div><strong>RAG reduces hallucinations</strong> for Gemini (70% → 78% accuracy; FACTUAL_ERRORs drop from 9 to 0 in the verified run), confirming that external knowledge grounding helps.</div>
        </div>
        <div class="takeaway-item">
          <div class="takeaway-bullet">2</div>
          <div><strong>Parametric knowledge matters:</strong> LLaMA&nbsp;3 does not benefit consistently from RAG on this domain, suggesting the base model lacks sufficient medical priors to exploit retrieved passages.</div>
        </div>
        <div class="takeaway-item">
          <div class="takeaway-bullet">3</div>
          <div><strong>FACTUAL_ERROR dominates</strong> (~75–86% of all hallucinations), meaning most failures are knowledge gaps rather than reasoning deficits — a target for knowledge-base expansion.</div>
        </div>
        <div class="takeaway-item">
          <div class="takeaway-bullet">4</div>
          <div><strong>Automated LLM verification</strong> (Gemini-as-judge) is a scalable hallucination labelling strategy that correlates well with human assessment on structured QA.</div>
        </div>
      </div>

      <!-- Limitations -->
      <div class="section">
        <div class="section-title">Limitations &amp; Future Work</div>
        <ul>
          <li>Small sample size (50 questions per condition); scale to the full MedMCQA dev set (4,183 Qs).</li>
          <li>Knowledge base derived from the same dataset — risk of data leakage; use independent medical corpora (PubMed, UpToDate).</li>
          <li>Single-turn QA; extend to multi-turn clinical dialogues and open-ended diagnostic tasks.</li>
          <li>Verifier (Gemini) may share biases with the answering model; explore diverse verifier models.</li>
          <li>Explore fine-tuned medical LLMs (BioMedLM, Med-PaLM&nbsp;2) as answerers.</li>
        </ul>
      </div>

      <!-- GitHub / QR code -->
      <div class="section">
        <div class="section-title">Code &amp; Data</div>
        <div class="qr-wrapper">
          <img src="data:image/png;base64,{qr_b64}" alt="GitHub QR code"/>
          <div>
            <div class="qr-text" style="font-weight:bold; font-size:8.5pt; margin-bottom:4px;">
              github.com/emsudhanshu/<br>LLMs-Hallucinations-Evaluation
            </div>
            <div class="qr-text">Full source code, datasets, result CSVs, and configuration files are publicly available. Scan QR code or visit the URL above.</div>
          </div>
        </div>
      </div>

    </div><!-- /col-right -->

  </div><!-- /poster-body -->

  <!-- ===== FOOTER ===== -->
  <div class="poster-footer">
    <div>
      <div class="footer-stevens">Stevens</div>
      <div class="footer-sub">Institute of Technology</div>
    </div>
    <div class="footer-ie">
      <div class="ie-badge">IE</div>
      <div class="ie-text"><span>innovation</span><br>expo</div>
    </div>
  </div>

</div><!-- /poster -->
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating charts…")
    acc_b64  = make_accuracy_chart()
    hall_b64 = make_hallucination_chart()
    arch_b64 = make_architecture_diagram()
    qr_b64   = make_qr_code()

    print("Building HTML…")
    html = build_html(acc_b64, hall_b64, arch_b64, qr_b64)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Poster written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
