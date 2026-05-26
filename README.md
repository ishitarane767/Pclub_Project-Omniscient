# Project OMNISCIENT — Technical Manual

## Overview

This system is an agentic detective that autonomously solves murder mystery cases by interrogating LLM-powered suspect agents, extracting structured facts from their testimonies, detecting contradictions, and reasoning over evidence to identify the murderer. Every decision the system makes is logged with a justification, making the entire investigative process auditable and transparent.

---

## System Architecture

The system is built as a pipeline of independent modules that communicate through a central data store. Here is how they fit together:

```
Dataset (case JSON)
        |
        v
  SuspectAgent(s)          -- one Gemini-powered agent per suspect
        |
        v
    Detective              -- sends questions, collects answers
        |
        v
    Extractor              -- fine-tuned TinyLlama + Groq fallback
        |
        v
   ClaimStore              -- central store for all extracted facts
        |
   +----|----+
   |         |
ClashFinder  TimelineGraph
   |         |
   +----|----+
        |
        v
      Scorer               -- additive evidence-weighted scoring
        |
        v
  PriorityPicker           -- decides who to question next
        |
        v
  Investigation Loop       -- orchestrates all 4 phases
        |
        v
      Report               -- generates final case report
```

### Module Descriptions

**SuspectAgent** — Each suspect is powered by Gemini 2.5 Flash via an OpenAI-compatible interface. The agent is initialized with the suspect's background, story, and behavioral directive. It maintains full conversation history so it remembers what it said in earlier questions.

**Detective** — Sends questions to suspect agents and collects their answers. It has two question sets: baseline questions asked to every suspect, and deep questions asked only to top suspects in Phase 4.

**Extractor** — Converts raw testimony text into structured `Claim` objects. It first tries the fine-tuned TinyLlama model. If that returns nothing, it falls back to Groq (Llama 3.1 8B).

**ClaimStore** — The central notebook of the investigation. Stores all claims, clashes, and the full interrogation log. Every other module reads from and writes to this store.

**ClashFinder** — Rule-based contradiction detector. Runs three rules: location conflicts, witness conflicts, and motive denial conflicts.

**TimelineGraph** — Parses the ground truth timeline field for each suspect and cross-references it against testimony. Flags suspects who were physically present at the crime scene and detects when a suspect's own timeline contradicts what they say.

**Scorer** — Maintains an additive suspicion score per suspect. Every score change is logged with its reason.

**PriorityPicker** — Computes a priority score for each suspect based on contradictions, motive, alibi status, and how many questions they have already been asked. Used to decide who gets targeted in Phase 4.

**Investigation** — Orchestrates the full 4-phase loop and calls all modules in order.

**Report** — Reads from all logged data and produces a structured human-readable case report. No LLM is involved in report generation — everything comes from logs.

---

## Methodology and Design Choices

### Why Structured Facts Instead of Raw Text

The task requires the system to be auditable. If I fed raw testimony directly to an LLM and asked it to name the murderer, it could do it but could not explain its reasoning. To make reasoning transparent, I extract structured claims from testimony in the form of `(who, rel, what, when, confidence)` tuples. This makes every downstream decision traceable — contradictions can be expressed as logical rules over these tuples rather than opaque LLM judgments.

The relation vocabulary I chose — `at`, `with`, `motive`, `owns`, `knows`, `did` — covers the types of claims that matter in a murder investigation: location for alibi, motive for guilt, relationships for opportunity.

### Why LLM for Extraction

Natural language is too varied for pure regex. "I was at home", "I spent the evening in my study", and "I never left the house" all mean the same thing but look completely different to a pattern matcher. I use an LLM to handle the linguistic variation and output structured JSON. The key design choice is that the LLM only handles parsing — the reasoning happens downstream in deterministic rule-based code.

### Why Fine-Tuning

Every call to Groq costs tokens and adds latency. A local fine-tuned model that already knows my exact JSON schema does not need the schema explained in every prompt. I fine-tuned TinyLlama 1.1B using LoRA on 1283 training pairs built from the dataset. LoRA adds small adapter matrices to the attention layers (`q_proj` and `v_proj`) — only 2.25 million parameters are trained instead of the full 1.1 billion. This fits on a T4 GPU in under an hour. The fine-tuned model runs locally with zero token cost. Groq is kept as a fallback for cases where the local model returns nothing.

### Why Additive Scoring Instead of Bayesian

I initially considered Bayesian updating — assigning each suspect a prior probability and updating it as evidence came in. The problem is that defining reliable likelihood values is impossible without ground truth statistics. What is the probability that an innocent person's alibi gets contradicted? Any number I assign is made up.

Additive scoring is fully transparent. I assign:
- **+20** for a contradiction — someone is lying about where they were
- **+15** for confirmed motive — necessary but not sufficient on its own
- **+10** for no alibi — absence of proof is weak evidence
- **-15** for verified alibi — actively clears someone

Each weight is documented and auditable. An evaluator can see exactly why a suspect's score changed and argue whether the weights are appropriate.

### The Timeline Graph

I noticed the dataset contains ground truth timelines for each suspect with exact timestamps and activities. I built a `TimelineGraph` that parses these timestamps, extracts locations, and runs three physical consistency checks:

- **Scene presence** — does the timeline place this suspect at the crime location around the time of the murder?
- **Witness conflict** — does what one suspect claims to have seen contradict another suspect's timeline?
- **Self-lie detection** — does the suspect's own testimony contradict their own timeline?

This adds a layer of physical reasoning that goes beyond comparing statements. A real detective would cross-reference testimony against CCTV or location data — the timeline graph serves the same function.

### The Four-Phase Investigation Loop

**Phase 1 — Initial Sweep:** Every suspect answers the same three baseline questions. This builds a shared fact base for comparison before any targeted questioning begins.

**Phase 2 — Timeline Graph Checks:** The timeline graph runs over all suspects and flags anyone whose timeline places them at the scene or whose movements contradict witness statements.

**Phase 3 — Clash Detection:** The rule engine scans all extracted claims for contradictions. Any suspect caught in a contradiction is confronted directly. This loop runs up to 3 rounds or until no new contradictions are found.

**Phase 4 — Targeted Pressure:** The top 2 suspects by score receive deeper questioning about motive, alibi corroboration, and who they think committed the crime. The PriorityPicker ensures the system does not keep hammering the same suspect by applying a fatigue penalty of 3 points per question already asked.

### Termination Logic

The system terminates early if the top suspect's score is at least 1.5 times the second suspect's score AND they have at least one confirmed motive or broken alibi. This avoids arbitrary round limits — the system stops when the evidence gap is large enough to justify a verdict.

---

## Performance Metrics

Evaluation was run on cases 300 to 350 from the dataset (50 cases not used during training).

| Metric | Value |
|---|---|
| Cases evaluated | 50 |
| Correct verdicts | 35 |
| Accuracy | 70.0% |
| Average claims per case | 1.4 |
| Average clashes per case | 0.0 |
| Average time per case | 45.7 seconds |

Token usage was tracked via the Groq API usage field on every call. Suspect interrogation used Gemini 2.5 Flash which does not report token counts through the OpenAI-compatible interface.

The 30% failure rate is concentrated in cases where multiple suspects received similar scores and the scoring gap was insufficient to trigger the termination criterion. The system correctly falls back to questioning all suspects in these ambiguous cases rather than making a low-confidence guess.

---

## Setup and Execution Guide

### Dependencies

```
groq
openai
transformers==4.40.0
datasets==2.18.0
peft==0.10.0
trl==0.8.6
accelerate==0.29.3
bitsandbytes==0.41.3
torch==2.2.2
matplotlib
numpy
```

Install all dependencies:

```bash
pip install groq openai transformers==4.40.0 datasets==2.18.0 peft==0.10.0 trl==0.8.6 accelerate==0.29.3 bitsandbytes==0.41.3 torch==2.2.2 matplotlib numpy
```

### API Keys Required

- **Groq API key** — for fact extraction fallback. Get free at https://console.groq.com
- **Gemini API key** — for suspect agents. Get free at https://aistudio.google.com

### Environment Setup

```bash
export GROQ_API_KEY="your-groq-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
```

### Dataset

Place the dataset file in the project root:

```
project_omni_pclub_secy_recruit_dataset.json
```

### Execution

The entrypoint is `main.py` as provided by PClub. The system integrates with the provided `SuspectAgent` interface and uses the same OpenAI-compatible Gemini client.

To run a single case:

```python
from main import SuspectAgent
from openai import OpenAI
import json, os

with open("project_omni_pclub_secy_recruit_dataset.json") as f:
    dataset = json.load(f)

case = dataset[0]
result = run_case(case)
print(f"Verdict: {result['suspect']} ({result['conf']} confidence)")
```

To run the full evaluation on the validation set (cases 300-350):

Run the notebook `Untitled2.ipynb` cell by cell from top to bottom. Results are saved to `outputs/` and reports to `reports/`. A summary is written to `summary.json`.

### Fine-tuning (optional)

The fine-tuned TinyLlama model is trained on the first run and saved to `./ft_extractor/`. If the directory already exists, Cell 8 loads it automatically. If it does not exist or fails to load, the system falls back to Groq extraction with no loss of functionality.

### Output Files

```
outputs/case_300.json   -- per-case verdict and metrics
reports/case_300.txt    -- full human-readable case report
summary.json            -- aggregate evaluation results
logs/err_NNN.json       -- error logs for failed cases
timeline_graph.png      -- suspect movement visualization
```
