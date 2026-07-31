import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Descriptive Answer Evaluator", page_icon="📝", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#f7f8fc;}
section[data-testid="stSidebar"]{background:#1a1f36;}
section[data-testid="stSidebar"] *{color:#e0e4f0 !important;}
.score-hero{background:linear-gradient(135deg,#1a1f36,#2d3561);border-radius:18px;padding:32px;text-align:center;color:white;margin-bottom:20px;}
.score-number{font-size:72px;font-weight:600;line-height:1;}
.metric-row{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-radius:10px;margin-bottom:6px;background:white;border:1px solid #eef0f7;}
.metric-name{font-weight:500;font-size:14px;color:#1a1f36;}
.metric-desc{font-size:12px;color:#888;margin-top:1px;}
.bar-bg{background:#f0f1f6;border-radius:6px;height:8px;width:120px;overflow:hidden;}
.rubric-covered{background:#f0fdf4;border-left:3px solid #22c55e;border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:6px;font-size:13px;}
.rubric-partial{background:#fefce8;border-left:3px solid #eab308;border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:6px;font-size:13px;}
.rubric-missed{background:#fff1f2;border-left:3px solid #f43f5e;border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:6px;font-size:13px;}
.feedback-box{border-radius:12px;padding:16px 20px;font-size:14px;font-weight:500;margin-top:12px;}
#MainMenu,footer,header{visibility:hidden;}
.stButton>button{background:#1a1f36;color:white;border:none;border-radius:10px;padding:10px 28px;font-weight:500;font-size:15px;width:100%;}
.stButton>button:hover{background:#2d3561;}
</style>
""", unsafe_allow_html=True)

PRESETS = {
    "Select a question...":{"question":"","rubric":[]},
    "Q1 — Deadlock":{"question":"Explain deadlock and its four necessary conditions.","rubric":["deadlock is a situation where processes wait indefinitely for resources held by each other","mutual exclusion means only one process can use a resource at a time","hold and wait means a process holds at least one resource while waiting for additional resources","no preemption means resources cannot be forcibly taken from a process","circular wait means a chain of processes each waiting for a resource held by the next"]},
    "Q2 — CPU Scheduling":{"question":"Explain CPU scheduling and compare FCFS and Round Robin algorithms.","rubric":["CPU scheduling determines which process runs next on the CPU from the ready queue","FCFS schedules processes in order of arrival and is non-preemptive","FCFS causes convoy effect where short processes wait behind long processes","Round Robin assigns a fixed time quantum to each process in cyclic order","Round Robin is preemptive and provides fair CPU allocation among all processes"]},
    "Q3 — Binary Search Trees":{"question":"Explain binary search trees and their operations.","rubric":["a binary search tree is a binary tree where left child is smaller and right child is greater than the parent node","search operation compares target with current node and traverses left or right accordingly","insertion places new node at correct position maintaining BST property","deletion handles three cases leaf node node with one child node with two children","BST operations have average time complexity of O log n but degrade to O n in worst case"]},
    "Q4 — Virtual Memory":{"question":"Explain virtual memory and the concept of paging.","rubric":["virtual memory allows processes to use more memory than physically available by using disk as extension of RAM","paging divides logical memory into fixed size blocks called pages and physical memory into frames of same size","page table maps virtual page numbers to physical frame numbers for address translation","page fault occurs when a required page is not in physical memory and must be loaded from disk","demand paging loads pages only when needed thus reducing memory usage and improving multiprogramming"]},
    "Q5 — Database Normalisation":{"question":"Explain the concept of normalisation in databases and its different forms.","rubric":["normalisation is the process of organising database to reduce redundancy and improve data integrity","first normal form requires atomic values in each column and no repeating groups","second normal form requires no partial dependency where non-key attributes depend on entire primary key","third normal form requires no transitive dependency where non-key attributes depend only on primary key","normalisation reduces update anomalies insertion anomalies and deletion anomalies in the database"]},
}

@st.cache_resource(show_spinner="Loading NLP models... first run ~3 minutes")
def load_models():
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    import spacy, torch
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nli   = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small", device=0 if torch.cuda.is_available() else -1)
    nlp   = spacy.load("en_core_web_sm")
    return sbert, nli, nlp

def relevance_score(sbert, question, student_answer):
    from sentence_transformers import util
    q_emb = sbert.encode(question, convert_to_tensor=True)
    a_emb = sbert.encode(student_answer, convert_to_tensor=True)
    return round(float(util.cos_sim(q_emb, a_emb).clamp(0,1)), 4)

def coverage_score(sbert, nlp, student_answer, rubric_points):
    from sentence_transformers import util
    doc   = nlp(student_answer)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    if not sents: sents = [student_answer]
    per_point = []
    for point in rubric_points:
        p_emb = sbert.encode(point, convert_to_tensor=True)
        best_sim = 0.0
        for sent in sents:
            s_emb = sbert.encode(sent, convert_to_tensor=True)
            sim = float(util.cos_sim(s_emb, p_emb))
            if sim > best_sim: best_sim = sim
        if   best_sim >= 0.65: partial = 1.0
        elif best_sim >= 0.50: partial = 0.75
        elif best_sim >= 0.40: partial = 0.50
        elif best_sim >= 0.30: partial = 0.25
        else:                  partial = 0.0
        per_point.append({'rubric_point':point,'similarity':round(best_sim,4),'partial':partial,'covered':best_sim>=0.40})
    score = sum(p['partial'] for p in per_point) / len(rubric_points)
    return {'score':round(score,4),'per_point':per_point,'covered_count':sum(1 for p in per_point if p['covered']),'total_points':len(rubric_points)}

def consistency_score(sbert, nli, nlp, student_answer, rubric_points):
    from sentence_transformers import util
    doc   = nlp(student_answer)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    if not sents: sents = [student_answer[:200]]
    scores = []
    for point in rubric_points:
        p_emb = sbert.encode(point, convert_to_tensor=True)
        best_sent = student_answer[:300]; best_sim = 0.0
        for sent in sents:
            s_emb = sbert.encode(sent, convert_to_tensor=True)
            sim = float(util.cos_sim(s_emb, p_emb))
            if sim > best_sim: best_sim = sim; best_sent = sent
        if best_sim < 0.25: scores.append(0.5); continue
        res   = nli(f'{point} [SEP] {best_sent[:300]}')[0]
        label = res['label'].lower(); conf = res['score']
        scores.append(1.0*conf if 'entail' in label else (0.5 if 'neutral' in label else 0.3))
    return round(float(np.mean(scores)), 4)

def reasoning_quality_score(nlp, student_answer):
    markers = ['because','therefore','thus','hence','since','as a result','consequently','this means','which causes','leading to','due to','this leads','resulting in','so that','this causes','which results','this results','in order to','which allows','this allows','this ensures','which ensures','this prevents','which prevents','this improves','enabling','which enables','this enables','which reduces','this reduces','which increases','which means','this guarantees','thereby','this avoids','accordingly','for this reason','as such','in turn','which in turn','and therefore','and thus']
    doc   = nlp(student_answer)
    sents = [s for s in doc.sents if len(s.text.strip()) > 3]
    words = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 2]
    if not sents or not words: return 0.0
    text_lower = student_answer.lower()
    found = sum(1 for m in markers if m in text_lower)
    ms = min(found/3.0, 1.0)
    ls = max(0.0, min((np.mean([len(s.text.split()) for s in sents])-5)/15.0, 1.0))
    vs = min(len(set(words))/max(len(words),1)*1.5, 1.0)
    ds = min(sum(1 for t in doc if t.dep_ in ('advcl','relcl','csubj','xcomp') and t.head.pos_ in ('VERB','AUX'))/max(len(sents),1), 1.0)
    return round(min(0.30*ms + 0.30*ls + 0.25*vs + 0.15*ds, 1.0), 4)

def coherence_score(sbert, nlp, student_answer):
    from sentence_transformers import util
    doc   = nlp(student_answer)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]
    if len(sents) < 2: return 0.5
    embs = sbert.encode(sents, convert_to_tensor=True)
    sims = [float(util.cos_sim(embs[i], embs[i+1])) for i in range(len(embs)-1)]
    return round(float(np.mean(sims)), 4)

def internal_consistency_score(nli, nlp, student_answer):
    doc   = nlp(student_answer)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 15]
    if len(sents) < 2: return 1.0
    sc = []
    for i in range(min(len(sents)-1, 4)):
        res = nli(f'{sents[i][:200]} [SEP] {sents[i+1][:200]}')[0]
        label = res['label'].lower()
        sc.append(1.0 if 'entail' in label else (0.8 if 'neutral' in label else 0.4))
    return round(float(np.mean(sc)), 4)

def evaluate_answer(sbert, nli, nlp, question, student_answer, rubric_points, max_marks=10.0, weights=None):
    if weights is None:
        weights = {'R':0.15,'V':0.25,'C':0.15,'Q':0.30,'H':0.10,'IC':0.05}
    R   = relevance_score(sbert, question, student_answer)
    cov = coverage_score(sbert, nlp, student_answer, rubric_points)
    V   = cov['score']
    C   = consistency_score(sbert, nli, nlp, student_answer, rubric_points)
    Q   = reasoning_quality_score(nlp, student_answer)
    H   = coherence_score(sbert, nlp, student_answer)
    IC  = internal_consistency_score(nli, nlp, student_answer)
    Veff = V * (0.5 + 0.5 * Q)
    raw  = (weights.get('R',0.15)*R + weights.get('V',0.25)*Veff + weights.get('C',0.15)*C +
            weights.get('Q',0.30)*Q + weights.get('H',0.10)*H   + weights.get('IC',0.05)*IC)
    final = round(max(0.0, min(raw,1.0)) * max_marks, 2)
    return {'R_relevance':R,'V_coverage':V,'V_effective':round(Veff,4),'C_consistency':C,'Q_reasoning':Q,'H_coherence':H,'IC_self_consistency':IC,'final_score':final,'max_marks':max_marks,'coverage_detail':cov['per_point'],'covered_count':cov['covered_count'],'total_points':cov['total_points']}

def bar_html(value, color):
    return f'<div class="bar-bg"><div style="width:{int(value*100)}%;background:{color};height:100%;border-radius:6px"></div></div>'

def score_color(pct):
    return "#22c55e" if pct>=0.75 else ("#f59e0b" if pct>=0.55 else ("#f97316" if pct>=0.35 else "#f43f5e"))

def get_feedback(pct):
    if pct>=0.80: return "🟢","Excellent","#f0fdf4","#16a34a","Comprehensive coverage with strong reasoning."
    if pct>=0.65: return "🟡","Good","#fefce8","#ca8a04","Covers main concepts, could improve depth."
    if pct>=0.45: return "🟠","Average","#fff7ed","#ea580c","Several key concepts missing or not well explained."
    if pct>=0.25: return "🔴","Below average","#fff1f2","#e11d48","Most key concepts not adequately addressed."
    return "⛔","Insufficient","#fff1f2","#be123c","Answer does not demonstrate understanding."

# SIDEBAR
with st.sidebar:
    st.markdown("## 📝 Answer Evaluator")
    st.markdown("*NLP Project — IIIT Bangalore*")
    st.markdown("---")
    page = st.radio("Navigate", ["🎯  Evaluate Answer","📊  Batch Test","ℹ️  About"], label_visibility="collapsed")
    st.markdown("---")
    max_marks = st.slider("Max marks", 5, 20, 10)
    st.markdown("---")
    st.markdown("**Models**\n• Sentence-BERT\n• DeBERTa NLI\n• spaCy")
    st.markdown("---")
    st.caption("Nitheesh Vemula · Varun Reddy\nIIIT Bangalore · April 2026")

# PAGE 1 — EVALUATE
if "Evaluate" in page:
    st.markdown("# Descriptive Answer Evaluation")
    st.markdown("Select a preset question or type your own. Paste a student answer and click **Evaluate**.")
    st.markdown("---")
    col_in, col_out = st.columns([1,1], gap="large")

    with col_in:
        st.markdown("### ✏️ Input")
        preset_key = st.selectbox("Load preset question", list(PRESETS.keys()))
        preset = PRESETS[preset_key]
        question    = st.text_area("Question", value=preset["question"], height=80, placeholder="Type the exam question here...")
        rubric_text = st.text_area("Rubric points (one per line)", value="\n".join(preset["rubric"]), height=160, placeholder="One expected concept per line...")
        answer      = st.text_area("Student answer", height=200, placeholder="Paste student answer here...")
        go_btn = st.button("Evaluate Answer →")

    with col_out:
        st.markdown("### 📊 Result")
        if go_btn:
            if not question.strip(): st.warning("Please enter a question."); st.stop()
            if not rubric_text.strip(): st.warning("Please enter rubric points."); st.stop()
            if not answer.strip(): st.warning("Please enter a student answer."); st.stop()
            rubric_points = [r.strip() for r in rubric_text.strip().split("\n") if r.strip()]
            with st.spinner("Evaluating..."):
                sbert, nli_m, nlp_m = load_models()
                result = evaluate_answer(sbert, nli_m, nlp_m, question, answer, rubric_points, max_marks)
            score = result['final_score']; pct = score/max_marks
            color = score_color(pct); icon,label,fb_bg,fb_clr,fb_txt = get_feedback(pct)
            st.markdown(f'<div class="score-hero"><div class="score-number" style="color:{color}">{score}</div><div style="font-size:18px;opacity:0.6;margin-top:4px">/ {max_marks}</div><div style="font-size:14px;opacity:0.7;margin-top:8px;letter-spacing:1px;text-transform:uppercase">{icon} {label}</div></div>', unsafe_allow_html=True)
            st.markdown("**Metric breakdown**")
            minfo = [("R","Relevance",result['R_relevance'],"#6366f1","Is the answer on-topic?"),("V","Coverage",result['V_coverage'],"#0ea5e9",f"{result['covered_count']}/{result['total_points']} rubric points"),("C","Consistency",result['C_consistency'],"#8b5cf6","Aligns with expected facts?"),("Q","Reasoning Quality",result['Q_reasoning'],"#f59e0b","Explains WHY and HOW?"),("H","Coherence",result['H_coherence'],"#14b8a6","Flows sentence to sentence?"),("IC","Self-Consistency",result['IC_self_consistency'],"#64748b","No self-contradictions?")]
            for key,name,val,clr,desc in minfo:
                st.markdown(f'<div class="metric-row"><div><div class="metric-name">{key} &nbsp;{name}</div><div class="metric-desc">{desc}</div></div><div style="display:flex;align-items:center;gap:10px">{bar_html(val,clr)}<span style="font-size:13px;font-weight:600;color:{clr};min-width:40px;text-align:right">{val}</span></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:#f8f9fc;border-radius:10px;padding:10px 14px;font-size:12px;color:#666;margin:8px 0">Effective Coverage (Veff) = {result["V_coverage"]} × (0.5 + 0.5 × {result["Q_reasoning"]}) = <b>{result["V_effective"]}</b></div>', unsafe_allow_html=True)
            st.markdown("**Rubric point coverage**")
            for pt in result['coverage_detail']:
                sim=pt['similarity']; text=pt['rubric_point']
                if pt['partial']>=1.0:   css="rubric-covered"; ic2="✅"; badge='<span style="float:right;font-size:10px;background:#dcfce7;color:#16a34a;padding:2px 8px;border-radius:10px;font-weight:600">full</span>'
                elif pt['covered']:      css="rubric-partial"; ic2="⚡"; badge='<span style="float:right;font-size:10px;background:#fef9c3;color:#854d0e;padding:2px 8px;border-radius:10px;font-weight:600">partial</span>'
                else:                    css="rubric-missed";  ic2="❌"; badge='<span style="float:right;font-size:10px;background:#ffe4e6;color:#be123c;padding:2px 8px;border-radius:10px;font-weight:600">missed</span>'
                st.markdown(f'<div class="{css}">{ic2} {text}{badge}<span style="float:right;font-size:11px;color:#999;margin-right:70px">sim={sim}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="feedback-box" style="background:{fb_bg};color:{fb_clr};border:1px solid {fb_clr}40">{icon} <b>{label}</b> — {fb_txt}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:white;border-radius:14px;padding:48px;text-align:center;color:#bbb;border:1px dashed #dde1ef"><div style="font-size:48px">📝</div><div style="margin-top:12px;font-size:15px;color:#888">Fill in the form on the left<br>and click <b style="color:#1a1f36">Evaluate Answer</b></div></div>', unsafe_allow_html=True)

# PAGE 2 — BATCH
elif "Batch" in page:
    st.markdown("# Batch Evaluation")
    st.markdown("Run all sample answers for a question at once and compare system vs human scores.")
    st.markdown("---")
    BATCH = {
        "Q1 — Deadlock":{"question":"Explain deadlock and its four necessary conditions.","max_marks":10,"rubric":["deadlock is a situation where processes wait indefinitely for resources held by each other","mutual exclusion means only one process can use a resource at a time","hold and wait means a process holds at least one resource while waiting for additional resources","no preemption means resources cannot be forcibly taken from a process","circular wait means a chain of processes each waiting for a resource held by the next"],"answers":[{"id":"A1","label":"Excellent","human":9.75,"text":"Deadlock is a situation in operating systems where two or more processes are unable to proceed because each is waiting for a resource held by another. Mutual exclusion means a resource can only be held by one process at a time. Hold and wait means a process is holding at least one resource and waiting to acquire additional resources. No preemption means resources cannot be forcibly removed. Circular wait means there exists a circular chain of processes where each process waits for a resource held by the next. All four conditions must hold simultaneously for deadlock to occur."},{"id":"A2","label":"Good","human":7.75,"text":"Deadlock occurs when processes are stuck waiting for each other. The four conditions are mutual exclusion where resources cannot be shared, hold and wait where processes keep resources while asking for more, no preemption which means you cannot take resources away, and circular wait which means processes form a cycle. Because these four conditions exist together deadlock cannot be resolved without external intervention."},{"id":"A3","label":"Average","human":5.25,"text":"Deadlock is when processes wait forever. Conditions are mutual exclusion, hold and wait, no preemption, circular wait. Mutual exclusion means one process uses resource. Hold and wait is when process holds resource. Circular wait is a cycle."},{"id":"A4","label":"Poor","human":3.25,"text":"Deadlock happens in OS. There are four conditions. Mutual exclusion is one condition. The other conditions are hold and wait, no preemption, and circular wait."},{"id":"A5","label":"Very poor","human":1.75,"text":"Deadlock is when process waits. Conditions are four. Mutual exclusion. Hold wait. Circular."}]},
        "Q2 — CPU Scheduling":{"question":"Explain CPU scheduling and compare FCFS and Round Robin algorithms.","max_marks":10,"rubric":["CPU scheduling determines which process runs next on the CPU from the ready queue","FCFS schedules processes in order of arrival and is non-preemptive","FCFS causes convoy effect where short processes wait behind long processes","Round Robin assigns a fixed time quantum to each process in cyclic order","Round Robin is preemptive and provides fair CPU allocation among all processes"],"answers":[{"id":"A1","label":"Excellent","human":9.75,"text":"CPU scheduling is the process by which the operating system decides which process in the ready queue gets to use the CPU next. FCFS schedules processes strictly in the order they arrive and is non-preemptive. The main disadvantage of FCFS is the convoy effect where short processes are forced to wait behind long processes thus increasing average waiting time. Round Robin is a preemptive algorithm that assigns a fixed time quantum to each process. After the quantum expires the process is moved to the back. This ensures fair CPU allocation because every process gets an equal share of CPU time."},{"id":"A2","label":"Good","human":7.75,"text":"CPU scheduling selects which process runs next. FCFS runs processes in arrival order and is non-preemptive. It has convoy effect problem where long processes block short ones. Round Robin gives each process a time quantum and preempts after quantum expires. Round Robin is fairer therefore preferred for time-sharing systems."},{"id":"A3","label":"Average","human":4.75,"text":"CPU scheduling is important in OS. FCFS is first come first served. Round Robin uses time quantum. FCFS is non preemptive. Round Robin is preemptive. FCFS has convoy effect."},{"id":"A4","label":"Poor","human":3.0,"text":"CPU scheduling is selection of process from ready queue. FCFS is first algorithm. Round Robin is second. FCFS has problems. Round Robin is better."},{"id":"A5","label":"Very poor","human":1.75,"text":"Scheduling is when OS picks process. FCFS and Round Robin both schedule. They are different algorithms."}]},
    }
    q_choice = st.selectbox("Select question", list(BATCH.keys()))
    q_data   = BATCH[q_choice]
    if st.button("Run batch evaluation"):
        sbert,nli_m,nlp_m = load_models()
        rows=[]; prog=st.progress(0); n=len(q_data["answers"])
        for i,ans in enumerate(q_data["answers"]):
            prog.progress((i+1)/n)
            r=evaluate_answer(sbert,nli_m,nlp_m,q_data["question"],ans["text"],q_data["rubric"],q_data["max_marks"])
            rows.append({"ID":ans["id"],"Quality":ans["label"],"Human":ans["human"],"System":r["final_score"],"Error":round(abs(r["final_score"]-ans["human"]),2),"R":r["R_relevance"],"V":r["V_coverage"],"C":r["C_consistency"],"Q":r["Q_reasoning"],"H":r["H_coherence"],"IC":r["IC_self_consistency"]})
        df=pd.DataFrame(rows)
        st.dataframe(df,use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Mean Absolute Error",f'{df["Error"].mean():.2f}')
        c2.metric("System avg",f'{df["System"].mean():.2f}')
        c3.metric("Human avg",f'{df["Human"].mean():.2f}')

# PAGE 3 — ABOUT
elif "About" in page:
    st.markdown("# About This System")
    st.markdown("---")
    st.markdown("### What it does\nAutomatically evaluates student descriptive answers against teacher-defined rubric points using 6 NLP metrics. No fixed model answer needed. Produces explainable per-dimension score breakdown.")
    st.markdown("### The 6 metrics")
    for row in [("R — Relevance","Sentence-BERT cosine similarity","Is the answer on-topic?"),("V — Coverage","SBERT sentence-level matching (4-tier)","Did student cover all expected rubric concepts?"),("C — Consistency","DeBERTa NLI entailment (sentence-level)","Does answer logically align with expected facts?"),("Q — Reasoning Quality","Discourse markers + spaCy deps + sentence length","Does student explain WHY and HOW?"),("H — Coherence","SBERT consecutive sentence similarity","Does answer flow logically sentence to sentence?"),("IC — Self-Consistency","DeBERTa NLI between answer sentence pairs","Does student contradict themselves?")]:
        st.markdown(f'<div style="background:white;border-radius:12px;padding:14px 18px;margin-bottom:8px;border:1px solid #eef0f7"><b>{row[0]}</b><br><span style="font-size:12px;color:#6366f1">{row[1]}</span><br><span style="font-size:13px;color:#555">{row[2]}</span></div>', unsafe_allow_html=True)
    st.markdown("### Scoring formula")
    st.latex(r"\text{Final} = \left(w_R R + w_V V_{\text{eff}} + w_C C + w_Q Q + w_H H + w_{IC} IC\right) \times M")
    st.latex(r"V_{\text{eff}} = V \times (0.5 + 0.5 \times Q)")
    st.markdown("### System results")
    st.dataframe(pd.DataFrame({"Method":["TF-IDF baseline","SBERT-only baseline","6-metric default","6-metric tuned (ours)"],"Pearson r":[0.758,0.534,0.658,0.847],"Spearman":[0.740,0.516,0.591,0.813],"QWK":[0.424,0.157,0.373,0.764]}),use_container_width=True,hide_index=True)
    st.markdown("**Team:** Nitheesh Vemula (MT2025079) · Varun Reddy")
    st.markdown("*NLP Project — IIIT Bangalore — Prof. Anutosh Maitra — April 2026*")
