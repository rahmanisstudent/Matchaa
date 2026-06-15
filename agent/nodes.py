"""
nodes.py — Semua Node Agent untuk Matcha (LangGraph)
Setiap fungsi adalah satu node yang menerima state dan mengembalikan state update.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from groq import Groq

from .state import MatchaState
from .prompts import (
    INTENT_CLASSIFIER_PROMPT,
    USER_PROFILER_PROMPT,
    SKILL_GAP_PROMPT,
    RESOURCE_RECOMMENDER_PROMPT,
    CV_REVIEWER_PROMPT,
    LINKEDIN_REVIEWER_PROMPT,
)
from .chroma_client import (
    get_required_skills_for_role,
    find_matching_courses,
    find_matching_jobs,
    get_keywords_for_skills,
    format_courses_for_prompt,
    format_jobs_for_prompt,
)
from agent import state
from .tavily_search import search_all_parallel, format_tavily_results_for_prompt

# Setup LLM — Groq

from dotenv import load_dotenv
load_dotenv(override=True)

_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "")) 
MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant"
]
print("DEBUG GROQ KEY:", os.environ.get("GROQ_API_KEY", "NOT FOUND")[:10])

def _call_llm(prompt: str) -> str:
    """Helper: panggil Groq dan kembalikan teks respons dengan auto-fallback."""
    last_err = None
    for model in MODEL_CANDIDATES:
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"ERROR: Model {model} failed: {e}. Trying next...")
            last_err = e
    raise last_err


def _call_llm_json(prompt: str) -> Dict[str, Any]:
    """Helper: panggil Groq dengan format JSON dan kembalikan dict parsed dengan auto-fallback."""
    last_err = None
    
    # Try with JSON mode first
    for model in MODEL_CANDIDATES:
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            print(f"ERROR JSON: Model {model} failed in JSON mode: {e}. Trying next...")
            last_err = e

    # Fallback to standard completion and manual parsing
    print("All models failed JSON mode. Trying standard mode + manual parsing...")
    for model in MODEL_CANDIDATES:
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()
            return _parse_json_response(content)
        except Exception as e:
            print(f"ERROR Standard: Model {model} failed in standard fallback mode: {e}. Trying next...")
            last_err = e

    return {}


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    Helper: parse JSON dari respons LLM.
    Coba ekstrak blok JSON jika ada markdown code fence.
    """
    # Bersihkan markdown code fence kalau ada
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Coba cari pola {...} pertama di teks
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


_INDOBERT_MODEL = None
_INDOBERT_TOKENIZER = None
_INDOBERT_LABELS = [
    "CAREER_EXPLORATION", "SKILL_INQUIRY", "RESOURCE_REQUEST", "CONSTRAINT_UPDATE",
    "PUSH_BACK", "CONFIRMATION", "CV_REVIEW", "LINKEDIN_REVIEW"
]

def _classify_intent_indobert(user_input: str):
    """Attempt IndoBERT inference. Download from Hugging Face if local files are missing."""
    global _INDOBERT_MODEL, _INDOBERT_TOKENIZER
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(current_dir, "indobert_intent_model")
        
        # Jika file model lokal lengkap, gunakan lokal. Jika tidak, unduh dari Hugging Face Hub.
        if os.path.exists(os.path.join(local_model_path, "model.safetensors")) or os.path.exists(os.path.join(local_model_path, "pytorch_model.bin")):
            model_path = local_model_path
        else:
            # Ganti dengan path repositori Hugging Face Anda
            model_path = os.environ.get("INDOBERT_HF_REPO", "username/matcha-indobert-intent")
            
        if _INDOBERT_MODEL is None or _INDOBERT_TOKENIZER is None:
            print(f"Loading IndoBERT intent classifier from {model_path}...")
            _INDOBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            _INDOBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
            _INDOBERT_MODEL.eval()
            
        inputs = _INDOBERT_TOKENIZER(
            user_input, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=64
        )
        
        with torch.no_grad():
            outputs = _INDOBERT_MODEL(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()
        intent = _INDOBERT_LABELS[pred_idx]
        
        print(f"IndoBERT Classified: {intent} (confidence: {confidence:.2f})")
        return intent, confidence, {}
    except Exception as e:
        print(f"IndoBERT classification failed, falling back to LLM. Error: {e}")
        return None


# ─────────────────────────────────────────────
# NODE 1: Intent Classifier
# ─────────────────────────────────────────────

def intent_classifier_node(state: MatchaState) -> MatchaState:
    """
    Mengklasifikasikan intent dari input user.
    Output: detected_intent, intent_confidence, extracted_info
    """
    user_input = state.get("user_input") or ""
    if not isinstance(user_input, str):
        user_input = str(user_input)

    # Bypass LLM intent classifier for programmatic unified analysis requests
    if (user_input.startswith("tolong analisis job description ini:") or 
        "tolong lakukan analisis skill gap, buatkan learning roadmap" in user_input.lower()):
        intent = "CAREER_EXPLORATION"
        confidence = 1.0
        extracted_info = {}
    else:
        # Try local IndoBERT first
        indobert_res = _classify_intent_indobert(user_input)
        if indobert_res is not None:
            intent, confidence, extracted_info = indobert_res
        else:
            # Fallback to LLM
            prompt = INTENT_CLASSIFIER_PROMPT.format(user_input=user_input)
            try:
                raw = _call_llm(prompt)
                parsed = _parse_json_response(raw)

                intent = parsed.get("intent", "CAREER_EXPLORATION").upper()
                confidence = float(parsed.get("confidence", 0.7))
                extracted_info = parsed.get("extracted_info", {})
            except Exception:
                intent = "CAREER_EXPLORATION"
                confidence = 0.5
                extracted_info = {}

    # ── Drift Detection ──
    history = state.get("previous_intent_history")
    if not isinstance(history, list):
        history = []
    drift_detected = False

    # Jika ada 3+ intent terakhir dan intent sekarang berbeda jauh dari mayoritas
    if len(history) >= 3:
        recent = history[-3:]
        most_common = max(set(recent), key=recent.count)
        if intent != most_common and intent not in ("CONFIRMATION", "PUSH_BACK"):
            drift_detected = True

    updated_history = history + [intent]

    return {
        **state,
        "detected_intent": intent,
        "intent_confidence": confidence,
        "extracted_info": extracted_info,
        "drift_detected": drift_detected,
        "previous_intent_history": updated_history,
    }


# ─────────────────────────────────────────────
# NODE 2: User Profiler
# ─────────────────────────────────────────────

def user_profiler_node(state: MatchaState) -> MatchaState:
    """
    Membangun dan mengupdate profil karir user dari percakapan.
    Output: user_profile (di-merge dengan profil sebelumnya)
    """
    existing_profile = dict(state.get("user_profile") or {})

    # ── Fast-path: skip LLM jika profil sudah lengkap dari onboarding ──
    # Ini hemat ~30k token per request untuk user yang sudah setup
    if existing_profile.get("target_role"):
        print("DEBUG user_profiler: profil sudah ada, skip LLM call")
        if "current_skills" not in existing_profile:
            existing_profile["current_skills"] = []
        return {
            **state,
            "user_profile": existing_profile,
        }

    user_input = state.get("user_input") or ""
    detected_intent = state.get("detected_intent") or ""
    messages = state.get("messages")
    if not isinstance(messages, list):
        messages = []

    # Format chat history untuk prompt
    chat_history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages[-10:]  # max 10 pesan terakhir
    )

    prompt = USER_PROFILER_PROMPT.format(
        chat_history=chat_history_text,
        user_input=user_input,
        detected_intent=detected_intent,
    )
    try:
        raw = _call_llm(prompt)
        new_profile = _parse_json_response(raw)
    except Exception as e:
        print("DEBUG ERROR user_profiler:", type(e).__name__, str(e))
        new_profile = {}

    # Merge dengan profil lama: hanya timpa field yang ada nilainya (bukan null)
    for key, value in new_profile.items():
        if value is not None and value != [] and value != "":
            existing_profile[key] = value

    # Pastikan current_skills selalu list
    if "current_skills" not in existing_profile:
        existing_profile["current_skills"] = []

    print("DEBUG profile:", existing_profile)
    return {
        **state,
        "user_profile": existing_profile,
    }

    


# ─────────────────────────────────────────────
# NODE 3: Skill Gap Analyzer
# ─────────────────────────────────────────────

def skill_gap_analyzer_node(state: MatchaState) -> MatchaState:
    """
    Menganalisis skill gap dan menghasilkan learning path.
    Data dikumpulkan paralel dari Chroma (lokal) + Tavily (web).
    Output: skill_gaps (teks), agent_response, learning_roadmap, ats_analysis
    """
    print("DEBUG skill_gap input profile:", state.get("user_profile"))
    user_profile    = state.get("user_profile") or {}
    detected_intent = state.get("detected_intent", "")

    if not user_profile.get("target_role"):
        return {
            **state,
            "agent_response": (
                "Untuk membuat learning path yang tepat, aku perlu tahu dulu "
                "target karir kamu. Kamu ingin berkarir sebagai apa? 🎯"
            ),
        }

    target_role    = user_profile.get("target_role", "")
    current_skills = user_profile.get("current_skills", [])

    # ── Check if CV has been uploaded ──
    if not state.get("cv_uploaded") and not state.get("cv_text"):
        user_name = user_profile.get("user_name") or "Pengguna"
        return {
            **state,
            "agent_response": (
                f"Halo {user_name}! Saya telah mencatat latar belakang dan target karir Anda sebagai {target_role}. "
                "Untuk memulai analisis ATS Match Rate, mendeteksi kesenjangan keahlian (skill gap), dan membuat "
                "peta jalan belajar (learning roadmap) yang terpersonalisasi, silakan unggah berkas CV Anda terlebih dahulu pada area di atas. 😊"
            )
        }

    # ── Paralel: Chroma + Tavily ──
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_chroma = executor.submit(
            lambda: {
                "required_skills": get_required_skills_for_role(target_role),
                "matching_jobs":   find_matching_jobs(target_role, current_skills, top_k=3),
                "matching_courses": find_matching_courses(target_role, current_skills, top_k=15),
            }
        )
        future_tavily = executor.submit(
            search_all_parallel,
            target_role,
            current_skills,
            False,  # career insights off supaya hemat quota
        )
        chroma_data  = future_chroma.result()
        tavily_data  = future_tavily.result()

    tavily_context = format_tavily_results_for_prompt(tavily_data)

    enriched_profile = {
        **user_profile,
        "market_required_skills":  chroma_data["required_skills"][:15],
        "sample_job_requirements": format_jobs_for_prompt(chroma_data["matching_jobs"]),
        "web_context":             tavily_context,
    }

    # Format courses catalog for prompt
    courses_catalog_list = []
    for c in chroma_data.get("matching_courses", []):
        meta = c.get("metadata", {})
        courses_catalog_list.append({
            "title": meta.get("title") or meta.get("course_name") or "",
            "platform": meta.get("platform") or "",
            "url": meta.get("url") or "",
            "skills_covered": meta.get("skills_covered") or []
        })

    # Potong cv/linkedin text agar tidak terlalu banyak token
    # Potong cv/linkedin/job_desc text agar menghemat banyak token secara signifikan
    cv_text_raw = state.get("cv_text") or ""
    linkedin_text_raw = state.get("linkedin_text") or ""
    job_desc_raw = state.get("job_description") or ""

    cv_text_for_prompt = cv_text_raw[:1800] if cv_text_raw else "Tidak ada CV diupload."
    linkedin_text_for_prompt = linkedin_text_raw[:1000] if linkedin_text_raw else "Tidak ada profil LinkedIn dihubungkan."
    job_desc_for_prompt = job_desc_raw[:1500] if job_desc_raw else "Tidak ada job description spesifik diinput."

    # Buat profil ringkas untuk prompt (tanpa web_context yang besar)
    compact_profile = {
        "user_name": user_profile.get("user_name"),
        "target_role": target_role,
        "current_role": user_profile.get("current_role"),
        "current_skills": current_skills,
        "hours_per_week": user_profile.get("hours_per_week"),
        "market_required_skills": chroma_data["required_skills"][:10],
    }

    prompt = SKILL_GAP_PROMPT.format(
        target_role=target_role,
        user_profile=json.dumps(compact_profile, ensure_ascii=False),
        cv_text=cv_text_for_prompt,
        linkedin_text=linkedin_text_for_prompt,
        job_description=job_desc_for_prompt,
        courses_catalog=json.dumps(courses_catalog_list[:6], ensure_ascii=False)
    )

    mastered_skills = current_skills
    skill_gaps_list = []

    try:
        parsed = _call_llm_json(prompt)
        chat_response = parsed.get("chat_response", "Berikut analisis kesenjangan keahlian dan rancangan peta jalan belajarmu.")
        match_rate = parsed.get("match_rate", 29)
        mastered_skills = parsed.get("mastered_skills", current_skills)
        skill_gaps_list = parsed.get("skill_gaps", [])
        ats_analysis = parsed.get("ats_analysis", {
            "cv_pros": [],
            "cv_cons": [],
            "suggested_keywords": []
        })
        learning_roadmap = parsed.get("learning_roadmap", {
            "total_weeks": 12,
            "start_date": "Juni 2026",
            "end_date": "September 2026",
            "phases": []
        })

        # Update user profile with extracted skills if not empty
        updated_profile = {**user_profile}
        if mastered_skills:
            updated_profile["current_skills"] = mastered_skills
            
    except Exception as e:
        print("ERROR parsing skill gap JSON:", e)
        chat_response = f"Maaf, terjadi kendala saat menganalisis skill gap. Coba lagi ya! ({e})"
        match_rate = 29
        skill_gaps_list = []
        ats_analysis = {"cv_pros": [], "cv_cons": [], "suggested_keywords": []}
        learning_roadmap = {"total_weeks": 12, "start_date": "Juni 2026", "end_date": "September 2026", "phases": []}
        updated_profile = user_profile

    # Calculate ATS match rate using custom model and overwrite the value before updating the state
    cv_or_linkedin = cv_text_raw if cv_text_raw else linkedin_text_raw
    custom_match_rate = None
    
    # Try calling the FastAPI model serving microservice first
    try:
        import os
        import requests
        api_url = os.environ.get("ATS_MODEL_API_URL", "http://localhost:8080").rstrip("/")
        payload = {
            "cv_text": cv_or_linkedin,
            "job_description": job_desc_raw,
            "target_role": target_role
        }
        api_res = requests.post(f"{api_url}/predict", json=payload, timeout=2)
        if api_res.status_code == 200:
            custom_match_rate = api_res.json().get("match_rate")
            print(f"DEBUG Custom ATS Model Match Rate (API Server): {custom_match_rate}%")
    except Exception as api_err:
        print(f"FastAPI model server offline (http://localhost:8080), falling back to local file: {api_err}")
        
    # Fallback to local import if API server is offline
    if custom_match_rate is None:
        try:
            from ats_model_app.model import predict_ats_match_rate
            custom_match_rate = predict_ats_match_rate(
                cv_text=cv_or_linkedin,
                job_description=job_desc_raw,
                target_role=target_role
            )
            print(f"DEBUG Custom ATS Model Match Rate (Local Fallback): {custom_match_rate}%")
        except Exception as model_err:
            print("ERROR calculating custom ATS model match rate prediction locally:", model_err)

    if custom_match_rate is not None:
        match_rate = custom_match_rate

    return {
        **state,
        "user_profile": updated_profile,
        "skill_gaps": chat_response,
        "agent_response": chat_response,
        "learning_roadmap": learning_roadmap,
        "ats_analysis": {
            "match_rate": match_rate,
            "mastered_skills": mastered_skills,
            "skill_gaps": skill_gaps_list,
            **ats_analysis
        }
    }


# ─────────────────────────────────────────────
# NODE 4: CV Reviewer
# ─────────────────────────────────────────────

def cv_reviewer_node(state: MatchaState) -> MatchaState:
    """
    Mereview CV user dan memberikan feedback vs target karir.
    Output: agent_response, ats_analysis
    """
    cv_text = state.get("cv_text", "")
    user_profile = state.get("user_profile") or {}
    skill_gaps = state.get("skill_gaps", "Belum dianalisis")

    if not cv_text:
        return {
            **state,
            "agent_response": (
                "Kamu belum upload CV-nya nih! Upload dulu di panel kiri atas ya, "
                "baru aku bisa kasih feedback yang spesifik. 📄"
            ),
        }

    target_role = user_profile.get("target_role", "belum ditentukan")

    prompt = CV_REVIEWER_PROMPT.format(
        cv_text=cv_text[:2500],
        target_role=target_role,
        skill_gaps=skill_gaps,
    )

    try:
        parsed = _call_llm_json(prompt)
        chat_response = parsed.get("chat_response", "Berikut hasil review CV Anda.")
        ats_analysis = parsed.get("ats_analysis", {
            "cv_pros": [],
            "cv_cons": [],
            "suggested_keywords": []
        })
    except Exception as e:
        chat_response = f"Maaf, terjadi kendala saat review CV. Coba lagi ya! ({e})"
        ats_analysis = {"cv_pros": [], "cv_cons": [], "suggested_keywords": []}

    # Merge or initialize ats_analysis
    existing_ats = dict(state.get("ats_analysis") or {})
    existing_ats.update(ats_analysis)

    return {
        **state,
        "agent_response": chat_response,
        "ats_analysis": existing_ats
    }


# ─────────────────────────────────────────────
# NODE 5: LinkedIn Reviewer
# ─────────────────────────────────────────────

def linkedin_reviewer_node(state: MatchaState) -> MatchaState:
    """
    Mereview profil LinkedIn user dan memberikan saran optimasi.
    Output: agent_response, ats_analysis
    """
    linkedin_text = state.get("linkedin_text", "")
    user_profile = state.get("user_profile") or {}
    skill_gaps = state.get("skill_gaps", "Belum dianalisis")

    if not linkedin_text:
        return {
            **state,
            "agent_response": (
                "Kamu belum upload profil LinkedIn-nya nih! Export PDF dari LinkedIn "
                "(Me → Save to PDF) lalu upload di panel kiri atas ya. 💼"
            ),
        }

    target_role = user_profile.get("target_role", "belum ditentukan")

    prompt = LINKEDIN_REVIEWER_PROMPT.format(
        linkedin_text=linkedin_text[:1500],
        target_role=target_role,
        skill_gaps=skill_gaps,
    )

    try:
        parsed = _call_llm_json(prompt)
        chat_response = parsed.get("chat_response", "Berikut hasil review profil LinkedIn Anda.")
        ats_analysis = parsed.get("ats_analysis", {
            "cv_pros": [],
            "cv_cons": [],
            "suggested_keywords": []
        })
    except Exception as e:
        chat_response = f"Maaf, terjadi kendala saat review LinkedIn. Coba lagi ya! ({e})"
        ats_analysis = {"cv_pros": [], "cv_cons": [], "suggested_keywords": []}

    # Merge or initialize ats_analysis
    existing_ats = dict(state.get("ats_analysis") or {})
    existing_ats.update(ats_analysis)

    return {
        **state,
        "agent_response": chat_response,
        "ats_analysis": existing_ats
    }


# ─────────────────────────────────────────────
# NODE 6: General Responder
# Untuk intent yang tidak butuh analisis berat
# (PUSH_BACK, CONFIRMATION, CONSTRAINT_UPDATE)
# ─────────────────────────────────────────────

def general_responder_node(state: MatchaState) -> MatchaState:
    """
    Menangani intent umum: push back, konfirmasi, update constraint.
    Output: agent_response
    """
    user_input = state.get("user_input", "")
    detected_intent = state.get("detected_intent", "")
    user_profile = state.get("user_profile") or {}
    drift_detected = state.get("drift_detected", False)
    messages = state.get("messages", [])

    chat_history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages[-6:]
    )

    drift_note = ""
    if drift_detected:
        drift_note = (
            "\n\nCatatan: Aku mendeteksi kamu mungkin mengubah arah tujuan karir. "
            "Apakah kamu ingin aku memperbarui rencana belajarmu sesuai tujuan terbaru?"
        )

    prompt = f"""Kamu adalah Matcha, asisten karir adaptif yang ramah dan suportif dalam Bahasa Indonesia.

Riwayat percakapan:
{chat_history_text}

Intent user saat ini: {detected_intent}
Input terbaru user: {user_input}
Profil user: {json.dumps(user_profile, ensure_ascii=False)}

Berikan respons yang natural, empatis, dan membantu sesuai konteks.
- Jika PUSH_BACK: akui keberatan user, tawarkan alternatif atau klarifikasi.
- Jika CONFIRMATION: konfirmasi pemahaman dan tanyakan langkah selanjutnya.
- Jika CONSTRAINT_UPDATE: akui update constraint dan tanyakan apakah perlu revisi learning path.
{drift_note}

Respons dalam Bahasa Indonesia yang hangat dan tidak lebih dari 3 paragraf."""

    try:
        response_text = _call_llm(prompt)
    except Exception as e:
        response_text = (
            "Terima kasih sudah berbagi! Ada yang bisa aku bantu lebih lanjut? 😊"
        )

    return {
        **state,
        "agent_response": response_text,
    }