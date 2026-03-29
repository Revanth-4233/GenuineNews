import os
import io
import json
import base64
import threading
import traceback
import requests

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import UserProfile, PredictionHistory
from .ml_engine import get_detector, scrape_url, LABEL_COLORS, LABEL_ICONS
from .dataset_loader import load_combined_dataset, get_dataset_stats


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE SYSTEM — Sir's Requirement
# SerpAPI (Google) + NewsAPI + GDELT simultaneously
# ══════════════════════════════════════════════════════════════════════════════

SERP_API_KEY = "76f36bd7858edb9ae36f2415e93960ce09e1b627ba1d80de7c08b77a5e61a4ed"
NEWS_API_KEY = "bd21a50ada494060a230ce6340ff5cde"

TRUSTED_SOURCES = [
    "ndtv.com", "thehindu.com", "timesofindia.com",
    "hindustantimes.com", "indianexpress.com",
    "livemint.com", "economictimes.com",
    "business-standard.com", "indiatoday.in",
    "theprint.in", "scroll.in", "thewire.in",
    "firstpost.com", "news18.com", "outlookindia.com",
    "sakshi.com", "andhrajyothi.com", "eenadu.net",
    "telanganatoday.com", "thenewsminute.com",
    "ntv.in", "tv9telugu.com",
    "bbc.com", "reuters.com", "apnews.com",
    "theguardian.com", "aljazeera.com",
    "wikipedia.org", "ac.in", "edu.in",
    "risekrishnasaigandhi.edu.in", "risekrishnasaiprakasam.edu.in",
    "qiscet.edu.in", "qis.edu.in", "qiscollege.com", "qiscet.ac.in",
]


def _build_query(news_text):
    """Build clean search query — removes stopwords."""
    stop = {
        'the','a','an','in','on','at','is','are','was','were',
        'of','to','for','and','or','but','with','from','by',
        'it','its','as','be','has','had','have','will','that',
        'this','these','those','their','they','we','you','i',
        'current','currently','now','today','recently',
        'mr','mrs','ms','dr','sir','shri','smt','prof',
        'he','she','his','her','our','my','your',
        'said','says','told','according','reported',
    }
    import re
    text_no_punct = re.sub(r'[^\w\s]', ' ', news_text)
    words = text_no_punct.split()
    filtered = [w for w in words if w.lower() not in stop and len(w) > 1]
    query = ' '.join(filtered[:8])
    print(f"[Query] '{query}'")
    return query


def search_serpapi(query):
    """Search Google via SerpAPI — free 100/month."""
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "q":       query,
                "api_key": SERP_API_KEY,
                "engine":  "google",
                "num":     5,
                "gl":      "in",
                "hl":      "en",
            }, timeout=8
        )
        data = resp.json()
        if "error" in data:
            print(f"[SerpAPI] Error: {data['error']}")
            return []
        results = []
        for item in data.get("organic_results", [])[:4]:
            url = item.get("link", "")
            results.append({
                "title":      item.get("title","")[:120],
                "snippet":    item.get("snippet","")[:200],
                "url":        url,
                "source":     item.get("displayed_link",""),
                "date":       item.get("date","Recent"),
                "is_trusted": any(d in url for d in TRUSTED_SOURCES),
                "api":        "Google",
            })
        print(f"[SerpAPI] Found {len(results)} results")
        return results
    except Exception as e:
        print(f"[SerpAPI] Exception: {str(e)[:80]}")
        return []


def search_newsapi(query):
    """Search NewsAPI for today's news."""
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        query,
                "apiKey":   NEWS_API_KEY,
                "sortBy":   "publishedAt",
                "pageSize": 5,
                "language": "en",
            }, timeout=5
        )
        data = resp.json()
        if data.get("status") != "ok":
            print(f"[NewsAPI] Error: {data.get('message')}")
            return []
        results = []
        for a in data.get("articles", [])[:3]:
            url   = a.get("url","")
            title = a.get("title","") or ""
            if not title or title == "[Removed]" or not url:
                continue
            results.append({
                "title":      title[:120],
                "snippet":    (a.get("description","") or "")[:200],
                "url":        url,
                "source":     a.get("source",{}).get("name","Unknown"),
                "date":       (a.get("publishedAt","") or "")[:10],
                "is_trusted": any(d in url for d in TRUSTED_SOURCES),
                "api":        "NewsAPI",
            })
        print(f"[NewsAPI] Found {len(results)} results")
        return results
    except Exception as e:
        print(f"[NewsAPI] Exception: {str(e)[:80]}")
        return []


def search_gdelt(query):
    """Search GDELT — no API key needed."""
    try:
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query":      query,
                "mode":       "artlist",
                "maxrecords": "5",
                "format":     "json",
                "sourcelang": "english",
            }, timeout=8
        )
        results = []
        for a in resp.json().get("articles", [])[:3]:
            url   = a.get("url","")
            title = a.get("title","") or ""
            if not title or not url:
                continue
            results.append({
                "title":      title[:120],
                "snippet":    "",
                "url":        url,
                "source":     a.get("domain",""),
                "date":       (a.get("seendate","") or "")[:8],
                "is_trusted": any(d in url for d in TRUSTED_SOURCES),
                "api":        "GDELT",
            })
        print(f"[GDELT] Found {len(results)} results")
        return results
    except Exception as e:
        print(f"[GDELT] Exception: {str(e)[:80]}")
        return []


# ── Official college/institution domains ─────────────────────────────────────
OFFICIAL_DOMAINS = [
    ".edu.in", ".ac.in", ".edu",
    "rise", "paavai", "jntu", "vit", "srm", "bits",
    "iit", "nit", "iiit", "anna", "osmania", "andhra",
]

# ── College official info keywords ────────────────────────────────────────────
OFFICIAL_INFO_KEYWORDS = [
    "chairman", "principal", "director", "vice chancellor",
    "founder", "president", "dean", "registrar",
    "head of department", "hod", "faculty", "professor",
    "management", "governing body", "governing board",
]

# ── College event keywords ─────────────────────────────────────────────────────
COLLEGE_EVENT_KEYWORDS = [
    "fest", "festival", "event", "seminar", "workshop",
    "hackathon", "competition", "sports", "cultural",
    "tech fest", "annual day", "convocation", "inauguration",
    "conducted", "organized", "held", "celebrated",
]


def _is_official_site(url):
    """Check if URL is an official college/institution website."""
    return any(d in url.lower() for d in OFFICIAL_DOMAINS)


def _detect_news_type(news_text):
    """
    Detect what type of claim this is:
    - official_info : chairman/principal/staff → check ONLY official sites
    - college_event : fest/seminar/sports      → check ANYWHERE
    - general_news  : accident/politics/other  → check ANYWHERE
    """
    text_lower = news_text.lower()
    if any(k in text_lower for k in OFFICIAL_INFO_KEYWORDS):
        return "official_info"
    elif any(k in text_lower for k in COLLEGE_EVENT_KEYWORDS):
        return "college_event"
    else:
        return "general_news"


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — Completely rewritten _extract_proper_names()
# Handles initials like "Mr. I. C. Rangamannar" correctly.
# Only returns real name words — single-letter initials are IGNORED.
# ══════════════════════════════════════════════════════════════════════════════
def _extract_proper_names(news_text):
    """
    Extract meaningful name parts from claim for official-site matching.
    Scans for the words before role keywords (chairman, principal, etc).
    """
    import re as _re

    text_lower = news_text.lower()
    name_str = ""

    # Structure 2: Role ... is/was/by ... Name
    role_first = _re.search(
        r'(?:chairman|principal|director|vice chancellor|founder|president|dean|registrar|head|hod|faculty|professor).*?(?:is|was|by|named)\s+([a-z\.\s]+)', 
        text_lower
    )
    if role_first:
        name_str = role_first.group(1)
    else:
        # Structure 1: Name ... is/was ... Role
        name_first = _re.search(
            r'^([a-z\.\s]+?)\s+(?:is|was|will|has|appointed|elected|made|the|a|an)*\s*(?:chairman|principal|director|vice chancellor|founder|president|dean|registrar|head|hod|faculty|professor)', 
            text_lower
        )
        if name_first:
            name_str = name_first.group(1)
            
    if not name_str:
        # 3. Fallback: look for titles anywhere
        match_title = _re.search(
            r'(?:dr|mr|mrs|ms|prof|sri|shri|hon|smt)\.?\s*([a-z\.]+?(?:\s+[a-z\.]+){0,3})', 
            text_lower
        )
        if match_title:
            name_str = match_title.group(1)
            
    if not name_str:
        print("[Names] No person name found in input")
        return []

    print(f"[Names] Raw extracted: {name_str!r}")

    # Remove titles
    clean_str = _re.sub(r'\b(dr|mr|mrs|ms|prof|sri|shri|hon|smt)\b\.?', ' ', name_str)

    words = clean_str.split()
    parts = []
    for w in words:
        clean_w = w.replace('.', '').replace(',', '').strip()
        # skip single initials and stopwords
        if len(clean_w) > 2 and clean_w not in ['the', 'that', 'this', 'and', 'college', 'university', 'institute', 'in', 'of', 'for', 'on', 'at']:
            parts.append(clean_w)

    print(f"[Names] Parts to match on official site: {parts}")
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — Smarter _compare_claim_vs_evidence()
# Uses longest/surname match as primary signal — not strict ALL-parts rule.
# ══════════════════════════════════════════════════════════════════════════════
def _compare_claim_vs_evidence(news_text, evidence_list):
    """
    Smart 3-mode comparison:

    Mode 1 — Official Info (chairman/principal/staff):
      → Check ONLY official college websites (.edu.in, .ac.in)
      → Surname / longest name part found on official site = CONFIRMED = TRUE
      → No name parts found at all = CONTRADICTED = FAKE

    Mode 2 — College Events (fest/seminar/sports):
      → Check ALL sources (Google + NewsAPI + GDELT)
      → Found anywhere = CONFIRMED = TRUE

    Mode 3 — General News (accident/politics/other):
      → Check ALL sources
      → Trusted source found = CONFIRMED
    """
    if not evidence_list:
        return "UNVERIFIED", "No evidence found anywhere", ""

    news_type = _detect_news_type(news_text)
    print(f"[Compare] News type detected: {news_type}")

    # ── Mode 1: Official Info — check only official sites ──────────────────
    if news_type == "official_info":
        official_sources = [
            e for e in evidence_list if _is_official_site(e.get("url",""))
        ]
        print(f"[Compare] Official sites found: {len(official_sources)}")

        if not official_sources:
            return (
                "UNVERIFIED",
                "No official college website found to verify this claim",
                ""
            )

        # Build evidence text — fetch actual page for full content
        official_parts = []
        for e in official_sources:
            if not _is_official_site(e.get("url","")):
                continue
            official_parts.append(
                e.get("title","") + " " +
                e.get("snippet","") + " " +
                e.get("url","").replace("-"," ").replace("/"," ")
            )
            # Fetch actual webpage for full name verification
            try:
                import re as _re2
                page_resp = requests.get(
                    e.get("url",""), timeout=5,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if page_resp.status_code == 200:
                    page_clean = _re2.sub(r"<[^>]+>", " ", page_resp.text)
                    page_clean = _re2.sub(r"\s+", " ", page_clean)
                    official_parts.append(page_clean[:5000])
                    print(f"[Fetch] Got page: {e.get('url','')[:50]}")
            except Exception as fe:
                print(f"[Fetch] Failed: {str(fe)[:40]}")

        official_text = " ".join(official_parts).lower()

        proper_names = _extract_proper_names(news_text)
        print(f"[Compare] Names to verify: {proper_names}")
        print(f"[Compare] Official text sample: {official_text[:200]}")

        if not proper_names:
            # No person name extracted — cannot verify the specific person
            return (
                "UNVERIFIED",
                "Found official site but could not extract a clear name from your claim to verify. Please try rephrasing.",
                official_sources[0].get("url","")[:60]
            )

        matched   = [n for n in proper_names if n in official_text]
        unmatched = [n for n in proper_names if n not in official_text]
        print(f"[Compare] Matched={matched}  Unmatched={unmatched}")

        # ── FIX 2: Smart surname/longest-part rule ──────────────────────────
        # Find the longest name part — typically the surname, most unique
        longest_part   = max(proper_names, key=len) if proper_names else ""
        longest_matched = longest_part in official_text

        if not matched:
            # Zero name parts found anywhere on official site → FAKE
            return (
                "CONTRADICTED",
                "Name NOT found on official college website — claim is FALSE",
                f"Not found: {', '.join(unmatched)}"
            )
        elif longest_matched:
            # The most significant name part (surname/longest) is confirmed
            # → REAL even if minor parts like first name don't appear
            return (
                "CONFIRMED",
                "Name verified on official college website",
                f"Confirmed: {', '.join(matched)}"
            )
        elif len(matched) >= max(1, len(proper_names) // 2 + 1):
            # Majority of name parts matched → CONFIRMED
            return (
                "CONFIRMED",
                "Name substantially verified on official college website",
                f"Confirmed: {', '.join(matched)}"
            )
        else:
            # Minority matched and surname missing → CONTRADICTED
            return (
                "CONTRADICTED",
                "Full name NOT verified on official website — claim is FALSE",
                f"Not found: {', '.join(unmatched)}"
            )

    # ── Mode 2 & 3: Events and General News — check anywhere ───────────────
    else:
        all_text = " ".join([
            e.get("title","") + " " + e.get("snippet","")
            for e in evidence_list
        ]).lower()

        trusted = [e for e in evidence_list if e.get("is_trusted")]
        total   = len(evidence_list)

        proper_names = _extract_proper_names(news_text)
        print(f"[Compare] Names to verify: {proper_names}")

        if proper_names:
            matched   = [n for n in proper_names if n in all_text]
            unmatched = [n for n in proper_names if n not in all_text]
            print(f"[Compare] Matched={matched}  Unmatched={unmatched}")

            longest_part    = max(proper_names, key=len) if proper_names else ""
            longest_matched = longest_part in all_text

            if matched and longest_matched:
                return (
                    "CONFIRMED",
                    f"News verified — found in {total} source(s)",
                    f"Confirmed: {', '.join(matched)}"
                )
            elif matched and not unmatched:
                return (
                    "CONFIRMED",
                    f"News verified — found in {total} source(s)",
                    f"Confirmed: {', '.join(matched)}"
                )
            elif unmatched and not matched:
                return (
                    "CONTRADICTED",
                    "Claim details not found in any source",
                    f"Not found: {', '.join(unmatched)}"
                )
            else:
                return (
                    "PARTIALLY-CONFIRMED",
                    "Some details found in sources",
                    f"Matched: {', '.join(matched)}"
                )
        else:
            if len(trusted) >= 1:
                return (
                    "UNVERIFIED",
                    f"Found related articles in {len(trusted)} trusted source(s), but could not verify exact claim details",
                    trusted[0].get("source","")
                )
            elif total >= 2:
                return (
                    "UNVERIFIED",
                    f"Found in {total} source(s) but none are trusted outlets",
                    ""
                )
            else:
                return "UNVERIFIED", "Could not verify from available sources", ""


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — _get_news_tag() now accepts confidence parameter.
# When evidence is UNVERIFIED and AI is ≥80% confident → trust the AI.
# ══════════════════════════════════════════════════════════════════════════════
def _get_news_tag(result_label, final_verdict, trusted_count, confidence=0):
    """
    Strict Verification: ONLY exact match or trusted sources return REAL NEWS.
    """
    if final_verdict in ["CONFIRMED", "REAL", "LIKELY-REAL"]:
        return "REAL NEWS", "#27ae60", "🟢"
    else:
        return "FAKE NEWS", "#e74c3c", "🔴"


def search_all_evidence(news_text):
    """
    Master evidence function.
    Returns: evidence, verdict, reason, trusted_count, comparison
    """
    query = _build_query(news_text)

    all_evidence = []
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(search_serpapi, query)
        f2 = executor.submit(search_newsapi, query)
        f3 = executor.submit(search_gdelt, query)
        all_evidence.extend(f1.result())
        all_evidence.extend(f2.result())
        all_evidence.extend(f3.result())

    seen   = set()
    unique = []
    for e in all_evidence:
        if e["url"] not in seen and e["url"]:
            seen.add(e["url"])
            unique.append(e)

    unique = sorted(unique,
                    key=lambda x: (x["is_trusted"], x["api"] == "Google"),
                    reverse=True)[:5]

    print(f"[Evidence] Total={len(unique)}")

    if not unique:
        return [], "LIKELY-FAKE", "Not found in any source", 0, {}

    trusted_count = sum(1 for e in unique if e["is_trusted"])
    print(f"[Evidence] Trusted={trusted_count}")

    comp_verdict, comp_reason, comp_detail = _compare_claim_vs_evidence(
        news_text, unique)
    comparison = {
        "verdict": comp_verdict,
        "reason":  comp_reason,
        "detail":  comp_detail,
    }

    _news_type = _detect_news_type(news_text)

    if comp_verdict == "CONFIRMED" and trusted_count >= 1:
        verdict = "CONFIRMED"
        reason  = f"Claim CONFIRMED by {trusted_count} trusted source(s)"
    elif comp_verdict == "CONFIRMED":
        # Confirmed even without a "trusted" source — official site confirmation
        verdict = "CONFIRMED"
        reason  = "Claim CONFIRMED by official source"
    elif comp_verdict == "CONTRADICTED":
        verdict = "CONTRADICTED"
        reason  = f"Evidence CONTRADICTS your claim — {comp_reason}"
    elif comp_verdict == "PARTIALLY-CONFIRMED":
        verdict = "PARTIALLY-CONFIRMED"
        reason  = comp_reason
    elif comp_verdict == "UNVERIFIED" and _news_type == "official_info":
        verdict = "UNVERIFIED"
        reason  = "Could not verify name on official college website"
    elif trusted_count >= 3:
        verdict = "REAL"
        reason  = f"Found in {trusted_count} trusted sources"
    elif trusted_count >= 1:
        verdict = "LIKELY-REAL"
        reason  = f"Found in {trusted_count} trusted source(s)"
    elif len(unique) >= 2:
        verdict = "UNVERIFIED"
        reason  = f"Found in {len(unique)} sources — none trusted"
    else:
        verdict = "LIKELY-FAKE"
        reason  = "Found in only 1 unknown source"

    return unique, verdict, reason, trusted_count, comparison


# ── Training state ─────────────────────────────────────────────────────────────
_train_log    = []
_train_status = 'idle'
_train_lock   = threading.Lock()


def _bg_train():
    global _train_log, _train_status
    try:
        with _train_lock:
            _train_status = 'running'
            _train_log    = []
        df = load_combined_dataset()
        def log(msg):
            _train_log.append(msg)
        detector = get_detector()
        metrics  = detector.train(
            df['News'].tolist(), df['target'].tolist(), progress_cb=log)
        _train_log.append(f"✔ Training complete! Accuracy: {metrics['accuracy']}%")
        _train_status = 'done'
    except Exception as e:
        _train_log.append(f"✘ Error: {str(e)}")
        _train_log.append(traceback.format_exc())
        _train_status = 'error'


# ── Auth views ─────────────────────────────────────────────────────────────────
def index(request):
    if 'username' in request.session:
        return redirect('dashboard')
    return redirect('register')


def Register(request):
    return render(request, 'Register.html', {})


def RegisterAction(request):
    if request.method == 'POST':
        username        = request.POST.get('t1', '').strip()
        password        = request.POST.get('t2', '').strip()
        confirm_password = request.POST.get('t6', '').strip()
        email           = request.POST.get('t4', '').strip()
        if not username or not password:
            return render(request, 'Register.html',
                          {'error': 'Username and password are required.'})
        if password != confirm_password:
            return render(request, 'Register.html',
                          {'error': 'Passwords do not match.'})
        if UserProfile.objects.filter(username=username).exists():
            return render(request, 'Register.html',
                          {'error': f'Username "{username}" already exists.'})
        import hashlib
        hashed = hashlib.sha256(password.encode()).hexdigest()
        UserProfile.objects.create(username=username, password=hashed,
                                   email=email, contact='', address='')
        return render(request, 'UserLogin.html',
                      {'success': 'Registration successful! Please login.'})
    return redirect('register')


def UserLogin(request):
    if 'username' in request.session:
        return redirect('dashboard')
    return render(request, 'UserLogin.html', {})


def UserLoginAction(request):
    if request.method == 'POST':
        import hashlib
        username = request.POST.get('t1', '').strip()
        password = request.POST.get('t2', '').strip()
        hashed   = hashlib.sha256(password.encode()).hexdigest()
        try:
            user = UserProfile.objects.get(username=username, password=hashed)
            request.session['username'] = user.username
            return redirect('dashboard')
        except UserProfile.DoesNotExist:
            return render(request, 'UserLogin.html',
                          {'error': 'Invalid username or password.'})
    return redirect('login')


def UserLogout(request):
    request.session.flush()
    return redirect('index')


def Dashboard(request):
    if 'username' not in request.session:
        return redirect('login')
    detector = get_detector()
    detector._lazy_load()
    history  = PredictionHistory.objects.filter(
        username=request.session['username']).order_by('-predicted_at')[:10]
    return render(request, 'Dashboard.html', {
        'username':   request.session['username'],
        'is_trained': detector.is_trained,
        'metrics':    detector.metrics,
        'history':    history,
    })


def LoadDataset(request):
    if 'username' not in request.session:
        return redirect('login')
    df    = load_combined_dataset()
    stats = get_dataset_stats(df)
    preview = df.head(50).to_dict('records')
    return render(request, 'LoadDataset.html', {
        'username': request.session['username'],
        'stats':    stats,
        'preview':  preview,
    })


def TrainModel(request):
    global _train_status, _train_log
    if 'username' not in request.session:
        return redirect('login')
    if request.method == 'POST':
        if _train_status != 'running':
            _train_status = 'running'
            _train_log    = ['Starting training pipeline...']
            t = threading.Thread(target=_bg_train, daemon=True)
            t.start()
        return render(request, 'Training.html', {
            'username': request.session['username'],
            'status':   _train_status,
        })
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'status': _train_status, 'log': _train_log})
    return render(request, 'Training.html', {
        'username': request.session['username'],
        'status':   _train_status,
    })


def Predict(request):
    if 'username' not in request.session:
        return redirect('login')
    detector = get_detector()
    return render(request, 'Predict.html', {
        'username':   request.session['username'],
        'is_trained': detector.is_trained,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 — PredictTextAction
# • Passes confidence to _get_news_tag()
# • UNVERIFIED + high AI confidence → keeps AI label (not UNCERTAIN)
# ══════════════════════════════════════════════════════════════════════════════
def PredictTextAction(request):
    if request.method == 'POST':
        if 'username' not in request.session:
            return redirect('login')
        news_text = request.POST.get('t1', '').strip()
        if not news_text:
            return render(request, 'Predict.html',
                          {'error': 'Please enter some news text.', 'is_trained': True})
        if len(news_text.split()) < 5:
            return render(request, 'Predict.html',
                          {'error': 'Please enter at least 5 words.', 'is_trained': True})
        detector = get_detector()
        if not detector.is_trained:
            return render(request, 'Predict.html',
                          {'error': 'Model not trained yet.', 'is_trained': False})
        try:
            result = detector.predict_one(news_text)
            evidence, verdict, reason, trusted, comparison = search_all_evidence(news_text)

            ev_verdict = comparison.get('verdict', '')
            ai_confidence = result['confidence']

            # FIX 3 — pass confidence so AI wins when evidence is inconclusive
            # Use the final resolved verdict instead of the intermediate comparison verdict
            news_tag, news_tag_color, news_tag_icon = _get_news_tag(
                result['label'], verdict, trusted, ai_confidence)

            # ── Display label logic ──────────────────────────────────────────
            display_label = result['label']
            display_color = result['color']
            display_icon  = result['icon']

            if verdict in ['CONFIRMED', 'REAL', 'LIKELY-REAL']:
                # Strong evidence that claim is true or heavily supported
                display_label = 'TRUE'
                display_color = '#27ae60'
                display_icon  = '✔'
            elif verdict == 'CONTRADICTED':
                display_label = 'FALSE'
                display_color = '#e74c3c'
                display_icon  = '✘'
            else:
                display_label = result['label']
                display_color = result['color']
                display_icon  = result['icon']

            result['display_label'] = display_label
            result['display_color'] = display_color
            result['display_icon']  = display_icon

            PredictionHistory.objects.create(
                username   = request.session['username'],
                news_text  = news_text[:500],
                prediction = result['label'],
                confidence = result['confidence'],
            )
            return render(request, 'Result.html', {
                'username':       request.session['username'],
                'news_text':      news_text,
                'result':         result,
                'source':         'Manual Input',
                'evidence':       evidence,
                'verdict':        verdict,
                'reason':         reason,
                'trusted_count':  trusted,
                'comparison':     comparison,
                'news_tag':       news_tag,
                'news_tag_color': news_tag_color,
                'news_tag_icon':  news_tag_icon,
            })
        except Exception as e:
            return render(request, 'Predict.html',
                          {'error': str(e), 'is_trained': True})
    return redirect('predict')


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 (URL version) — PredictURLAction
# Same fixes as PredictTextAction applied here too.
# ══════════════════════════════════════════════════════════════════════════════
def PredictURLAction(request):
    if request.method == 'POST':
        if 'username' not in request.session:
            return redirect('login')
        url = request.POST.get('url', '').strip()
        if not url:
            return render(request, 'Predict.html',
                          {'error': 'Please enter a URL.', 'is_trained': True})
        scraped = scrape_url(url)
        if scraped['error']:
            return render(request, 'Predict.html', {
                'error': f"Could not scrape URL: {scraped['error']}",
                'is_trained': True,
            })
        if not scraped['text']:
            return render(request, 'Predict.html',
                          {'error': 'No text found at URL.', 'is_trained': True})
        detector = get_detector()
        if not detector.is_trained:
            return render(request, 'Predict.html',
                          {'error': 'Model not trained yet.', 'is_trained': False})
        try:
            result = detector.predict_one(scraped['text'])
            evidence, verdict, reason, trusted, comparison = search_all_evidence(scraped['text'])

            ev_verdict    = comparison.get('verdict', '')
            ai_confidence = result['confidence']

            # FIX 3 — pass final verdict
            news_tag, news_tag_color, news_tag_icon = _get_news_tag(
                result['label'], verdict, trusted, ai_confidence)

            display_label = result['label']
            display_color = result['color']
            display_icon  = result['icon']

            if verdict in ['CONFIRMED', 'REAL', 'LIKELY-REAL']:
                display_label = 'TRUE'
                display_color = '#27ae60'
                display_icon  = '✔'
            elif verdict == 'CONTRADICTED':
                display_label = 'FALSE'
                display_color = '#e74c3c'
                display_icon  = '✘'
            else:
                display_label = result['label']
                display_color = result['color']
                display_icon  = result['icon']

            result['display_label'] = display_label
            result['display_color'] = display_color
            result['display_icon']  = display_icon

            PredictionHistory.objects.create(
                username   = request.session['username'],
                news_text  = scraped['text'][:500],
                source_url = url,
                prediction = result['label'],
                confidence = result['confidence'],
            )
            return render(request, 'Result.html', {
                'username':       request.session['username'],
                'news_text':      scraped['text'][:800],
                'news_title':     scraped['title'],
                'result':         result,
                'source':         url,
                'evidence':       evidence,
                'verdict':        verdict,
                'reason':         reason,
                'trusted_count':  trusted,
                'comparison':     comparison,
                'news_tag':       news_tag,
                'news_tag_color': news_tag_color,
                'news_tag_icon':  news_tag_icon,
            })
        except Exception as e:
            return render(request, 'Predict.html',
                          {'error': str(e), 'is_trained': True})
    return redirect('predict')


def PredictFileAction(request):
    if request.method == 'POST':
        if 'username' not in request.session:
            return redirect('login')
        if 'csvfile' not in request.FILES:
            return render(request, 'Predict.html',
                          {'error': 'Please upload a CSV file.', 'is_trained': True})
        detector = get_detector()
        if not detector.is_trained:
            return render(request, 'Predict.html',
                          {'error': 'Model not trained yet.', 'is_trained': False})
        try:
            f   = request.FILES['csvfile']
            df  = pd.read_csv(f, encoding='utf-8')
            col = next((c for c in df.columns
                        if c.lower() in ['news', 'text', 'statement', 'headline']), None)
            if col is None:
                col = df.columns[0]
            texts   = df[col].astype(str).tolist()
            results = detector.predict_batch(texts)
            rows    = []
            for text, res in zip(texts, results):
                rows.append({
                    'text':       text[:200],
                    'label':      res['label'],
                    'confidence': res['confidence'],
                    'color':      res['color'],
                    'icon':       res['icon'],
                })
                PredictionHistory.objects.create(
                    username   = request.session['username'],
                    news_text  = text[:500],
                    prediction = res['label'],
                    confidence = res['confidence'],
                )
            return render(request, 'BatchResult.html', {
                'username': request.session['username'],
                'rows':     rows,
                'total':    len(rows),
            })
        except Exception as e:
            return render(request, 'Predict.html',
                          {'error': str(e), 'is_trained': True})
    return redirect('predict')


def Results(request):
    if 'username' not in request.session:
        return redirect('login')
    detector = get_detector()
    detector._lazy_load()
    if not detector.is_trained or not detector.metrics:
        return redirect('train_model')
    metrics = detector.metrics
    labels  = metrics.get('labels', [])
    cm      = np.array(metrics.get('conf_matrix', []))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    if cm.size > 0 and len(labels) > 0:
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[0], linewidths=0.5)
        axes[0].set_title('Confusion Matrix', color='white', fontsize=13)
        axes[0].tick_params(colors='white')
        axes[0].set_facecolor('#1a1d27')
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_vals  = [metrics.get('accuracy', 0), metrics.get('precision', 0),
                    metrics.get('recall', 0), metrics.get('f1', 0)]
    colors_bar   = ['#27ae60', '#3498db', '#f39c12', '#9b59b6']
    bars = axes[1].bar(metric_names, metric_vals, color=colors_bar, edgecolor='none')
    for bar, val in zip(bars, metric_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom',
                     color='white', fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].set_title('Model Performance Metrics', color='white', fontsize=13)
    axes[1].set_facecolor('#1a1d27')
    axes[1].tick_params(colors='white')
    axes[1].spines['bottom'].set_color('#444')
    axes[1].spines['left'].set_color('#444')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    chart_b64 = base64.b64encode(buf.getvalue()).decode()

    # Pre-process per-class report for template rendering
    report = metrics.get('report', {})
    class_rows = []
    for label in labels:
        row_data = report.get(label, {})
        if isinstance(row_data, dict):
            class_rows.append({
                'label':     label,
                'precision': round(row_data.get('precision', 0) * 100, 1),
                'recall':    round(row_data.get('recall', 0) * 100, 1),
                'f1':        round(row_data.get('f1-score', 0) * 100, 1),
                'support':   int(row_data.get('support', 0)),
            })

    return render(request, 'Results.html', {
        'username':    request.session['username'],
        'metrics':     metrics,
        'chart':       chart_b64,
        'labels':      labels,
        'class_rows':  class_rows,
    })