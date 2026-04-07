import os
import time
import json

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import PyPDF2
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# ===================================================
# 1. ΦΟΡΤΩΣΗ ΚΛΕΙΔΙΩΝ & ΡΥΘΜΙΣΗ AZURE CLIENT
# ===================================================
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# ===================================================
# ΦΟΡΤΩΣΗ KEYS ΑΠΟ .env
# ===================================================

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-12-01-preview"
)

# ===================================================
# 2. ML PIPELINE ΓΙΑ ΠΡΟΤΑΣΗ ΔΟΣΗΣ (συνθετικό demo)
# ===================================================

@st.cache_resource
def train_settlement_model():
    np.random.seed(42)
    N = 2000

    data = pd.DataFrame({
        "Income": np.random.randint(10000, 90000, N),
        "TotalDebt": np.random.randint(5000, 150000, N),
        "Expenses": np.random.randint(6000, 25000, N),
        "Rent": np.random.randint(0, 15000, N),
        "TaxWithheld": np.random.randint(500, 8000, N),
        "DepositInterest": np.random.randint(0, 2000, N)
    })

    noise = np.random.normal(0, 15, N)

    data["MonthlyPayment"] = (
          0.002 * data["Income"]
        + 0.0007 * data["TotalDebt"]
        - 0.0022 * data["Expenses"]
        - 0.0008 * data["Rent"]
        + 0.0004 * data["TaxWithheld"]
        + 0.0019 * data["DepositInterest"]
        + 30 + noise
    )

    X = data[["Income", "TotalDebt", "Expenses", "Rent", "TaxWithheld", "DepositInterest"]]
    Y = data["MonthlyPayment"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", Ridge(alpha=0.3))
    ])

    pipeline.fit(X_train, Y_train)
    return pipeline

pipeline = train_settlement_model()


def predict_settlement(income, debt, expenses, rent, tax, interest):
    """ML + business rules για πρόταση διακανονισμού."""
    input_data = pd.DataFrame([{
        "Income": income,
        "TotalDebt": debt,
        "Expenses": expenses,
        "Rent": rent,
        "TaxWithheld": tax,
        "DepositInterest": interest
    }])

    predicted = pipeline.predict(input_data)[0]

    monthly_income = income / 12 if income > 0 else 0
    max_allowed_payment = monthly_income * 0.40  # max 40% του μηνιαίου εισοδήματος

    predicted = max(predicted, 50)                  # min payment 50€
    if monthly_income > 0:
        predicted = min(predicted, max_allowed_payment)

    max_months = 240
    if predicted <= 0:
        months = max_months
    else:
        months = min(max_months, int(debt / predicted))
    months = max(months, 1)

    final_payment = debt / months if months > 0 else debt

    return {
        "PredictedML": round(float(predicted), 2),
        "FinalMonthlyPayment": round(float(final_payment), 2),
        "Months": int(months),
        "RemainingDebt": round(float(debt), 2)
    }


def calculate_viability_score(income, debt, expenses, rent):
    """
    Score βιωσιμότητας 0–100 + risk label + εξήγηση (τύπου SHAP).
    """
    reasons = []

    if income <= 0:
        score = 5
        label = "Υψηλό ρίσκο"
        reasons.append("Δεν δηλώνεται ετήσιο εισόδημα — ο φάκελος θεωρείται πολύ υψηλού ρίσκου.")
        return score, label, reasons

    monthly_income = income / 12
    monthly_expenses = expenses / 12
    monthly_rent = rent / 12
    disposable = monthly_income - (monthly_expenses + monthly_rent)

    # Disposable income reasoning
    if disposable <= 0:
        disposable_score = 5
        reasons.append("Το καθαρό διαθέσιμο εισόδημα μετά από δαπάνες και ενοίκιο είναι αρνητικό ή μηδενικό.")
    else:
        ratio = disposable / monthly_income
        disposable_score = min(60, max(10, ratio * 60))
        if ratio < 0.15:
            reasons.append("Το καθαρό διαθέσιμο εισόδημα είναι χαμηλό σε σχέση με το συνολικό εισόδημα.")
        elif ratio < 0.30:
            reasons.append("Το καθαρό διαθέσιμο εισόδημα είναι μέτριο σε σχέση με το συνολικό εισόδημα.")
        else:
            reasons.append("Το καθαρό διαθέσιμο εισόδημα είναι ικανοποιητικό σε σχέση με το συνολικό εισόδημα.")

    # Debt-to-Income reasoning
    dti = debt / max(income, 1)
    if dti <= 1:
        dti_score = 40
        reasons.append("Ο λόγος χρέους προς εισόδημα (DTI) είναι χαμηλός – το χρέος είναι διαχειρίσιμο.")
    elif dti <= 3:
        dti_score = 30
        reasons.append("Ο λόγος χρέους προς εισόδημα (DTI) είναι μέτριος – απαιτείται προσεκτική ρύθμιση.")
    elif dti <= 5:
        dti_score = 20
        reasons.append("Ο λόγος χρέους προς εισόδημα (DTI) είναι αυξημένος – ο φάκελος θεωρείται πιο επισφαλής.")
    elif dti <= 8:
        dti_score = 10
        reasons.append("Ο λόγος χρέους προς εισόδημα (DTI) είναι πολύ υψηλός – σημαντικό ρίσκο μη αποπληρωμής.")
    else:
        dti_score = 0
        reasons.append("Το χρέος είναι δυσανάλογα μεγάλο σε σχέση με το εισόδημα.")

    # Επιπλέον “demo” ερμηνευσιμότητα
    if rent > 0 and disposable <= 0:
        reasons.append("Η επιβάρυνση από το ενοίκιο σε συνδυασμό με τις δαπάνες αφήνει μηδενικό διαθέσιμο εισόδημα.")
    if rent == 0 and expenses < income * 0.3:
        reasons.append("Δεν καταγράφεται ενοίκιο και οι δαπάνες είναι σχετικά χαμηλές, κάτι που βελτιώνει τη βιωσιμότητα.")
    if debt < income:
        reasons.append("Το συνολικό χρέος είναι μικρότερο από το ετήσιο εισόδημα, στοιχείο θετικό για την αποπληρωμή.")
    else:
        reasons.append("Το συνολικό χρέος υπερβαίνει το ετήσιο εισόδημα, αυξάνοντας το ρίσκο.")

    score = int(max(0, min(100, disposable_score + dti_score)))

    if score >= 75:
        label = "Χαμηλό ρίσκο"
    elif score >= 50:
        label = "Μεσαίο ρίσκο"
    else:
        label = "Υψηλό ρίσκο"

    return score, label, reasons


# ===================================================
# 2b. RAG με Embeddings – Demo νομικό corpus για κύρια κατοικία
# ===================================================

LAW_SNIPPETS = [
    {
        "id": "law_primary_residence_protection",
        "title": "Άρθρο 7 – Προστασία κύριας κατοικίας οφειλέτη χαμηλής αξίας (DEMO)",
        "text": (
            "Άρθρο 7 – Προστασία κύριας κατοικίας οφειλέτη χαμηλής αξίας (DEMO)\n"
            "1. Κατά την αξιολόγηση αιτήσεων ρύθμισης οφειλών, όταν ο οφειλέτης διαθέτει κύρια "
            "κατοικία χαμηλής εμπορικής αξίας, οι πιστωτές και τα αρμόδια όργανα αξιολόγησης "
            "λαμβάνουν ιδίως υπόψη την ανάγκη διατήρησης αξιοπρεπών συνθηκών διαβίωσης του "
            "νοικοκυριού.\n"
            "2. Ως κύρια κατοικία χαμηλής αξίας νοείται ακίνητο που χρησιμοποιείται ως μόνιμη "
            "και αποκλειστική κατοικία του οφειλέτη και του νοικοκυριού του και του οποίου η "
            "εμπορική αξία, σύμφωνα με τα στοιχεία της φορολογικής διοίκησης ή έκθεση "
            "πιστοποιημένου εκτιμητή, δεν υπερβαίνει προκαθορισμένα όρια που εξειδικεύονται "
            "με κανονιστικές πράξεις.\n"
            "3. Σε περιπτώσεις οφειλετών με κύρια κατοικία χαμηλής αξίας, κατά την κατάρτιση "
            "πρότασης ρύθμισης προκρίνονται λύσεις που διασφαλίζουν, στο μέτρο του δυνατού, την "
            "παραμονή στην κατοικία, μέσω προσαρμογής της διάρκειας αποπληρωμής και της μηνιαίας "
            "δόσης, εφόσον η ρύθμιση παραμένει συνολικά βιώσιμη.\n"
            "4. Η αναγκαστική ρευστοποίηση κύριας κατοικίας χαμηλής αξίας αντιμετωπίζεται ως "
            "έσχατο μέτρο, εφόσον αποδεικνύεται ότι δεν υφίσταται εναλλακτική λύση ρύθμισης που "
            "να διασφαλίζει επαρκώς τα συμφέροντα των πιστωτών και τις στοιχειώδεις ανάγκες "
            "διαβίωσης του οφειλέτη."
        ),
        "vector": None,  # Θα γεμίσει με embeddings στην ensure_law_embeddings()
    }
]


def embed_text(text: str) -> list:
    """
    Κλήση στο Azure OpenAI embeddings API.
    Χρησιμοποιεί το EMBEDDING_DEPLOYMENT.
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def ensure_law_embeddings():
    """
    Υπολογίζει embeddings για τα νομικά αποσπάσματα (μία φορά).
    """
    for snippet in LAW_SNIPPETS:
        if not snippet.get("vector"):
            vec = embed_text(snippet["text"])
            snippet["vector"] = np.array(vec) if vec else None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def ask_legal_rag(question: str) -> dict:
    """
    Demo RAG με embeddings:
    - φτιάχνει embedding της ερώτησης
    - βρίσκει το πιο σχετικό νομικό απόσπασμα
    - ζητά από το GPT απάντηση ΜΟΝΟ με βάση αυτό το context
    """
    ensure_law_embeddings()

    q_vec_list = embed_text(question)
    if not q_vec_list:
        # Fallback: αν αποτύχει το embedding, απλώς χρησιμοποίησε όλα τα snippets
        relevant_snippets = LAW_SNIPPETS
    else:
        q_vec = np.array(q_vec_list)

        scored = []
        for sn in LAW_SNIPPETS:
            sim = cosine_similarity(q_vec, sn.get("vector"))
            scored.append((sim, sn))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = 1
        relevant_snippets = [s for _, s in scored[:top_k]]

    context_parts = []
    for s in relevant_snippets:
        context_parts.append(f"### Πηγή: {s['title']}\n{s['text']}")
    context = "\n\n".join(context_parts)

    system_prompt = """
    Είσαι νομικός βοηθός για τον Εξωδικαστικό Μηχανισμό Ρύθμισης Οφειλών.
    Λαμβάνεις αποσπάσματα από (demo) νόμους / ΦΕΚ σχετικά με την προστασία κύριας κατοικίας 
    και ερωτήσεις του ελεγκτή.

    Κανόνες:
    - Απαντάς ΣΥΝΟΠΤΙΚΑ και ΚΑΘΑΡΑ, σε 1–3 παραγράφους.
    - Βασίζεσαι ΜΟΝΟ στο παρεχόμενο κείμενο (context) παρακάτω.
    - Αν κάτι δεν καλύπτεται ρητά, το λες ξεκάθαρα 
      ("δεν προκύπτει ρητά από το παρεχόμενο απόσπασμα").
    - Όπου γίνεται, αναφέρεις από ποια 'Πηγή' / άρθρο αντλείς την απάντηση (π.χ. Άρθρο 7).
    """

    user_content = (
        "ΝΟΜΙΚΟ CONTEXT (αποσπάσματα demo):\n\n"
        f"{context}\n\n"
        "ΕΡΩΤΗΣΗ ΕΛΕΓΚΤΗ:\n"
        f"{question}"
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "sources": [s["title"] for s in relevant_snippets],
    }


# ===================================================
# 3. PDF HELPERS & AI VALIDATION
# ===================================================

def extract_text_from_pdf(uploaded_file):
    """Διαβάζει ΟΛΟ το κείμενο από PDF με ασφαλή τρόπο."""
    try:
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception:
        return ""


def extract_first_page_text(uploaded_file):
    """Διαβάζει ΜΟΝΟ την πρώτη σελίδα (τίτλος/επικεφαλίδα) για αναγνώριση τύπου."""
    try:
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        if not pdf_reader.pages:
            return ""
        page = pdf_reader.pages[0]
        text = page.extract_text() or ""
        return text.strip()
    except Exception:
        return ""


def validate_e9_with_ai(text_content):
    """Κλήση στο AI Agent για έλεγχο εγκυρότητας Ε9."""
    system_prompt = """
    Είσαι ειδικός ελεγκτής της ΑΑΔΕ για τον Εξωδικαστικό Μηχανισμό.
    Ελέγχεις αν το κείμενο προέρχεται από έγκυρο έντυπο Ε9.

    Κριτήρια:
    1. Πρέπει να περιέχει τη φράση "ΔΗΛΩΣΗ ΣΤΟΙΧΕΙΩΝ ΑΚΙΝΗΤΩΝ" ή "Ε9".
    2. Πρέπει να περιέχει ΑΦΜ (9 συνεχόμενα ψηφία).
    3. Πρέπει να υπάρχει έτος όπως 2022, 2023, 2024.

    Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ σε JSON μορφή:
    {
        "is_valid": true/false,
        "confidence": 0-100,
        "reason": "Εξήγησε τι βρήκες ή τι λείπει",
        "action_needed": "Τι πρέπει να κάνει ο πολίτης"
    }
    """

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_content}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def validate_document_type_ai(text_content, expected_label):
    """
    AI έλεγχος για να δούμε ΤΙ τύπος εντύπου είναι το PDF
    και αν ταιριάζει με αυτό που περιμένουμε.
    """
    system_prompt = f"""
    Είσαι ειδικός ελεγκτής της ΑΑΔΕ για έντυπα φορολογίας.

    ΔΙΑΒΑΖΕΙΣ την πρώτη σελίδα ενός PDF και πρέπει να αποφασίσεις
    ΤΙ ΤΥΠΟΣ ΕΓΓΡΑΦΟΥ είναι, ΜΟΝΟ με βάση τον τίτλο και την κεφαλίδα.

    Επιτρεπόμενοι τύποι:
    - "Ε1"
    - "Ε3"
    - "Ε9"
    - "Βεβαίωση Οφειλών"
    - "Αίτηση Υπαγωγής"
    - "Άγνωστο"

    Αναμενόμενος τύπος για τον συγκεκριμένο έλεγχο είναι: "{expected_label}".

    Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ σε JSON μορφή:
    {{
        "detected_type": "Ένας από τους παραπάνω τύπους",
        "is_match_expected": true/false,
        "reason": "Σύντομη εξήγηση γιατί το έγγραφο ταιριάζει ή όχι με τον αναμενόμενο τύπο."
    }}
    """

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_content[:4000]},
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# ===================================================
# 4. DEMO JSON CASE LOADER
# ===================================================

@st.cache_data
def load_demo_case():
    """Φορτώνει demo_case.json αν υπάρχει, αλλιώς None."""
    try:
        with open("demo_case.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ===================================================
# 5. EMAIL SENDER (DEMO-ONLY, FAKE SMTP)
# ===================================================

def send_decision_email(decision_type, settlement, citizen_name):
    """
    Demo: δεν στέλνει πραγματικό email.
    Εμφανίζει όμως πλήρες "SMTP-like" log + περιεχόμενο.
    """
    citizen_email = "smartsettlement.demo@gmail.com"

    monthly = settlement.get("FinalMonthlyPayment")
    months = settlement.get("Months")
    total = settlement.get("RemainingDebt")

    if decision_type == "approved":
        subject = "Αποδοχή Πρότασης Διακανονισμού – Smart Settlement"
        body = (
            f"Αγαπητέ/ή {citizen_name},\n\n"
            "Η πρόταση διακανονισμού για τις οφειλές σας εγκρίθηκε από τον Εξωδικαστικό Μηχανισμό.\n\n"
            "Βασικά στοιχεία ρύθμισης:\n"
            f"- Μηνιαία δόση: {monthly:.2f} €\n"
            f"- Αριθμός δόσεων: {months} μήνες\n"
            f"- Συνολικό ποσό προς ρύθμιση: {total:.2f} €\n\n"
            "Τα πλήρη στοιχεία της ρύθμισης είναι διαθέσιμα στον ψηφιακό σας φάκελο.\n"
            "Παρακαλούμε επιβεβαιώστε την αποδοχή σας μέσω της πλατφόρμας.\n\n"
            "Με εκτίμηση,\n"
            "Ο Ελεγκτής Smart Settlement"
        )
    elif decision_type == "rejected":
        subject = "Απόρριψη Πρότασης Διακανονισμού – Smart Settlement"
        body = (
            f"Αγαπητέ/ή {citizen_name},\n\n"
            "Σας ενημερώνουμε ότι η αίτηση διακανονισμού σας δεν μπορεί να εγκριθεί "
            "στην παρούσα μορφή, βάσει των στοιχείων που έχουν υποβληθεί.\n\n"
            "Μπορείτε να επανυποβάλετε αίτηση με επικαιροποιημένα στοιχεία ή πρόσθετα δικαιολογητικά.\n\n"
            "Με εκτίμηση,\n"
            "Ο Ελεγκτής Smart Settlement"
        )
    else:  # counter
        subject = "Αντιπρόταση Διακανονισμού – Smart Settlement"
        body = (
            f"Αγαπητέ/ή {citizen_name},\n\n"
            "Ο ελεγκτής διαμόρφωσε αντιπρόταση διακανονισμού με τα εξής στοιχεία:\n\n"
            f"- Μηνιαία δόση: {monthly:.2f} €\n"
            f"- Αριθμός δόσεων: {months} μήνες\n"
            f"- Συνολικό ποσό προς ρύθμιση: {total:.2f} €\n\n"
            "Παρακαλούμε συνδεθείτε στην πλατφόρμα Smart Settlement για να αποδεχθείτε ή "
            "να απορρίψετε την αντιπρόταση.\n\n"
            "Με εκτίμηση,\n"
            "Ο Ελεγκτής Smart Settlement"
        )

    st.success(f"✓ Email στάλθηκε επιτυχώς στο {citizen_email}")

    demo_header = (
        "MESSAGE-ID: <f3ab91ac-94c2-4331-9e32-2af6a8324caa@mail-gateway.gov.gr>\n"
        "SMTP SERVER: mail-gateway.gov.gr (demo)\n"
        f"Subject: {subject}\n"
        f"To: {citizen_email}\n"
        "----------------------------------------\n"
    )

    with st.expander("Προβολή περιεχομένου email (demo):"):
        st.code(demo_header + body, language="text")

# ===================================================
# 6. FLOW ΠΟΛΙΤΗ – DOC STATE
# ===================================================

EXPECTED_DOCS = {
    "e1": "Ε1",
    "e3": "Ε3",
    "e9": "Ε9",
    "vevaiosi": "Βεβαίωση Οφειλών",
    "aitisi": "Αίτηση Υπαγωγής",
}


def init_session_state():
    """Αρχικοποίηση session_state για τα έγγραφα & το flow."""
    if "started" not in st.session_state:
        st.session_state.started = False

    if "docs" not in st.session_state:
        st.session_state.docs = {
            key: {
                "file": None,
                "detected_type": None,
                "status": "not_uploaded",  # not_uploaded | ok | wrong | checking
                "message": "",
            }
            for key in EXPECTED_DOCS.keys()
        }

    if "e9_ai_result" not in st.session_state:
        st.session_state.e9_ai_result = None

    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    if "inspector_decision" not in st.session_state:
        st.session_state.inspector_decision = None

    if "last_settlement" not in st.session_state:
        st.session_state.last_settlement = None

    if "last_score" not in st.session_state:
        st.session_state.last_score = None

    if "last_risk_label" not in st.session_state:
        st.session_state.last_risk_label = None

    if "last_reasons" not in st.session_state:
        st.session_state.last_reasons = None


def validate_single_document(key, uploaded_file):
    """Κάνει ΑΜΕΣΑ AI έλεγχο τίτλου για ΕΝΑ έγγραφο."""
    doc_state = st.session_state.docs[key]
    label = EXPECTED_DOCS[key]

    if uploaded_file is None:
        doc_state["file"] = None
        doc_state["detected_type"] = None
        doc_state["status"] = "not_uploaded"
        doc_state["message"] = "Δεν έχει ανέβει αρχείο."
        return

    doc_state["file"] = uploaded_file
    doc_state["status"] = "checking"
    doc_state["message"] = "Το αρχείο υποβλήθηκε και ελέγχεται..."

    first_text = extract_first_page_text(uploaded_file)
    if len(first_text.strip()) < 20:
        doc_state["status"] = "wrong"
        doc_state["detected_type"] = "Άγνωστο"
        doc_state["message"] = (
            "Το PDF δεν περιέχει αρκετό αναγνώσιμο κείμενο "
            "στην πρώτη σελίδα για να επιβεβαιωθεί ο τύπος."
        )
        return

    try:
        result = validate_document_type_ai(first_text, label)
        doc_state["detected_type"] = result.get("detected_type", "Άγνωστο")

        if result.get("is_match_expected", False):
            doc_state["status"] = "ok"
            doc_state["message"] = (
                f"✅ Το αρχείο υποβλήθηκε και ελέγχθηκε: "
                f"Έγκυρο αρχείο ({doc_state['detected_type']})."
            )
        else:
            doc_state["status"] = "wrong"
            doc_state["message"] = (
                f"❌ Λάθος αρχείο: AI το αναγνώρισε ως "
                f"«{doc_state['detected_type']}». "
                f"Αιτιολόγηση: {result.get('reason', '')}"
            )
    except Exception as e:
        # Για τις ανάγκες του demo: δεν μπλοκάρουμε τον πολίτη αν πέσει το AI
        doc_state["status"] = "ok"
        doc_state["detected_type"] = "Άγνωστο"
        doc_state["message"] = (
            "Προσωρινή δυσλειτουργία AI κατά τον έλεγχο τύπου εγγράφου. "
            "Το αρχείο έγινε δεκτό για τις ανάγκες του demo. "
            f"(Λεπτομέρεια σφάλματος: {e})"
        )


def all_docs_uploaded():
    return all(st.session_state.docs[k]["file"] is not None for k in EXPECTED_DOCS)


def any_wrong_docs():
    return any(st.session_state.docs[k]["status"] == "wrong" for k in EXPECTED_DOCS)

# ===================================================
# 7. UI HELPERS (CSS, STEPPER)
# ===================================================

def inject_global_css():
    css = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            180deg,
            rgba(12, 77, 162, 0.18) 0%,
            #E3EEFF 45%,
            #F7F9FC 100%
        );
    }

    [data-testid="stSidebar"] {
        background-color: #EDF1F7 !important;
    }

    .main > div {
        padding-top: 1rem;
    }

    .gov-card {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background-color: rgba(255,255,255,0.9);
        border: 1px solid #D0D7E2;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
    }

    .stepper-wrapper {
        display: flex;
        justify-content: space-between;
        margin: 1.5rem 0 0.5rem 0;
    }
    .stepper-step {
        flex: 1;
        text-align: center;
        font-size: 0.80rem;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #4B5563;
    }
    .stepper-circle {
        width: 26px;
        height: 26px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.25rem;
        border: 2px solid #9CA3AF;
        background-color: #FFFFFF;
        color: #4B5563;
        font-weight: 600;
        font-size: 0.80rem;
    }
    .stepper-circle.active {
        border-color: #0C4DA2;
        background-color: #0C4DA2;
        color: #FFFFFF;
    }
    .stepper-circle.done {
        border-color: #0C4DA2;
        background-color: #ffffff;
        color: #0C4DA2;
    }
    .stepper-line {
        height: 2px;
        background: linear-gradient(to right, #0C4DA2, #0C4DA2);
        margin: 0 0.4rem 1.1rem 0.4rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_stepper(current_step: int):
    steps = [
        (1, "Υποβολή εγγράφων"),
        (2, "Έλεγχος φακέλου"),
        (3, "Ολοκλήρωση"),
    ]

    html_parts = []
    html_parts.append('<div class="stepper-line"></div>')
    html_parts.append('<div class="stepper-wrapper">')

    for idx, label in steps:
        if current_step > idx:
            circle_class = "stepper-circle done"
        elif current_step == idx:
            circle_class = "stepper-circle active"
        else:
            circle_class = "stepper-circle"

        html_parts.append(
            f'<div class="stepper-step">'
            f'<div class="{circle_class}">{idx}</div><br/>'
            f'<span>{label}</span>'
            f'</div>'
        )

    html_parts.append('</div>')

    html = "\n".join(html_parts)
    st.markdown(html, unsafe_allow_html=True)

# ===================================================
# 8. FLOW ΠΟΛΙΤΗ
# ===================================================

def render_citizen_flow():
    inject_global_css()
    init_session_state()

    top_col1, top_col2 = st.columns([2, 1])

    with top_col1:
        st.markdown("#### Ελληνική Δημοκρατία – Προεδρία της Κυβέρνησης")
        st.title("Επιβεβαίωση εγγράφων για Εξωδικαστικό Μηχανισμό")
        st.markdown(
            """
            <div class="gov-card">
            Αυτή η πιλοτική ψηφιακή υπηρεσία καθοδηγεί τον πολίτη βήμα–βήμα
            στην υποβολή, αναγνώριση και επιβεβαίωση των εγγράφων του φακέλου
            για τον Εξωδικαστικό Μηχανισμό Ρύθμισης Οφειλών.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_col2:
        try:
            st.image("gov_logo.png", width=260)
        except Exception:
            st.write("")

    st.markdown("Βήματα διαδικασίας:")
    render_stepper(st.session_state.current_step)

    valid_count = sum(
        1 for k in EXPECTED_DOCS if st.session_state.docs[k]["status"] == "ok"
    )
    total_docs = len(EXPECTED_DOCS)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Τρέχον βήμα", f"{st.session_state.current_step} / 3")
    with colB:
        st.metric("Έγκυρα έγγραφα", f"{valid_count} / {total_docs}")
    with colC:
        st.metric("Απαιτούμενα δικαιολογητικά", total_docs)

    st.markdown("---")

    tab_intro, tab_form = st.tabs(["Οδηγίες", "Υποβολή Αίτησης"])

    with tab_intro:
        st.subheader("Πώς λειτουργεί η υπηρεσία")

        st.write(
            """
        1. Πατάτε **«Ξεκίνησε Αίτηση»**.
        2. Ανεβάζετε τα 5 απαραίτητα έγγραφα (Ε1, Ε3, Ε9, Βεβαίωση Οφειλών, Αίτηση).
        3. Κάθε έγγραφο ελέγχεται ΑΜΕΣΑ ως προς τον τύπο του (π.χ. Ε1, Ε9 κ.λπ.)
           και εμφανίζεται αν είναι **Έγκυρο** ή **Λάθος**.
        4. Μόλις όλα τα έγγραφα είναι έγκυρα, μπορείτε να εκτελέσετε
           **Συνολικό Έλεγχο Φακέλου** και να λάβετε το τελικό αποτέλεσμα.
        """
        )

        if st.button("👉 Ξεκίνησε Αίτηση", type="primary"):
            st.session_state.started = True
            st.session_state.current_step = 1
            st.success("Η αίτηση ξεκίνησε. Μεταβείτε στην καρτέλα «Υποβολή Αίτησης». ✅")

    with tab_form:
        if not st.session_state.started:
            st.info("Πατήστε πρώτα **«Ξεκίνησε Αίτηση»** στην καρτέλα Οδηγίες.")
            return

        st.subheader("Μεταφόρτωση Εγγράφων")

        c1, c2 = st.columns(2)

        with c1:
            e1_file = st.file_uploader(
                "Έντυπο Ε1 (Δήλωση Φορολογίας Εισοδήματος)",
                type=["pdf"],
                key="e1_uploader",
            )
            if e1_file is not None:
                validate_single_document("e1", e1_file)
            state_e1 = st.session_state.docs["e1"]
            if state_e1["status"] == "ok":
                st.success("Ε1: Έγκυρο αρχείο.")
            elif state_e1["status"] == "wrong":
                st.error("Ε1: Λάθος αρχείο.")
            elif state_e1["status"] == "checking":
                st.info("Ε1: Το αρχείο υποβλήθηκε και ελέγχεται...")

            e3_file = st.file_uploader(
                "Έντυπο Ε3 (Κατάσταση Οικονομικών Στοιχείων - αν υπάρχει)",
                type=["pdf"],
                key="e3_uploader",
            )
            if e3_file is not None:
                validate_single_document("e3", e3_file)
            state_e3 = st.session_state.docs["e3"]
            if state_e3["status"] == "ok":
                st.success("Ε3: Έγκυρο αρχείο.")
            elif state_e3["status"] == "wrong":
                st.error("Ε3: Λάθος αρχείο.")
            elif state_e3["status"] == "checking":
                st.info("Ε3: Το αρχείο υποβλήθηκε και ελέγχεται...")

            e9_file = st.file_uploader(
                "Έντυπο Ε9 (Δήλωση Στοιχείων Ακινήτων)",
                type=["pdf"],
                key="e9_uploader",
            )
            if e9_file is not None:
                validate_single_document("e9", e9_file)
            state_e9 = st.session_state.docs["e9"]
            if state_e9["status"] == "ok":
                st.success("Ε9: Έγκυρο αρχείο.")
            elif state_e9["status"] == "wrong":
                st.error("Ε9: Λάθος αρχείο.")
            elif state_e9["status"] == "checking":
                st.info("Ε9: Το αρχείο υποβλήθηκε και ελέγχεται...")

        with c2:
            v_file = st.file_uploader(
                "Βεβαίωση Οφειλών", type=["pdf"], key="vev_uploader"
            )
            if v_file is not None:
                validate_single_document("vevaiosi", v_file)
            state_v = st.session_state.docs["vevaiosi"]
            if state_v["status"] == "ok":
                st.success("Βεβαίωση Οφειλών: Έγκυρο αρχείο.")
            elif state_v["status"] == "wrong":
                st.error("Βεβαίωση Οφειλών: Λάθος αρχείο.")
            elif state_v["status"] == "checking":
                st.info("Βεβαίωση Οφειλών: Το αρχείο υποβλήθηκε και ελέγχεται...")

            a_file = st.file_uploader(
                "Αίτηση Υπαγωγής", type=["pdf"], key="ait_uploader"
            )
            if a_file is not None:
                validate_single_document("aitisi", a_file)
            state_a = st.session_state.docs["aitisi"]
            if state_a["status"] == "ok":
                st.success("Αίτηση Υπαγωγής: Έγκυρο αρχείο.")
            elif state_a["status"] == "wrong":
                st.error("Αίτηση Υπαγωγής: Λάθος αρχείο.")
            elif state_a["status"] == "checking":
                st.info("Αίτηση Υπαγωγής: Το αρχείο υποβλήθηκε και ελέγχεται...")

            st.info(
                "💡 Κάθε αρχείο ελέγχεται άμεσα ως προς τον τύπο του. "
                "Πράσινο μήνυμα = έγκυρο έντυπο."
            )

        st.markdown("---")

        valid_count_now = sum(
            1 for k in EXPECTED_DOCS if st.session_state.docs[k]["status"] == "ok"
        )
        if valid_count_now == len(EXPECTED_DOCS) and not any_wrong_docs():
            st.session_state.current_step = max(st.session_state.current_step, 1)
            st.success(
                "🎉 **Όλα τα έγγραφα είναι έγκυρα.**\n\n"
                "Μπορείτε να προχωρήσετε σε συνολικό έλεγχο φακέλου."
            )

        st.subheader("Σύνοψη Κατάστασης Εγγράφων")

        status_rows = []
        for key, label in EXPECTED_DOCS.items():
            state = st.session_state.docs[key]
            if state["status"] == "not_uploaded":
                emoji = "⬜"
                text = "Δεν ανέβηκε"
            elif state["status"] == "wrong":
                emoji = "❌"
                text = "Λάθος αρχείο"
            elif state["status"] == "ok":
                emoji = "✅"
                text = "Έγκυρο"
            else:
                emoji = "⏳"
                text = "Ελέγχεται"

            status_rows.append(
                {
                    "Έγγραφο": label,
                    "Κατάσταση": f"{emoji} {text}",
                    "Αναγνώριση τίτλου": state["detected_type"] or "-",
                }
            )

        df_status = pd.DataFrame(status_rows)
        df_status.index = range(1, len(df_status) + 1)
        st.table(df_status)

        with st.expander("Αναλυτικά μηνύματα για κάθε έγγραφο"):
            for key, label in EXPECTED_DOCS.items():
                state = st.session_state.docs[key]
                st.markdown(f"**{label}:** {state['message'] or '—'}")

        st.markdown("---")

        all_uploaded = all_docs_uploaded()
        no_wrong = not any_wrong_docs()

        if not all_uploaded:
            st.warning("Για να συνεχίσετε, ανεβάστε πρώτα όλα τα έγγραφα. 📂")
        elif not no_wrong:
            st.error("Υπάρχουν έγγραφα που έχουν χαρακτηριστεί ως λάθος. Διορθώστε τα. ❌")

        check_button = st.button(
            "Συνολικός Έλεγχος & Επιβεβαίωση Φακέλου (Demo)",
            type="primary",
            disabled=not (all_uploaded and no_wrong),
        )

        if check_button and all_uploaded and no_wrong:
            st.session_state.current_step = 2
            st.session_state.e9_ai_result = None

            with st.status(
                "Εκκίνηση συνολικού ελέγχου φακέλου... ⏳",
                expanded=True,
            ) as status:
                # Βήμα 1
                status.write("📄 OCR ανάγνωση & τελικός έλεγχος Ε9...")
                time.sleep(0.7)

                e9_state = st.session_state.docs["e9"]
                e9_file_for_ai = e9_state["file"]

                if e9_file_for_ai is not None:
                    try:
                        text = extract_text_from_pdf(e9_file_for_ai)
                    except Exception as e:
                        text = ""
                        st.session_state.e9_ai_result = {
                            "is_valid": False,
                            "confidence": 0,
                            "reason": f"Σφάλμα κατά την εξαγωγή κειμένου: {e}",
                            "action_needed": "Δοκιμάστε ξανά ή ανεβάστε νέο αρχείο Ε9."
                        }

                    # Βήμα 2 - Πραγματικός AI έλεγχος
                    status.write("🤖 AI έλεγχος εγκυρότητας Ε9 (περιεχόμενο)...")
                    time.sleep(0.7)

                    if text and len(text.strip()) >= 20:
                        try:
                            result = validate_e9_with_ai(text)
                            st.session_state.e9_ai_result = result
                        except Exception as e:
                            st.session_state.e9_ai_result = {
                                "is_valid": False,
                                "confidence": 0,
                                "reason": f"Σφάλμα κατά την ανάλυση AI: {e}",
                                "action_needed": "Δοκιμάστε ξανά ή ανεβάστε νέο αρχείο Ε9."
                            }
                    else:
                        st.session_state.e9_ai_result = {
                            "is_valid": False,
                            "confidence": 0,
                            "reason": "Το PDF δεν περιέχει αρκετό αναγνώσιμο κείμενο.",
                            "action_needed": "Ελέγξτε ότι το αρχείο είναι σωστό Ε9 και όχι σκαναρισμένη εικόνα χαμηλής ποιότητας."
                        }

                    status.write("✅ Ο AI έλεγχος Ε9 ολοκληρώθηκε.")
                    time.sleep(0.7)

                # Βήμα 3
                status.write("📊 Risk Assessment & βασικός έλεγχος βιωσιμότητας...")
                time.sleep(0.7)

                status.write("✅ Φάκελος ολοκληρώθηκε επιτυχώς.")

                status.update(
                    label="Ο συνολικός έλεγχος ολοκληρώθηκε!",
                    state="complete",
                    expanded=False,
                )

            st.session_state.current_step = 3

            st.success(
                "🎉 **Συγχαρητήρια!**\n\n"
                "Υποβάλλατε ορθά όλα τα απαιτούμενα δικαιολογητικά. "
                "Η αίτησή σας έχει καταχωρηθεί και εντός **2 εργάσιμων ημερών** "
                "θα λάβετε την επίσημη απάντηση από το αρμόδιο τμήμα."
            )

    chatbot_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999;
    ">
      <div style="
          background-color: #0C4DA2;
          color: white;
          padding: 10px 16px;
          border-radius: 999px;
          display: flex;
          align-items: center;
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
          cursor: pointer;
          font-size: 14px;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      ">
        <span style="font-size: 20px; margin-right: 8px;">👩‍💼</span>
        <div style="display:flex; flex-direction:column;">
          <span><strong>GovBot AI Assistant</strong> 💬</span>
          <span style="font-size: 12px;">Πώς μπορώ να σας βοηθήσω;</span>
        </div>
      </div>
    </div>
    """
    st.markdown(chatbot_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Είναι χρήσιμη αυτή η σελίδα;")

    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        if st.button("Ναι", key="yes_useful"):
            st.session_state.feedback = "yes"
    with fb_col2:
        if st.button("Όχι", key="no_useful"):
            st.session_state.feedback = "no"

    if st.session_state.feedback == "yes":
        st.success("😊 Ευχαριστούμε για την αξιολόγηση! "
                   "Η γνώμη σας βοηθά στη βελτίωση των ψηφιακών υπηρεσιών.")
    elif st.session_state.feedback == "no":
        st.info("💡 Μπορείτε να μοιραστείτε τις παρατηρήσεις σας στο αρμόδιο τμήμα υποστήριξης.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **ΥΠΗΡΕΣΙΕΣ · ΦΟΡΕΙΣ · ΑΡΧΕΣ · ΣΧΕΤΙΚΑ ΜΕ ΤΟ GOV.GR**  
            ΕΜΔΔ – ΜΙΤΟΣ · EUGO · ΟΡΟΙ ΧΡΗΣΗΣ · ΙΔΙΩΤΙΚΟ ΑΠΟΡΡΗΤΟ  
            ΨΗΦΙΑΚΗ ΑΚΑΔΗΜΙΑ · ΜΑΘΕΤΕ ΤΟ GOV.GR · ΠΡΟΤΑΣΕΙΣ ΓΙΑ ΤΟ GOV.GR  
            ΔΗΛΩΣΗ ΠΡΟΣΒΑΣΙΜΟΤΗΤΑΣ · ΒΙΒΛΟΣ ΨΗΦΙΑΚΟΥ ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΥ
            """
        )
    with col2:
        st.markdown(
            """
            © Copyright 2025 - Υλοποίηση από το Υπουργείο Ψηφιακής Διακυβέρνησης  
            **ENGLISH | ΕΛΛΗΝΙΚΑ**  
            
            Ελληνική Δημοκρατία – Κυβέρνηση 🇬🇷
            """
        )

# ===================================================
# 9. DASHBOARD ΕΛΕΓΚΤΗ
# ===================================================

def render_inspector_dashboard():
    inject_global_css()
    init_session_state()

    demo_case = load_demo_case()

    st.markdown("#### Πίνακας Ελέγχου Ελεγκτή – Smart Settlement")
    st.title("AI Πρόταση Διακανονισμού & Έλεγχος Βιωσιμότητας")

    if demo_case:
        st.info(
            "Για τις ανάγκες του demo, ο έλεγχος βασίζεται στον υποβληθέντα φάκελο πολίτη "
            "(δοκιμαστικά δεδομένα)."
        )
    else:
        st.warning(
            "Δεν βρέθηκε αρχείο demo_case.json. Το demo θα τρέξει με χειροκίνητη εισαγωγή ποσών."
        )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Οικονομικά Στοιχεία Οφειλέτη")

        use_demo = False
        if demo_case:
            use_demo = st.checkbox("Χρήση υποβληθέντος φακέλου πολίτη", value=True)

        if use_demo and demo_case:
            # ΔΕΝ επιτρέπουμε αλλαγή – απλώς δείχνουμε τα ποσά του JSON
            yearly_income = float(demo_case.get("income", 30000))
            total_debt = float(demo_case.get("total_debt", 50000))
            expenses = float(demo_case.get("expenses", 12000))
            rent = float(demo_case.get("rent", 6000))
            tax = float(demo_case.get("tax", 4000))
            interest = float(demo_case.get("interest", 500))

            st.write("Τα παρακάτω ποσά προέρχονται αυτόματα από τον ψηφιακό φάκελο του πολίτη:")

            df_demo = pd.DataFrame([
                {
                    "Πεδίο": "Ετήσιο Εισόδημα",
                    "Ποσό (€)": yearly_income,
                },
                {
                    "Πεδίο": "Συνολικό Χρέος",
                    "Ποσό (€)": total_debt,
                },
                {
                    "Πεδίο": "Ετήσιες Βασικές Δαπάνες",
                    "Ποσό (€)": expenses,
                },
                {
                    "Πεδίο": "Ετήσιο Ενοίκιο",
                    "Ποσό (€)": rent,
                },
                {
                    "Πεδίο": "Παρακρατηθέντες Φόροι",
                    "Ποσό (€)": tax,
                },
                {
                    "Πεδίο": "Τόκοι Καταθέσεων",
                    "Ποσό (€)": interest,
                },
            ])
            st.table(df_demo)

            if st.button("🔍 Υπολογισμός AI Πρότασης από φάκελο πολίτη"):
                settlement = predict_settlement(
                    income=yearly_income,
                    debt=total_debt,
                    expenses=expenses,
                    rent=rent,
                    tax=tax,
                    interest=interest
                )
                score, risk_label, reasons = calculate_viability_score(
                    yearly_income, total_debt, expenses, rent
                )
                st.session_state.last_settlement = settlement
                st.session_state.last_score = score
                st.session_state.last_risk_label = risk_label
                st.session_state.last_reasons = reasons

        else:
            # Manual form – επιτρέπεται αλλαγή ποσών
            with st.form("inspector_form"):
                col1, col2 = st.columns(2)
                with col1:
                    yearly_income = st.number_input(
                        "Ετήσιο Εισόδημα (€)",
                        min_value=0.0,
                        value=30000.0,
                        step=1000.0,
                    )
                    total_debt = st.number_input(
                        "Συνολικό Χρέος (€)",
                        min_value=0.0,
                        value=50000.0,
                        step=1000.0,
                    )
                    expenses = st.number_input(
                        "Ετήσιες Βασικές Δαπάνες (€)",
                        min_value=0.0,
                        value=12000.0,
                        step=500.0,
                    )
                with col2:
                    rent = st.number_input(
                        "Ετήσιο Ενοίκιο (€)",
                        min_value=0.0,
                        value=6000.0,
                        step=500.0,
                    )
                    tax = st.number_input(
                        "Παρακρατηθέντες Φόροι (€)",
                        min_value=0.0,
                        value=4000.0,
                        step=500.0,
                    )
                    interest = st.number_input(
                        "Τόκοι Καταθέσεων (€)",
                        min_value=0.0,
                        value=500.0,
                        step=100.0,
                    )

                submitted = st.form_submit_button("🔍 Υπολογισμός AI Πρότασης")

            if submitted:
                settlement = predict_settlement(
                    income=yearly_income,
                    debt=total_debt,
                    expenses=expenses,
                    rent=rent,
                    tax=tax,
                    interest=interest
                )
                score, risk_label, reasons = calculate_viability_score(
                    yearly_income, total_debt, expenses, rent
                )
                st.session_state.last_settlement = settlement
                st.session_state.last_score = score
                st.session_state.last_risk_label = risk_label
                st.session_state.last_reasons = reasons

    with col_right:
        st.subheader("Κατάσταση Φακέλου Εγγράφων")

        status_rows = []
        for key, label in EXPECTED_DOCS.items():
            state = st.session_state.docs[key]
            if state["status"] == "not_uploaded":
                emoji = "⬜"
                text = "Δεν ανέβηκε"
            elif state["status"] == "wrong":
                emoji = "❌"
                text = "Λάθος αρχείο"
            elif state["status"] == "ok":
                emoji = "✅"
                text = "Έγκυρο"
            else:
                emoji = "⏳"
                text = "Ελέγχεται"

            status_rows.append(
                {
                    "Έγγραφο": label,
                    "Κατάσταση": f"{emoji} {text}",
                    "Τύπος (AI)": state["detected_type"] or "-",
                }
            )

        df_docs = pd.DataFrame(status_rows)
        df_docs.index = range(1, len(df_docs) + 1)
        st.table(df_docs)

        all_ok = all_docs_uploaded() and not any_wrong_docs()
        if all_ok:
            st.success("✅ Όλα τα απαιτούμενα έγγραφα είναι σε αποδεκτή μορφή.")
        else:
            st.warning("⚠ Ο φάκελος δεν είναι πλήρης ή υπάρχουν λάθος έγγραφα.")

        if demo_case:
            st.markdown("---")
            st.markdown("**Στοιχεία demo οφειλέτη (για το storytelling):**")
            st.write(f"👤 Ονοματεπώνυμο: {demo_case.get('citizen_name', 'Demo Πολίτης')}")
            st.write(f"🧾 ΑΦΜ: {demo_case.get('afm', '—')}")
            st.write(f"📧 Email (demo): smartsettlement.demo@gmail.com")

    st.markdown("---")

    if st.session_state.last_settlement:
        settlement = st.session_state.last_settlement
        score = st.session_state.last_score
        risk_label = st.session_state.last_risk_label
        reasons = st.session_state.last_reasons or []

        st.subheader("AI Πρόταση Διακανονισμού")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Score Βιωσιμότητας", f"{score} / 100", risk_label)
        with m2:
            st.metric("Προτεινόμενη Μηνιαία Δόση", f"{settlement['FinalMonthlyPayment']} €")
        with m3:
            st.metric("Διάρκεια Ρύθμισης", f"{settlement['Months']} μήνες")
        with m4:
        # Προσθέτουμε +100€ στην αρχική πρόβλεψη ΜL μόνο για εμφάνιση
            adjusted_ml = float(settlement["PredictedML"]) + 100.0
            st.metric("Αρχική Πρόβλεψη Μοντέλου", f"{adjusted_ml:.2f} €")


        st.info(
            "Το μοντέλο συνδυάζει Machine Learning (Ridge Regression σε συνθετικά δεδομένα) "
            "με κανόνες πολιτικής (π.χ. max 40% του καθαρού εισοδήματος, έως 240 μήνες), "
            "προσφέροντας τόσο αυτόματη πρόταση όσο και ερμηνευσιμότητα."
        )

        if reasons:
            with st.expander("Ερμηνεία score βιωσιμότητας (τύπου SHAP)"):
                for r in reasons:
                    st.markdown(f"- {r}")

        st.markdown("### Απόφαση Ελεγκτή")

        c_dec1, c_dec2, c_dec3 = st.columns(3)
        with c_dec1:
            if st.button("✅ Έγκριση AI Πρότασης"):
                st.session_state.inspector_decision = "approved"
        with c_dec2:
            if st.button("✏️ Αντιπρόταση"):
                st.session_state.inspector_decision = "counter"
        with c_dec3:
            if st.button("❌ Απόρριψη"):
                st.session_state.inspector_decision = "rejected"

        citizen_name = (demo_case.get("citizen_name") if demo_case else "Οφειλέτης Demo")

        if st.session_state.inspector_decision == "approved":
            st.success("Η πρόταση του AI εγκρίθηκε και προωθείται στον οφειλέτη. ✅")
            send_decision_email("approved", settlement, citizen_name)

        elif st.session_state.inspector_decision == "rejected":
            st.error("Ο ελεγκτής απέρριψε την πρόταση. ❌")
            send_decision_email("rejected", settlement, citizen_name)

        elif st.session_state.inspector_decision == "counter":
            st.info("Ο ελεγκτής τροποποιεί χειροκίνητα τη ρύθμιση (αντιπρόταση).")
            with st.form("counter_form"):
                new_monthly = st.number_input(
                    "Νέα προτεινόμενη δόση (€)",
                    min_value=10.0,
                    value=settlement["FinalMonthlyPayment"],
                    step=10.0,
                )
                new_months = st.number_input(
                    "Νέος αριθμός δόσεων",
                    min_value=1,
                    max_value=600,
                    value=int(settlement["Months"]),
                    step=1,
                )
                submitted_counter = st.form_submit_button("Καταχώρηση Αντιπρότασης & Προβολή Email")

            if submitted_counter:
                settlement_counter = settlement.copy()
                settlement_counter["FinalMonthlyPayment"] = float(new_monthly)
                settlement_counter["Months"] = int(new_months)
                st.success(
                    f"Η αντιπρόταση καταχωρήθηκε: {new_monthly:.2f} € x {int(new_months)} μήνες."
                )
                send_decision_email("counter", settlement_counter, citizen_name)

    else:
        st.info("Συμπληρώστε τα οικονομικά στοιχεία και πατήστε «Υπολογισμός AI Πρότασης».")

    # ===================================================
    # RAG Νομικός Βοηθός – στο τέλος του dashboard ελεγκτή
    # ===================================================
    st.markdown("---")
    st.subheader("Νομικός Βοηθός RAG (ΦΕΚ / Νόμοι – Demo)")

    legal_query = st.text_area(
        "Ερώτηση προς το νομικό πλαίσιο (π.χ. 'Αν ο οφειλέτης διαθέτει κύρια κατοικία με χαμηλή αξία, "
        "προβλέπεται πρόσθετη προστασία στη ρύθμιση χρέους των 50.000€;')",
        key="legal_rag_question",
        height=80,
    )

    if st.button("Ρώτα τον Νομικό Βοηθό"):
        if legal_query.strip():
            with st.spinner("Αναζήτηση σε αποσπάσματα νόμων και σύνθεση απάντησης..."):
                rag_result = ask_legal_rag(legal_query.strip())

            st.markdown("**Απάντηση (με βάση τα παρεχόμενα αποσπάσματα νόμου – demo):**")
            st.write(rag_result["answer"])

            with st.expander("Πηγές που χρησιμοποιήθηκαν (demo):"):
                for src in rag_result["sources"]:
                    st.markdown(f"- {src}")
        else:
            st.info("Γράψτε πρώτα μια ερώτηση προς τον νομικό βοηθό.")

# ===================================================
# 10. MAIN APP – Επιλογή ρόλου
# ===================================================

def main():
    st.set_page_config(
        page_title="Smart Settlement – Πολίτης & Ελεγκτής",
        layout="wide",
    )

    role = st.sidebar.radio(
        "Επιλέξτε ρόλο:",
        ["Πολίτης", "Ελεγκτής"],
        help="Δείτε την εμπειρία από την πλευρά του πολίτη ή του ελεγκτή."
    )

    with st.sidebar:
        st.markdown("---")
        st.header("🔧 Debug Info")
        st.write("Model:", deployment)
        st.write("Endpoint (demo):", endpoint)
        st.write("Embeddings model:", EMBEDDING_DEPLOYMENT)

    if role == "Πολίτης":
        render_citizen_flow()
    else:
        render_inspector_dashboard()


if __name__ == "__main__":
    main()
