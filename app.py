"""
Admission Predictor – Flask app serving adaboost.pkl
Deploy on Render:  Build:  pip install -r requirements.txt
                   Start:  gunicorn app:app
"""
import os
import warnings

import joblib
import pandas as pd
from flask import Flask, render_template_string, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaboost.pkl")
model = joblib.load(MODEL_PATH)

# The model was trained on 19 columns, but only the 11 below affect its output
# (the other 8 have zero feature importance), so they are filled with 0.
MODEL_COLUMNS = list(model.feature_names_in_)

# key, label, min, max, step, default, hint
FIELDS = {
    "Academics": [
        ("high_school_gpa", "High school GPA", 0, 4, 0.01, 3.5, "Out of 4.0"),
        ("sat_score", "SAT score", 400, 1600, 1, 1200, "Out of 1600"),
        ("ap_courses", "AP courses taken", 0, 15, 1, 3, "Count"),
    ],
    "Activities": [
        ("extracurricular_count", "Extracurriculars", 0, 15, 1, 3, "Clubs, sports, arts"),
        ("volunteer_hours", "Volunteer hours", 0, 500, 1, 80, "Total hours"),
        ("leadership_positions", "Leadership roles", 0, 10, 1, 1, "Captain, president, etc."),
        ("coding_projects", "Coding projects", 0, 20, 1, 2, "Completed projects"),
        ("online_certifications", "Online certifications", 0, 20, 1, 2, "Completed courses"),
    ],
    "Evaluations": [
        ("essay_score", "Essay score", 0, 100, 1, 70, "Out of 100"),
        ("recommendation_score", "Recommendation score", 0, 100, 1, 70, "Out of 100"),
        ("interview_score", "Interview score", 0, 100, 1, 70, "Out of 100"),
    ],
}
ALL_FIELDS = [f for group in FIELDS.values() for f in group]

# ----------------------------------------------------------------------------
# STYLE (Fresh Pink Blur & Glassmorphism Theme)
# ----------------------------------------------------------------------------
STYLE = """
:root{
  --pink-deep: #831843;
  --pink-main: #db2777;
  --pink-light: #fce7f3;
  --glass-bg: rgba(255, 255, 255, 0.85);
  --text: #1e293b;
  --muted: #64748b;
  --radius: 20px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  color:var(--text);
  background-color: #0f172a;
  min-height:100vh;
  line-height:1.5;
  position: relative;
  overflow-x: hidden;
}

/* Fixed Background Image with Dark Overlay */
.bg-overlay {
  position: fixed;
  inset: 0;
  z-index: 0;
  background-image: url('https://images.unsplash.com/photo-1523050854058-8df90110c9f1?q=80&w=2070&auto=format&fit=crop');
  background-size: cover;
  background-position: center;
}
.bg-overlay::after {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(15, 23, 42, 0.8);
  backdrop-filter: blur(3px);
}

/* Pink Blur Glow Elements */
.glow-1 {
  position: absolute;
  top: -100px;
  left: 10%;
  width: 450px;
  height: 450px;
  background: rgba(236, 72, 153, 0.35);
  border-radius: 50%;
  filter: blur(100px);
  z-index: 0;
  pointer-events: none;
}
.glow-2 {
  position: absolute;
  bottom: -100px;
  right: 10%;
  width: 450px;
  height: 450px;
  background: rgba(244, 63, 94, 0.3);
  border-radius: 50%;
  filter: blur(100px);
  z-index: 0;
  pointer-events: none;
}

h1,h2,h3{font-family:'Bricolage Grotesque','DM Sans',system-ui,sans-serif;letter-spacing:-.02em}

.shell{
  max-width:1180px;margin:0 auto;padding:32px 20px 56px;
  display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:28px;align-items:start;
  position: relative;
  z-index: 10;
}

/* ---------- hero panel ---------- */
.hero{
  position:sticky;top:24px;
  background: linear-gradient(135deg, rgba(131, 24, 67, 0.85) 0%, rgba(30, 41, 59, 0.9) 100%);
  backdrop-filter: blur(12px);
  color:#fff;border-radius:28px;overflow:hidden;
  border: 1px solid rgba(244, 63, 94, 0.2);
  box-shadow:0 25px 50px -12px rgba(219, 39, 119, 0.25);
}
.hero-art{display:block;width:100%;height:auto}
.hero-body{padding:8px 32px 34px}
.hero h1{font-size:2.35rem;line-height:1.08;font-weight:800;margin-bottom:12px; color: #fff;}
.hero p{color:#fbcfe8;font-size:1rem;max-width:36ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.chip{
  font-size:.82rem;padding:6px 12px;border-radius:999px;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);color:#fff;
}

/* ---------- form card ---------- */
.card{
  background: var(--glass-bg);
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  border:1px solid rgba(244, 63, 94, 0.3);
  border-radius:28px;
  padding:30px 30px 34px;
  box-shadow:0 25px 50px -12px rgba(0, 0, 0, 0.3);
}
fieldset{border:0;margin-bottom:26px}
legend{
  font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.2rem;
  color:var(--pink-deep);padding-bottom:10px;margin-bottom:6px;
  border-bottom:2px solid #fce7f3;width:100%;
}
.row{display:grid;grid-template-columns:1fr 1fr;gap:6px 22px}
.field{padding:12px 0 6px}
.field label{font-weight:600;font-size:.95rem;color:#334155;}
.hint{font-size:.78rem;color:var(--muted);margin-top:6px}

input[type=number]{
  width:100%;margin-top:8px;padding:12px 14px;
  font-family:'Bricolage Grotesque','DM Sans',sans-serif;font-weight:700;font-size:1.1rem;color:var(--text);
  background:#ffffff;border:2px solid #fbcfe8;border-radius:14px;outline:none;
  transition:border-color .15s ease, box-shadow .15s ease;
  -moz-appearance:textfield;appearance:textfield;
}
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{opacity:1}
input[type=number]:hover{border-color:#f43f5e}
input[type=number]:focus{border-color:var(--pink-main);box-shadow:0 0 0 4px rgba(219, 39, 119, 0.15)}
input[type=number]:invalid{border-color:#e11d48}

.btn{
  width:100%;border:0;cursor:pointer;
  font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.1rem;
  padding:16px 22px;border-radius:14px;color:#fff;
  background:linear-gradient(135deg, #db2777 0%, #e11d48 100%);
  box-shadow:0 10px 20px -5px rgba(219, 39, 119, 0.5);
  transition:transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
.btn:hover{transform:translateY(-1px);filter: brightness(1.05);}
.btn:active{transform:translateY(1px)}
.btn:focus-visible{outline:3px solid var(--pink-main);outline-offset:3px}

/* ---------- result ---------- */
.result{
  display:grid;grid-template-columns:auto 1fr;gap:24px;align-items:center;
  margin-bottom:28px;padding:22px 24px;border-radius:22px;
  background:#ffffff;border:2px solid var(--tone);
  animation:pop .45s cubic-bezier(.2,.9,.3,1.2);
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);
}
.result.good{--tone:#10b981}
.result.bad{--tone:#f43f5e}
.ring{
  --pct:0;width:132px;height:132px;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--tone) calc(var(--pct)*1%), #f1f5f9 0);
}
.ring span{
  width:100px;height:100px;border-radius:50%;background:#fff;display:grid;place-items:center;
  font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.7rem;color:var(--text);
}
.result h2{font-size:1.5rem;color:var(--text);margin-bottom:4px}
.result p{color:var(--muted);font-size:.95rem}
.error{background:#fff1f2;border:1px solid #f43f5e;color:#881337;padding:12px 16px;border-radius:12px;margin-bottom:20px}
@keyframes pop{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:none}}

.foot{grid-column:1/-1;text-align:center;color:#cbd5e1;font-size:.85rem;margin-top:6px}

/* ---------- responsive ---------- */
@media (max-width:900px){
  .shell{grid-template-columns:1fr}
  .hero{position:relative;top:0}
  .hero h1{font-size:1.9rem}
}
@media (max-width:560px){
  .row{grid-template-columns:1fr}
  .card{padding:22px 18px 26px}
  .result{grid-template-columns:1fr;justify-items:center;text-align:center}
}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
}
"""

# Inline SVG illustration
HERO_SVG = """
<svg class="hero-art" viewBox="0 0 600 380" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Illustration of a university hall at dusk with a graduation cap">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#831843"/><stop offset=".65" stop-color="#be185d"/>
      <stop offset="1" stop-color="#f43f5e"/>
    </linearGradient>
    <linearGradient id="hall" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fff1f2"/><stop offset="1" stop-color="#ffe4e6"/>
    </linearGradient>
  </defs>
  <rect width="600" height="380" fill="url(#sky)"/>
  <g fill="#fff" opacity=".85">
    <circle cx="70" cy="50" r="2"/><circle cx="150" cy="28" r="1.6"/><circle cx="260" cy="60" r="1.8"/>
    <circle cx="470" cy="38" r="2"/><circle cx="540" cy="86" r="1.5"/><circle cx="395" cy="70" r="1.4"/>
  </g>
  <circle cx="500" cy="120" r="34" fill="#fbcfe8" opacity=".95"/>
  <path d="M0 300 Q150 250 300 285 T600 270 V380 H0Z" fill="#4c0519"/>
  <!-- hall -->
  <g>
    <rect x="130" y="205" width="340" height="110" fill="url(#hall)"/>
    <polygon points="115,205 300,140 485,205" fill="#fff1f2"/>
    <polygon points="115,205 300,140 485,205 470,205 300,150 130,205" fill="#fecdd3"/>
    <rect x="270" y="95" width="60" height="55" fill="#fff1f2"/>
    <polygon points="262,95 300,62 338,95" fill="#831843"/>
    <circle cx="300" cy="120" r="14" fill="#831843"/>
    <circle cx="300" cy="120" r="11" fill="#fff1f2"/>
    <path d="M300 120V112M300 120L306 123" stroke="#831843" stroke-width="2" stroke-linecap="round"/>
    <g fill="#f43f5e" opacity="0.3">
      <rect x="150" y="212" width="14" height="103"/><rect x="190" y="212" width="14" height="103"/>
      <rect x="230" y="212" width="14" height="103"/><rect x="356" y="212" width="14" height="103"/>
      <rect x="396" y="212" width="14" height="103"/><rect x="436" y="212" width="14" height="103"/>
    </g>
    <path d="M270 315V262a30 30 0 0 1 60 0V315Z" fill="#831843"/>
  </g>
  <rect x="0" y="312" width="600" height="68" fill="#380414"/>
  <path d="M300 315 L250 380 H350Z" fill="#f43f5e" opacity=".35"/>
  <!-- graduation cap -->
  <g transform="translate(78 120) rotate(-12)">
    <polygon points="0,30 60,5 120,30 60,55" fill="#f43f5e"/>
    <path d="M25 42v22c0 8 70 8 70 0V42L60 55Z" fill="#be185d"/>
    <path d="M120 30v34" stroke="#fbcfe8" stroke-width="3" stroke-linecap="round"/>
    <circle cx="120" cy="68" r="5" fill="#fbcfe8"/>
  </g>
</svg>
"""

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Admission Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>{{ style|safe }}</style>
</head>
<body>
  <div class="bg-overlay"></div>
  <div class="glow-1"></div>
  <div class="glow-2"></div>

  <main class="shell">
    <aside class="hero">
      {{ art|safe }}
      <div class="hero-body">
        <h1>Check your admission chances</h1>
        <p>Enter your grades, activities and interview results. An AdaBoost model trained on past applicants estimates how likely an admission is.</p>
        <div class="chips">
          <span class="chip">AdaBoost classifier</span>
          <span class="chip">11 profile signals</span>
          <span class="chip">Instant result</span>
        </div>
      </div>
    </aside>

    <section class="card">
      {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}

      {% if result %}
      <div class="result {{ 'good' if result.admitted else 'bad' }}" role="status">
        <div class="ring" style="--pct: {{ result.pct }}"><span>{{ result.pct }}%</span></div>
        <div>
          <h2>{{ 'Likely to be admitted' if result.admitted else 'Less likely to be admitted' }}</h2>
          <p>The model gives this profile a {{ result.pct }}% chance of admission. Change any value and predict again to compare.</p>
        </div>
      </div>
      {% endif %}

      <form method="post" action="/">
        {% for group, fields in groups.items() %}
        <fieldset>
          <legend>{{ group }}</legend>
          <div class="row">
            {% for key, label, lo, hi, step, default, hint in fields %}
            <div class="field">
              <label for="{{ key }}">{{ label }}</label>
              <input type="number" id="{{ key }}" name="{{ key }}" inputmode="decimal"
                     min="{{ lo }}" max="{{ hi }}" step="{{ step }}" value="{{ values[key] }}" required>
              <div class="hint">{{ hint }} &middot; range {{ lo }} to {{ hi }}</div>
            </div>
            {% endfor %}
          </div>
        </fieldset>
        {% endfor %}
        <button class="btn" type="submit">Predict admission</button>
      </form>
    </section>

    <p class="foot">Predictions are statistical estimates and not a guarantee of any real admission decision.</p>
  </main>

</body>
</html>
"""


def default_values():
    return {key: default for key, _, _, _, _, default, _ in ALL_FIELDS}


def parse_form(form):
    """Validate and clamp submitted values. Returns (values, error)."""
    values = default_values()
    for key, label, lo, hi, step, default, _ in ALL_FIELDS:
        raw = form.get(key, "")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return values, f"'{label}' needs a number."
        if not lo <= num <= hi:
            return values, f"'{label}' must be between {lo} and {hi}."
        values[key] = int(num) if float(step).is_integer() else round(num, 2)
    return values, None


def predict(values):
    row = {col: 0 for col in MODEL_COLUMNS}
    row.update(values)
    frame = pd.DataFrame([row])[MODEL_COLUMNS]
    prob = float(model.predict_proba(frame)[0][1])
    return {"admitted": prob >= 0.5, "pct": round(prob * 100)}


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    values, result, error = default_values(), None, None
    if request.method == "POST":
        values, error = parse_form(request.form)
        if error is None:
            result = predict(values)
    return render_template_string(
        PAGE, style=STYLE, art=HERO_SVG, groups=FIELDS,
        values=values, result=result, error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)