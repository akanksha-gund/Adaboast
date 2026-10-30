import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = 'adaboost.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# HTML Template with Tailwind CSS and responsive design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Admission / Evaluation Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-slate-100 min-h-screen py-10 px-4">
    <div class="max-w-4xl mx-auto">
        <!-- Header -->
        <div class="text-center mb-10">
            <h1 class="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400 mb-3">
                Student Evaluation Portal
            </h1>
            <p class="text-slate-400 text-lg">
                Powered by AdaBoost Machine Learning Model
            </p>
        </div>

        <!-- Main Card -->
        <div class="bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl p-8">
            {% if prediction_text %}
            <div class="mb-8 p-6 rounded-xl border {% if 'Accepted' in prediction_text or 'Positive' in prediction_text or '1' in prediction_text %} bg-emerald-950/40 border-emerald-500/50 text-emerald-300 {% else %} bg-indigo-950/40 border-indigo-500/50 text-indigo-300 {% endif %} text-center shadow-lg transition-all duration-300">
                <h3 class="text-sm font-semibold uppercase tracking-wider mb-1 opacity-80">Prediction Result</h3>
                <p class="text-2xl font-bold">{{ prediction_text }}</p>
                <a href="/" class="inline-block mt-4 text-xs font-medium uppercase tracking-widest bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 rounded-lg border border-slate-700 transition">Evaluate Another Candidate</a>
            </div>
            {% endif %}

            <form action="/predict" method="POST" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    
                    <!-- Demographic & Basic Info -->
                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Age</label>
                        <input type="number" name="age" required value="{{ form_data.get('age', '18') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Gender</label>
                        <select name="gender" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                            <option value="0">Female / Option 0</option>
                            <option value="1">Male / Option 1</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">State / Region Code</label>
                        <input type="number" name="state" required value="{{ form_data.get('state', '1') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Family Income ($)</label>
                        <input type="number" name="family_income" step="1000" required value="{{ form_data.get('family_income', '65000') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <!-- Academic Performance -->
                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">High School GPA</label>
                        <input type="number" step="0.01" name="high_school_gpa" required value="{{ form_data.get('high_school_gpa', '3.5') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">SAT Score</label>
                        <input type="number" name="sat_score" required value="{{ form_data.get('sat_score', '1250') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">ACT Score</label>
                        <input type="number" name="act_score" required value="{{ form_data.get('act_score', '28') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Attendance Rate (%)</label>
                        <input type="number" step="0.1" name="attendance_rate" required value="{{ form_data.get('attendance_rate', '95.0') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">AP Courses Count</label>
                        <input type="number" name="ap_courses" required value="{{ form_data.get('ap_courses', '3') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <!-- Extracurriculars & Activities -->
                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Extracurricular Count</label>
                        <input type="number" name="extracurricular_count" required value="{{ form_data.get('extracurricular_count', '2') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Volunteer Hours</label>
                        <input type="number" name="volunteer_hours" required value="{{ form_data.get('volunteer_hours', '50') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Leadership Positions</label>
                        <input type="number" name="leadership_positions" required value="{{ form_data.get('leadership_positions', '1') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Coding Projects</label>
                        <input type="number" name="coding_projects" required value="{{ form_data.get('coding_projects', '2') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Social Media Hours (Weekly)</label>
                        <input type="number" step="0.5" name="social_media_hours" required value="{{ form_data.get('social_media_hours', '10.0') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Online Certifications</label>
                        <input type="number" name="online_certifications" required value="{{ form_data.get('online_certifications', '2') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <!-- Scores & Ratings -->
                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Essay Score (1-10)</label>
                        <input type="number" step="0.1" name="essay_score" required value="{{ form_data.get('essay_score', '8.0') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Recommendation Score (1-10)</label>
                        <input type="number" step="0.1" name="recommendation_score" required value="{{ form_data.get('recommendation_score', '8.5') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-2">Interview Score (1-10)</label>
                        <input type="number" step="0.1" name="interview_score" required value="{{ form_data.get('interview_score', '9.0') }}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>

                </div>

                <div class="mt-8">
                    <button type="submit" class="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white font-bold text-lg shadow-lg shadow-indigo-500/20 transition transform hover:-translate-y-0.5">
                        Run Prediction
                    </button>
                </div>
            </form>
        </div>
        
        <footer class="text-center mt-8 text-slate-500 text-sm">
            Deployable on Render via Gunicorn.
        </footer>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: Model not loaded properly.", form_data=request.form)
    
    try:
        # Extract features in exact order specified by your pickle structure
        features = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['state']),
            float(request.form['family_income']),
            float(request.form['high_school_gpa']),
            float(request.form['sat_score']),
            float(request.form['act_score']),
            float(request.form['attendance_rate']),
            float(request.form['ap_courses']),
            float(request.form['extracurricular_count']),
            float(request.form['volunteer_hours']),
            float(request.form['leadership_positions']),
            float(request.form['coding_projects']),
            float(request.form['social_media_hours']),
            float(request.form['online_certifications']),
            float(request.form['essay_score']),
            float(request.form['recommendation_score']),
            float(request.form['interview_score'])
        ]
        
        final_features = np.array([features])
        prediction = model.predict(final_features)
        
        result_str = f"Model Prediction: {prediction[0]}"
        return render_template_string(HTML_TEMPLATE, prediction_text=result_str, form_data=request.form)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error during prediction: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)