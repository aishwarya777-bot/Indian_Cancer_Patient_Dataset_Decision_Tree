import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the decision tree model trained for Age, Gender, State, City, Cancer Type, Stage, Treatment Type, and Survival Months
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'decision_tree.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Integrated HTML Template using Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Prognosis Tracker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-4xl w-full space-y-8 bg-white p-10 rounded-2xl shadow-xl border border-slate-100">
        <div class="text-center">
            <h2 class="text-3xl font-extrabold text-slate-900 tracking-tight">Patient Survival Prognosis Model</h2>
            <p class="mt-2 text-sm text-slate-500">Enter patient metrics to predict survival outcomes (Alive/Deceased).</p>
        </div>

        {% if prediction %}
        <div class="p-4 rounded-xl text-center font-semibold text-lg border transition-all duration-300 {{ alert_class }}">
            {{ prediction }}
        </div>
        {% endif %}

        <form class="mt-8 space-y-6" method="POST">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Age</label>
                    <input type="number" step="any" name="Age" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Gender</label>
                    <select name="Gender" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="0">Female (0)</option>
                        <option value="1">Male (1)</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">State (Encoded)</label>
                    <input type="number" step="any" name="State" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">City (Encoded)</label>
                    <input type="number" step="any" name="City" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Cancer Type (Encoded)</label>
                    <input type="number" step="any" name="Cancer_Type" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Stage (Encoded)</label>
                    <input type="number" step="any" name="Stage" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Treatment Type (Encoded)</label>
                    <input type="number" step="any" name="Treatment_Type" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Survival Months</label>
                    <input type="number" step="any" name="Survival_Months" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
            </div>
            <button type="submit" class="w-full py-3 px-4 rounded-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 transition-all active:scale-[0.99]">
                Generate Prognosis
            </button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    alert_class = ""
    
    if request.method == 'POST':
        try:
            # Extract inputs from form based on model features
            data = [
                float(request.form.get('Age')),
                float(request.form.get('Gender')),
                float(request.form.get('State')),
                float(request.form.get('City')),
                float(request.form.get('Cancer_Type')),
                float(request.form.get('Stage')),
                float(request.form.get('Treatment_Type')),
                float(request.form.get('Survival_Months'))
            ]
            
            # Predict using decision_tree.pkl
            prediction = model.predict(np.array([data]))[0]
            
            if str(prediction).lower() == 'alive':
                prediction_text = "Result: Patient is likely to be ALIVE"
                alert_class = "bg-emerald-50 text-emerald-800 border-emerald-200"
            else:
                prediction_text = "Result: Patient is likely DECEASED"
                alert_class = "bg-rose-50 text-rose-800 border-rose-200"
                
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
            alert_class = "bg-amber-50 text-amber-800 border-amber-200"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction_text, alert_class=alert_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
