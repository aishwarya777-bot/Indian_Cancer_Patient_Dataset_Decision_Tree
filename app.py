import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 1. Safe Model Loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'decision_tree.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    model = None

# 2. Integrated Beautiful UI Layout (Tailwind CSS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Patient Survival Prognosis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-4xl w-full space-y-8 bg-white p-10 rounded-2xl shadow-xl border border-slate-100">
        
        <div class="text-center">
            <h2 class="text-3xl font-extrabold text-slate-900 tracking-tight">Patient Survival Prognosis Model</h2>
            <p class="mt-2 text-sm text-slate-500">Enter clinical and demographic metrics to evaluate patient outcomes.</p>
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
                    <input type="number" step="any" name="Age" required placeholder="e.g., 55" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Gender</label>
                    <select name="Gender" required class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="0">Female (0)</option>
                        <option value="1">Male (1)</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">State (Encoded Number)</label>
                    <input type="number" step="any" name="State" required placeholder="e.g., 2" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">City (Encoded Number)</label>
                    <input type="number" step="any" name="City" required placeholder="e.g., 12" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Cancer Type (Encoded Number)</label>
                    <input type="number" step="any" name="Cancer_Type" required placeholder="e.g., 1" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Stage (Encoded Number)</label>
                    <input type="number" step="any" name="Stage" required placeholder="e.g., 3" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Treatment Type (Encoded Number)</label>
                    <input type="number" step="any" name="Treatment_Type" required placeholder="e.g., 0" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Survival Months</label>
                    <input type="number" step="any" name="Survival_Months" required placeholder="e.g., 24" class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
            </div>
            <button type="submit" class="w-full py-3 px-4 rounded-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 transition-all active:scale-[0.99] shadow-md">
                Generate Prediction
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
        if model is None:
            return render_template_string(HTML_TEMPLATE, prediction="Error: Model file failed to load on server.", alert_class="bg-amber-50 text-amber-800 border-amber-200")
            
        try:
            # Structuring inputs in the exact layout expected by the Decision Tree
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
            
            # Formulate prediction matrix
            input_features = np.array([data])
            prediction = model.predict(input_features)[0]
            
            # Explicit string parsing matching the model properties ('Alive' vs 'Deceased')
            if str(prediction).strip().lower() == 'alive':
                prediction_text = "Analysis Complete: Patient status is predicted as ALIVE"
                alert_class = "bg-emerald-50 text-emerald-800 border-emerald-200"
            else:
                prediction_text = "Analysis Complete: Patient status is predicted as DECEASED"
                alert_class = "bg-rose-50 text-rose-800 border-rose-200"
                
        except Exception as e:
            prediction_text = f"Input Error: {str(e)}"
            alert_class = "bg-amber-50 text-amber-800 border-amber-200"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction_text, alert_class=alert_class)

if __name__ == '__main__':
    # Default port configuration for standard local validation
    app.run(host='0.0.0.0', port=5000, debug=True)

