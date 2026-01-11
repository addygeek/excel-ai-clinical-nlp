# 🎨 Clinical NLP Visualization - Quick Start Guide

## ✅ What Was Created

You now have **3 powerful ways** to visualize your Clinical NLP results:

### 📁 **Files Created:**

1. **`visualize_results.py`** - Core visualization library
2. **`demo_visualize.py`** - Complete demo (generates & visualizes)
3. **`interactive_viewer.py`** - Menu-driven interactive viewer
4. **`README_VISUALIZATION.md`** - Full documentation

### 📊 **Generated Output Files:**

- ✅ `clinical_nlp_results.json` - Complete NLP analysis results
- ✅ `entity_distribution.png` - Bar chart of entity types
- ✅ `confidence_scores.png` - Confidence score analysis
- ✅ `entity_pie_chart.png` - Entity proportions
- ✅ `comprehensive_dashboard.png` - Multi-panel dashboard
- ✅ `clinical_entities.csv` - Excel-ready data export

---

## 🚀 Three Ways to Use

### 1️⃣ **Run Complete Demo** (Recommended First)

```bash
python demo_visualize.py
```

**What it does:**
- Generates sample clinical text analysis
- Extracts 22 medical entities
- Creates all 4 visualizations
- Exports to CSV
- Saves JSON results

**Perfect for:** First-time users, demonstrations

---

### 2️⃣ **Visualize Your Own Results**

```bash
# First, make sure you have clinical_nlp_results.json
# Then run:
python visualize_results.py
```

**What it does:**
- Loads your JSON results
- Displays comprehensive summary
- Generates all visualizations
- Creates statistical report

**Perfect for:** When you have your own NLP results

---

### 3️⃣ **Interactive Explorer**

```bash
python interactive_viewer.py
```

**Menu Options:**
```
1. 📋 Display Summary          - View detailed analysis
2. 📊 Show Statistics Report   - Statistical breakdown
3. 📈 Plot Entity Distribution - Bar chart
4. 📉 Plot Confidence Scores   - Confidence analysis
5. 🥧 Plot Entity Pie Chart    - Proportions
6. 🎨 Create Dashboard         - All-in-one view
7. 💾 Export to CSV            - Export data
8. 🔍 Search Entities          - Find specific entities
9. 📄 View Raw JSON            - View raw data
0. 🚪 Exit                     - Quit
```

**Perfect for:** Exploring data, searching entities, custom analysis

---

## 📊 What Gets Visualized

### **Entity Distribution** (`entity_distribution.png`)
```
ANATOMY     ████████ (8)
SYMPTOM     ████████████ (5)
TREATMENT   ███████ (2)
DIAGNOSIS   ███████ (2)
PROGNOSIS   ██████ (3)
PROCEDURE   ███ (1)
FACILITY    ███ (1)
```

### **Confidence Scores** (`confidence_scores.png`)
- Histogram showing score distribution
- Box plots comparing confidence by entity type
- Mean confidence: **82.14%**

### **Entity Pie Chart** (`entity_pie_chart.png`)
- Visual proportions of each entity type
- Percentage breakdown
- Color-coded categories

### **Comprehensive Dashboard** (`comprehensive_dashboard.png`)
- All charts in one view
- Top 10 entities table
- Entity type breakdown
- Confidence distributions

---

## 📝 Sample Output

```
📋 CLINICAL NLP ANALYSIS SUMMARY
================================================================================

🏷️  ENTITIES EXTRACTED: 22

  DIAGNOSIS (2):
    • whiplash injury (confidence: 85.00%)
    • lower back strain (confidence: 80.00%)

  SYMPTOM (5):
    • neck pain (confidence: 70.00%)
    • back pain (confidence: 70.00%)
    • discomfort (confidence: 70.00%)
    • stiffness (confidence: 70.00%)
    • pain (confidence: 70.00%)

  TREATMENT (2):
    • physiotherapy (confidence: 85.00%)
    • painkillers (confidence: 70.00%)

  PROGNOSIS (3):
    • full recovery (confidence: 85.00%)
    • within six months (confidence: 80.00%)
    • no long-term damage (confidence: 80.00%)

  ANATOMY (8):
    • head, neck, back, cervical, lumbar, spine...

  PROCEDURE (1):
    • x-rays (confidence: 70.00%)

  FACILITY (1):
    • city hospital accident and emergency (confidence: 80.00%)
```

---

## 📈 Statistical Analysis

```
📈 STATISTICAL ANALYSIS REPORT
================================================================================

📊 Total Entities: 22
📊 Unique Entity Types: 7

📈 Confidence Statistics:
  • Mean: 0.7455
  • Median: 0.7000
  • Std Dev: 0.0668
  • Min: 0.7000
  • Max: 0.8500

🏷️  Entity Type Breakdown:
  • ANATOMY: 8 (36.4%)
  • SYMPTOM: 5 (22.7%)
  • PROGNOSIS: 3 (13.6%)
  • DIAGNOSIS: 2 (9.1%)
  • TREATMENT: 2 (9.1%)
  • FACILITY: 1 (4.5%)
  • PROCEDURE: 1 (4.5%)
```

---

## 💡 Pro Tips

### **Tip 1: Search Specific Entities**
```bash
python interactive_viewer.py
# Select option 8 - Search Entities
# Enter: "pain"
# Get: All pain-related entities with confidence scores
```

### **Tip 2: Export to Excel**
```bash
python interactive_viewer.py
# Select option 7 - Export to CSV
# Open clinical_entities.csv in Excel
# Create pivot tables, custom charts, etc.
```

### **Tip 3: Use as Python Module**
```python
from visualize_results import ClinicalNLPVisualizer

viz = ClinicalNLPVisualizer('my_results.json')
viz.display_summary()
viz.plot_entity_distribution()
viz.export_to_csv('my_export.csv')
```

### **Tip 4: Batch Process Multiple Files**
```python
from visualize_results import ClinicalNLPVisualizer

for file in ['patient1.json', 'patient2.json', 'patient3.json']:
    viz = ClinicalNLPVisualizer(file)
    viz.create_comprehensive_dashboard()
```

---

## 🎯 Common Use Cases

### **Use Case 1: Demonstrate to Non-Technical Users**
```bash
python demo_visualize.py
# Show the generated PNG files
# Easy to understand, visually appealing
```

### **Use Case 2: Analyze Your Own Clinical Text**
```python
from ner_module import ClinicalNER
import json

# Your text
my_text = "Patient has severe headache and fever..."

# Generate results
ner = ClinicalNER()
entities = ner.extract_entities(my_text)

# Save and visualize
with open('clinical_nlp_results.json', 'w') as f:
    json.dump({'entities': entities}, f)

# Run visualizer
from visualize_results import ClinicalNLPVisualizer
viz = ClinicalNLPVisualizer()
viz.create_comprehensive_dashboard()
```

### **Use Case 3: Research & Analysis**
```bash
# Generate CSV for statistical analysis
python interactive_viewer.py
# Option 7: Export to CSV
# Open in Excel, R, or Python for advanced analysis
```

---

## 🔧 Troubleshooting

### **Issue: "File not found"**
```bash
# Make sure clinical_nlp_results.json exists
# Run demo first:
python demo_visualize.py
```

### **Issue: "Module not found"**
```bash
# Install dependencies:
pip install matplotlib seaborn pandas numpy
```

### **Issue: Plots not showing**
```bash
# For Jupyter/Colab, add:
import matplotlib.pyplot as plt
plt.show()

# Or save to file instead
viz.plot_entity_distribution(save_fig=True)
```

---

## 📚 Additional Resources

- **Full Documentation**: `README_VISUALIZATION.md`
- **Main README**: `README.md`
- **Source Code**: `visualize_results.py`

---

## ⚠️ Clinical Disclaimer

**This is a demonstration system and should NOT be used for actual medical decisions.**
**Always consult qualified healthcare professionals for medical advice.**

---

## 🎉 Quick Win

**Get started in 30 seconds:**

```bash
python demo_visualize.py
```

Then open the generated PNG files to see beautiful visualizations! 📊✨

---

**Created for Excellence in Clinical NLP! 🏥💻**
