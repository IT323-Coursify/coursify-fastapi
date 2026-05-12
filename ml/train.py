"""
Course Recommendation Model Training
- 35 courses from programsData
- Big Five Personality (5 features)
- RIASEC Interests (6 features)
- Academic Scores (4 subjects: Math, English, Science, Abstract)
- Strand (1 categorical)
- 5 Algorithms: Decision Tree, SVC, Random Forest, KNN, GaussianNB
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Create directories
Path('ml').mkdir(exist_ok=True)

print("=" * 80)
print("COURSE RECOMMENDATION MODEL TRAINING")
print("=" * 80)

# ============================================
# ALL 35 COURSES FROM programsData
# ============================================

ALL_COURSES = [
    # Engineering (9 courses)
    "Civil Engineering", "Electrical Engineering", "Computer Engineering",
    "Mechanical Engineering", "Geodetic Engineering", "Electronics Engineering",
    "Environmental Engineering", "Agricultural and Biosystems Engineering",
    "Naval Architecture and Marine Engineering",
    
    # Computer Science and Information Systems (4 courses)
    "Computer Science", "Data Science", "Technology Communication Management",
    "Information Technology",
    
    # Technology (7 courses)
    "Agricultural Technology", "Autotronics", "Electro-Mechanical Technology",
    "Electronics Technology", "Energy Systems and Management",
    "Food Processing and Technology", "Manufacturing Engineering Technology",
    
    # Life Sciences (4 courses)
    "Agriculture", "Agroforestry", "Horticulture and Management", "Marine Biology",
    
    # Natural Sciences (4 courses)
    "Applied Mathematics", "Applied Physics", "Chemistry", "Environmental Science",
    
    # Social Sciences (5 courses)
    "Secondary Education (Math)", "Secondary Education (Science)", "Social Work",
    "Technical-Vocational Teacher", "Technology and Livelihood Education",
    
    # Art and Humanities (1 course)
    "Architecture"
]

print(f"\n Total courses: {len(ALL_COURSES)}")

# ============================================
# GENERATE STUDENT ASSESSMENT DATA
# ============================================

def generate_student_data(records_per_course=150):
    """Generate student assessment data for all courses"""
    np.random.seed(42)
    all_records = []
    
    # Define base profiles for each course type
    def get_profile(course_name):
        # Engineering profiles
        if course_name in ["Civil Engineering", "Mechanical Engineering", "Geodetic Engineering"]:
            return {
                "big_five": {"openness": 0.68, "conscientiousness": 0.82, "extraversion": 0.45, "agreeableness": 0.62, "neuroticism": 0.28},
                "riasec": {"R": 0.88, "I": 0.70, "A": 0.14, "S": 0.40, "E": 0.38, "C": 0.78},
                "academic": {"Math": 85, "English": 68, "Science": 82, "Abstract": 78},
                "strand": {"STEM": 0.90, "GAS": 0.08, "TVL": 0.02}
            }
        elif course_name == "Computer Engineering":
            return {
                "big_five": {"openness": 0.82, "conscientiousness": 0.75, "extraversion": 0.40, "agreeableness": 0.60, "neuroticism": 0.26},
                "riasec": {"R": 0.72, "I": 0.90, "A": 0.18, "S": 0.38, "E": 0.42, "C": 0.68},
                "academic": {"Math": 89, "English": 72, "Science": 86, "Abstract": 88},
                "strand": {"STEM": 0.88, "GAS": 0.10, "TVL": 0.02}
            }
        elif course_name == "Electrical Engineering":
            return {
                "big_five": {"openness": 0.72, "conscientiousness": 0.78, "extraversion": 0.42, "agreeableness": 0.58, "neuroticism": 0.30},
                "riasec": {"R": 0.85, "I": 0.80, "A": 0.10, "S": 0.35, "E": 0.40, "C": 0.72},
                "academic": {"Math": 87, "English": 66, "Science": 85, "Abstract": 80},
                "strand": {"STEM": 0.92, "GAS": 0.06, "TVL": 0.02}
            }
        elif course_name == "Electronics Engineering":
            return {
                "big_five": {"openness": 0.74, "conscientiousness": 0.76, "extraversion": 0.41, "agreeableness": 0.59, "neuroticism": 0.29},
                "riasec": {"R": 0.82, "I": 0.85, "A": 0.11, "S": 0.36, "E": 0.39, "C": 0.70},
                "academic": {"Math": 86, "English": 67, "Science": 84, "Abstract": 82},
                "strand": {"STEM": 0.91, "GAS": 0.07, "TVL": 0.02}
            }
        elif course_name == "Environmental Engineering":
            return {
                "big_five": {"openness": 0.78, "conscientiousness": 0.79, "extraversion": 0.50, "agreeableness": 0.72, "neuroticism": 0.30},
                "riasec": {"R": 0.75, "I": 0.82, "A": 0.20, "S": 0.58, "E": 0.44, "C": 0.68},
                "academic": {"Math": 80, "English": 74, "Science": 87, "Abstract": 77},
                "strand": {"STEM": 0.82, "GAS": 0.15, "TVL": 0.03}
            }
        elif course_name == "Agricultural and Biosystems Engineering":
            return {
                "big_five": {"openness": 0.72, "conscientiousness": 0.78, "extraversion": 0.52, "agreeableness": 0.70, "neuroticism": 0.32},
                "riasec": {"R": 0.88, "I": 0.72, "A": 0.13, "S": 0.50, "E": 0.40, "C": 0.65},
                "academic": {"Math": 76, "English": 68, "Science": 84, "Abstract": 72},
                "strand": {"STEM": 0.70, "GAS": 0.15, "TVL": 0.15}
            }
        elif course_name == "Naval Architecture and Marine Engineering":
            return {
                "big_five": {"openness": 0.69, "conscientiousness": 0.83, "extraversion": 0.46, "agreeableness": 0.63, "neuroticism": 0.29},
                "riasec": {"R": 0.89, "I": 0.74, "A": 0.12, "S": 0.41, "E": 0.37, "C": 0.79},
                "academic": {"Math": 83, "English": 69, "Science": 81, "Abstract": 75},
                "strand": {"STEM": 0.87, "GAS": 0.10, "TVL": 0.03}
            }
        
        # CS/IS Courses
        elif course_name == "Computer Science":
            return {
                "big_five": {"openness": 0.85, "conscientiousness": 0.73, "extraversion": 0.40, "agreeableness": 0.65, "neuroticism": 0.24},
                "riasec": {"R": 0.58, "I": 0.93, "A": 0.12, "S": 0.36, "E": 0.38, "C": 0.56},
                "academic": {"Math": 88, "English": 75, "Science": 90, "Abstract": 85},
                "strand": {"STEM": 0.85, "GAS": 0.12, "ABM": 0.03}
            }
        elif course_name == "Data Science":
            return {
                "big_five": {"openness": 0.87, "conscientiousness": 0.79, "extraversion": 0.38, "agreeableness": 0.62, "neuroticism": 0.25},
                "riasec": {"R": 0.52, "I": 0.95, "A": 0.14, "S": 0.34, "E": 0.40, "C": 0.60},
                "academic": {"Math": 92, "English": 78, "Science": 89, "Abstract": 93},
                "strand": {"STEM": 0.88, "GAS": 0.10, "ABM": 0.02}
            }
        elif course_name == "Technology Communication Management":
            return {
                "big_five": {"openness": 0.78, "conscientiousness": 0.74, "extraversion": 0.55, "agreeableness": 0.70, "neuroticism": 0.30},
                "riasec": {"R": 0.45, "I": 0.65, "A": 0.28, "S": 0.72, "E": 0.68, "C": 0.62},
                "academic": {"Math": 70, "English": 85, "Science": 68, "Abstract": 72},
                "strand": {"STEM": 0.40, "GAS": 0.35, "ABM": 0.25}
            }
        elif course_name == "Information Technology":
            return {
                "big_five": {"openness": 0.76, "conscientiousness": 0.71, "extraversion": 0.46, "agreeableness": 0.67, "neuroticism": 0.31},
                "riasec": {"R": 0.62, "I": 0.82, "A": 0.22, "S": 0.48, "E": 0.52, "C": 0.71},
                "academic": {"Math": 79, "English": 81, "Science": 76, "Abstract": 81},
                "strand": {"STEM": 0.58, "GAS": 0.27, "ABM": 0.15}
            }
        
        # Technology Courses
        elif course_name in ["Agricultural Technology", "Food Processing and Technology"]:
            return {
                "big_five": {"openness": 0.70, "conscientiousness": 0.73, "extraversion": 0.52, "agreeableness": 0.71, "neuroticism": 0.34},
                "riasec": {"R": 0.89, "I": 0.60, "A": 0.11, "S": 0.46, "E": 0.38, "C": 0.62},
                "academic": {"Math": 68, "English": 62, "Science": 76, "Abstract": 65},
                "strand": {"TVL": 0.60, "STEM": 0.25, "GAS": 0.15}
            }
        elif course_name in ["Autotronics", "Electro-Mechanical Technology", "Electronics Technology", "Manufacturing Engineering Technology"]:
            return {
                "big_five": {"openness": 0.69, "conscientiousness": 0.77, "extraversion": 0.45, "agreeableness": 0.61, "neuroticism": 0.31},
                "riasec": {"R": 0.87, "I": 0.72, "A": 0.09, "S": 0.39, "E": 0.41, "C": 0.72},
                "academic": {"Math": 76, "English": 63, "Science": 73, "Abstract": 70},
                "strand": {"TVL": 0.58, "STEM": 0.32, "GAS": 0.10}
            }
        elif course_name == "Energy Systems and Management":
            return {
                "big_five": {"openness": 0.74, "conscientiousness": 0.79, "extraversion": 0.50, "agreeableness": 0.66, "neuroticism": 0.29},
                "riasec": {"R": 0.72, "I": 0.80, "A": 0.12, "S": 0.48, "E": 0.55, "C": 0.68},
                "academic": {"Math": 81, "English": 72, "Science": 83, "Abstract": 76},
                "strand": {"STEM": 0.70, "GAS": 0.18, "TVL": 0.12}
            }
        
        # Life Sciences
        elif course_name in ["Agriculture", "Agroforestry", "Horticulture and Management"]:
            return {
                "big_five": {"openness": 0.73, "conscientiousness": 0.73, "extraversion": 0.53, "agreeableness": 0.73, "neuroticism": 0.34},
                "riasec": {"R": 0.84, "I": 0.70, "A": 0.17, "S": 0.52, "E": 0.43, "C": 0.58},
                "academic": {"Math": 68, "English": 66, "Science": 80, "Abstract": 66},
                "strand": {"STEM": 0.48, "GAS": 0.22, "TVL": 0.30}
            }
        elif course_name == "Marine Biology":
            return {
                "big_five": {"openness": 0.80, "conscientiousness": 0.73, "extraversion": 0.49, "agreeableness": 0.71, "neuroticism": 0.31},
                "riasec": {"R": 0.68, "I": 0.86, "A": 0.13, "S": 0.48, "E": 0.32, "C": 0.54},
                "academic": {"Math": 72, "English": 70, "Science": 88, "Abstract": 75},
                "strand": {"STEM": 0.75, "GAS": 0.15, "TVL": 0.10}
            }
        
        # Natural Sciences
        elif course_name == "Applied Mathematics":
            return {
                "big_five": {"openness": 0.82, "conscientiousness": 0.76, "extraversion": 0.36, "agreeableness": 0.58, "neuroticism": 0.26},
                "riasec": {"R": 0.48, "I": 0.94, "A": 0.10, "S": 0.30, "E": 0.35, "C": 0.62},
                "academic": {"Math": 95, "English": 72, "Science": 85, "Abstract": 90},
                "strand": {"STEM": 0.92, "GAS": 0.06, "ABM": 0.02}
            }
        elif course_name == "Applied Physics":
            return {
                "big_five": {"openness": 0.84, "conscientiousness": 0.75, "extraversion": 0.38, "agreeableness": 0.60, "neuroticism": 0.27},
                "riasec": {"R": 0.56, "I": 0.92, "A": 0.11, "S": 0.32, "E": 0.36, "C": 0.58},
                "academic": {"Math": 90, "English": 70, "Science": 94, "Abstract": 88},
                "strand": {"STEM": 0.94, "GAS": 0.05, "TVL": 0.01}
            }
        elif course_name == "Chemistry":
            return {
                "big_five": {"openness": 0.79, "conscientiousness": 0.77, "extraversion": 0.40, "agreeableness": 0.64, "neuroticism": 0.28},
                "riasec": {"R": 0.52, "I": 0.90, "A": 0.12, "S": 0.38, "E": 0.34, "C": 0.66},
                "academic": {"Math": 82, "English": 73, "Science": 92, "Abstract": 78},
                "strand": {"STEM": 0.90, "GAS": 0.08, "TVL": 0.02}
            }
        elif course_name == "Environmental Science":
            return {
                "big_five": {"openness": 0.81, "conscientiousness": 0.76, "extraversion": 0.52, "agreeableness": 0.74, "neuroticism": 0.30},
                "riasec": {"R": 0.64, "I": 0.84, "A": 0.22, "S": 0.62, "E": 0.46, "C": 0.64},
                "academic": {"Math": 76, "English": 78, "Science": 86, "Abstract": 74},
                "strand": {"STEM": 0.78, "GAS": 0.18, "HUMSS": 0.04}
            }
        
        # Social Sciences
        elif course_name in ["Secondary Education (Math)", "Secondary Education (Science)"]:
            return {
                "big_five": {"openness": 0.75, "conscientiousness": 0.80, "extraversion": 0.61, "agreeableness": 0.78, "neuroticism": 0.28},
                "riasec": {"R": 0.39, "I": 0.78, "A": 0.21, "S": 0.87, "E": 0.55, "C": 0.69},
                "academic": {"Math": 85, "English": 78, "Science": 80, "Abstract": 78},
                "strand": {"STEM": 0.46, "GAS": 0.34, "HUMSS": 0.20}
            }
        elif course_name == "Social Work":
            return {
                "big_five": {"openness": 0.76, "conscientiousness": 0.72, "extraversion": 0.67, "agreeableness": 0.88, "neuroticism": 0.34},
                "riasec": {"R": 0.30, "I": 0.56, "A": 0.26, "S": 0.94, "E": 0.62, "C": 0.66},
                "academic": {"Math": 62, "English": 84, "Science": 66, "Abstract": 64},
                "strand": {"HUMSS": 0.72, "GAS": 0.20, "ABM": 0.08}
            }
        elif course_name in ["Technical-Vocational Teacher", "Technology and Livelihood Education"]:
            return {
                "big_five": {"openness": 0.71, "conscientiousness": 0.77, "extraversion": 0.63, "agreeableness": 0.80, "neuroticism": 0.31},
                "riasec": {"R": 0.70, "I": 0.56, "A": 0.19, "S": 0.81, "E": 0.53, "C": 0.71},
                "academic": {"Math": 68, "English": 72, "Science": 66, "Abstract": 64},
                "strand": {"TVL": 0.56, "GAS": 0.24, "HUMSS": 0.20}
            }
        
        # Architecture
        elif course_name == "Architecture":
            return {
                "big_five": {"openness": 0.89, "conscientiousness": 0.76, "extraversion": 0.50, "agreeableness": 0.66, "neuroticism": 0.30},
                "riasec": {"R": 0.68, "I": 0.62, "A": 0.92, "S": 0.42, "E": 0.56, "C": 0.72},
                "academic": {"Math": 78, "English": 76, "Science": 72, "Abstract": 86},
                "strand": {"STEM": 0.42, "GAS": 0.30, "ARTS": 0.28}
            }
        
        # Default
        else:
            return {
                "big_five": {"openness": 0.70, "conscientiousness": 0.70, "extraversion": 0.50, "agreeableness": 0.65, "neuroticism": 0.30},
                "riasec": {"R": 0.60, "I": 0.60, "A": 0.40, "S": 0.50, "E": 0.50, "C": 0.60},
                "academic": {"Math": 70, "English": 70, "Science": 70, "Abstract": 70},
                "strand": {"STEM": 0.30, "GAS": 0.30, "ABM": 0.15, "HUMSS": 0.15, "TVL": 0.10}
            }
    
    for course in ALL_COURSES:
        profile = get_profile(course)
        
        for _ in range(records_per_course):
            noise = 0.08
            
            # Generate Big Five with noise
            openness = np.clip(profile["big_five"]["openness"] + np.random.normal(0, noise), 0, 1)
            conscientiousness = np.clip(profile["big_five"]["conscientiousness"] + np.random.normal(0, noise), 0, 1)
            extraversion = np.clip(profile["big_five"]["extraversion"] + np.random.normal(0, noise), 0, 1)
            agreeableness = np.clip(profile["big_five"]["agreeableness"] + np.random.normal(0, noise), 0, 1)
            neuroticism = np.clip(profile["big_five"]["neuroticism"] + np.random.normal(0, noise), 0, 1)
            
            # Generate RIASEC with noise
            realistic = np.clip(profile["riasec"]["R"] + np.random.normal(0, noise), 0, 1)
            investigative = np.clip(profile["riasec"]["I"] + np.random.normal(0, noise), 0, 1)
            artistic = np.clip(profile["riasec"]["A"] + np.random.normal(0, noise), 0, 1)
            social = np.clip(profile["riasec"]["S"] + np.random.normal(0, noise), 0, 1)
            enterprising = np.clip(profile["riasec"]["E"] + np.random.normal(0, noise), 0, 1)
            conventional = np.clip(profile["riasec"]["C"] + np.random.normal(0, noise), 0, 1)
            
            # Generate Academic Scores (0-100 scale, representing 10-question quiz)
            math_score = np.clip(profile["academic"]["Math"] + np.random.normal(0, 6), 0, 100)
            english_score = np.clip(profile["academic"]["English"] + np.random.normal(0, 6), 0, 100)
            science_score = np.clip(profile["academic"]["Science"] + np.random.normal(0, 6), 0, 100)
            abstract_score = np.clip(profile["academic"]["Abstract"] + np.random.normal(0, 6), 0, 100)
            
            # Select strand
            strands = list(profile["strand"].keys())
            weights = list(profile["strand"].values())
            strand = np.random.choice(strands, p=weights)
            
            record = {
                # Big Five
                "openness": round(openness, 3),
                "conscientiousness": round(conscientiousness, 3),
                "extraversion": round(extraversion, 3),
                "agreeableness": round(agreeableness, 3),
                "neuroticism": round(neuroticism, 3),
                # RIASEC
                "realistic": round(realistic, 3),
                "investigative": round(investigative, 3),
                "artistic": round(artistic, 3),
                "social": round(social, 3),
                "enterprising": round(enterprising, 3),
                "conventional": round(conventional, 3),
                # Academic
                "math_score": round(math_score, 1),
                "english_score": round(english_score, 1),
                "science_score": round(science_score, 1),
                "abstract_score": round(abstract_score, 1),
                # Strand
                "strand": strand,
                # Target
                "target_course": course
            }
            all_records.append(record)
    
    return pd.DataFrame(all_records)


# ============================================
# GENERATE STUDENT DATA
# ============================================

print("\n Generating student assessment data...")
df = generate_student_data(records_per_course=150)
print(f"   Generated {len(df)} student records")
print(f"   Total courses: {df['target_course'].nunique()}")
print(f"   Features per record: {len(df.columns)}")

# ============================================
# PREPARE FEATURES AND TARGET
# ============================================

feature_cols = [
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
    "realistic", "investigative", "artistic", "social", "enterprising", "conventional",
    "math_score", "english_score", "science_score", "abstract_score", "strand"
]

X = df[feature_cols].copy()
y = df["target_course"]

# Encode strand
strand_encoder = LabelEncoder()
X["strand"] = strand_encoder.fit_transform(X["strand"])

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n Training set: {len(X_train)} records")
print(f" Test set: {len(X_test)} records")

# ============================================
# TRAIN 5 MODELS
# ============================================

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42),
    "SVC": SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "Gaussian Naive Bayes": GaussianNB()
}

print("\n" + "=" * 80)
print(" TRAINING 5 MODELS")
print("=" * 80)

results = {}
best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    }
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print("\n" + "=" * 80)
print(f" BEST MODEL: {best_name} (Accuracy: {best_accuracy:.4f})")
print("=" * 80)

# ============================================
# SAVE MODEL AND ARTIFACTS
# ============================================

print("\n Saving model and artifacts...")

# Save best model
joblib.dump(best_model, "ml/coursify_model.pkl")

# Save all models
for name, result in results.items():
    joblib.dump(result["model"], f"ml/{name.lower().replace(' ', '_')}.pkl")

# Save encoders and scaler
joblib.dump(label_encoder, "ml/label_encoder.pkl")
joblib.dump(strand_encoder, "ml/strand_encoder.pkl")
joblib.dump(scaler, "ml/scaler.pkl")
joblib.dump(feature_cols, "ml/feature_columns.pkl")

print("\n Files saved to ml/ directory:")
print("   - coursify_model.pkl (best model)")
print("   - decision_tree.pkl")
print("   - svc.pkl")
print("   - random_forest.pkl")
print("   - k_nearest_neighbors.pkl")
print("   - gaussian_naive_bayes.pkl")
print("   - label_encoder.pkl")
print("   - strand_encoder.pkl")
print("   - scaler.pkl")
print("   - feature_columns.pkl")

# ============================================
# TEST PREDICTION
# ============================================

print("\n" + "=" * 80)
print("🔮 TEST PREDICTION")
print("=" * 80)

# Sample student (Computer Science candidate)
sample_student = {
    "big_five": {"openness": 0.85, "conscientiousness": 0.73, "extraversion": 0.40, "agreeableness": 0.65, "neuroticism": 0.24},
    "riasec": {"realistic": 0.58, "investigative": 0.93, "artistic": 0.12, "social": 0.36, "enterprising": 0.38, "conventional": 0.56},
    "academic": {"math": 88, "english": 75, "science": 90, "abstract": 85},
    "strand": "STEM"
}

# Create feature vector
features = [
    sample_student["big_five"]["openness"],
    sample_student["big_five"]["conscientiousness"],
    sample_student["big_five"]["extraversion"],
    sample_student["big_five"]["agreeableness"],
    sample_student["big_five"]["neuroticism"],
    sample_student["riasec"]["realistic"],
    sample_student["riasec"]["investigative"],
    sample_student["riasec"]["artistic"],
    sample_student["riasec"]["social"],
    sample_student["riasec"]["enterprising"],
    sample_student["riasec"]["conventional"],
    sample_student["academic"]["math"],
    sample_student["academic"]["english"],
    sample_student["academic"]["science"],
    sample_student["academic"]["abstract"],
    sample_student["strand"]
]

X_sample = pd.DataFrame([features], columns=feature_cols)
X_sample["strand"] = strand_encoder.transform([sample_student["strand"]])[0]
X_sample_scaled = scaler.transform(X_sample)

# Get predictions from best model
probs = best_model.predict_proba(X_sample_scaled)[0]
top5_idx = np.argsort(probs)[::-1][:5]

print(f"\n Top 5 Course Recommendations:")
for i, idx in enumerate(top5_idx):
    course = label_encoder.inverse_transform([idx])[0]
    confidence = probs[idx] * 100
    print(f"   {i+1}. {course} - {confidence:.1f}%")

print("\n" + "=" * 80)
print(" TRAINING COMPLETE!")
print("=" * 80)