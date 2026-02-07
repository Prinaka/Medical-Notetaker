# Medical Notetaker

A comprehensive Natural Language Processing pipeline for analyzing doctor-patient conversations and extracting structured medical information.

## Features

1. **Medical Named Entity Recognition (NER)**
   - Extracts symptoms, diagnoses, treatments, medications, and body parts
   - Uses medspaCy with custom medical entity rules
   - Identifies temporal information

2. **Medical Summarization**
   - Generates structured patient summaries
   - Extracts key medical information
   - Outputs in standardized JSON format

3. **Keyword Extraction**
   - Identifies important medical terms using KeyBERT
   - Extracts 1-3 word medical phrases
   - Ranks keywords by importance

4. **Sentiment & Intent Analysis**
   - Detects patient emotional state (Anxious/Neutral/Reassured)
   - Identifies patient intent (seeking reassurance, reporting symptoms, etc.)
   - Provides confidence scores

5. **SOAP Note Generation**
   - Creates clinical documentation in SOAP format
   - **S**ubjective: Patient's perspective
   - **O**bjective: Physical findings
   - **A**ssessment: Diagnosis and severity
   - **P**lan: Treatment and follow-up

## Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Medical Models

```bash
# Download scispaCy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```


## Usage

### Basic Usage

```python
from medical_nlp_pipeline import MedicalNLPPipeline

# Initialize pipeline
pipeline = MedicalNLPPipeline()

# Sample conversation
conversation = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""

# Process conversation
results = pipeline.process_conversation(
    conversation=conversation,
    patient_name="John Doe"
)

# Access results
print(results['summary'])          # Medical summary
print(results['sentiment_analysis']) # Sentiment & intent
print(results['soap_note'])        # SOAP note
print(results['keywords'])         # Important keywords
```

### Reading from File

```python
# Create input file
with open("patient_conversation.txt", "w") as f:
    f.write("""
    Doctor: What brings you in today?
    Patient: I've been having severe headaches for two weeks.
    Doctor: Any other symptoms?
    Patient: Yes, some dizziness and nausea.
    """)

# Process from file
with open("patient_conversation.txt", "r") as f:
    conversation = f.read()

pipeline = MedicalNLPPipeline()
results = pipeline.process_conversation(conversation, "Jane Smith")

# Export results
pipeline.export_results(results, "analysis_jane_smith.json")
```

### Command Line Usage

```bash
# Create your input file
echo "Doctor: How are you? Patient: My back hurts." > input.txt

# Run the pipeline
python medical_nlp_pipeline.py
# Enter patient name when prompted (or press Enter for default)
```

## Output Format

### Medical Summary
```json
{
  "patient_name": "John Doe",
  "symptoms": ["Neck pain", "Back pain"],
  "diagnosis": "Whiplash injury",
  "treatment": ["10 physiotherapy sessions", "Painkillers"],
  "current_status": "Improving",
  "prognosis": "Full recovery expected within 6 months"
}
```

### Sentiment Analysis
```json
{
  "analyzed_text": "I'm worried about my back pain, but I hope it gets better.",
  "sentiment": "Anxious",
  "intent": "Seeking reassurance",
  "confidence": 0.87
}
```

### SOAP Note
```json
{
  "subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident, experienced pain for 4 weeks."
  },
  "objective": {
    "Physical_Exam": "Full range of motion in cervical and lumbar spine.",
    "Observations": "Patient appears in normal health, normal gait."
  },
  "assessment": {
    "Diagnosis": "Whiplash injury",
    "Severity": "Mild, improving"
  },
  "plan": {
    "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
    "Follow_Up": "Patient to return if pain worsens or persists beyond 6 months."
  }
}
```

## Customization

### Adding Custom Medical Entities

Add new medical terms to the target rules:

```python
from medspacy.ner import TargetRule

# Add custom symptoms
custom_rules = [
    TargetRule("migraine", "SYMPTOM"),
    TargetRule("arthritis", "DIAGNOSIS"),
    TargetRule("aspirin", "MEDICATION"),
]

# In MedicalNERPipeline.__init__:
target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
target_matcher.add(custom_rules)
```

### Modifying Intent Patterns

Extend the intent detection patterns:

```python
# In IntentDetector.__init__:
self.intent_patterns = {
    "Seeking reassurance": [
        r"hope\s+it\s+gets\s+better",
        r"will\s+i\s+be\s+okay",
        # Add your patterns here
    ],
    "Your Custom Intent": [
        r"your\s+pattern\s+here",
    ]
}
```

## Project Structure

```
medical-nlp-pipeline/
├── medical_nlp_pipeline.py    # Main pipeline code
├── input.txt                  # Sample input file
├── README.md                  # This file
└── output/
    └── medical_analysis_results_{name}.json
```

## How It Works

### 1. Named Entity Recognition (NER)

Uses **medspaCy** with custom rules to identify medical entities:

```python
# Custom rules for medical terms
TargetRule("whiplash injury", "DIAGNOSIS")
TargetRule("physiotherapy", "TREATMENT")
TargetRule("neck pain", "SYMPTOM")
```

### 2. Medical Summarization

Combines NER results with context analysis:
- Extracts symptoms from patient statements
- Identifies diagnoses from doctor statements
- Captures treatment details and timeframes

### 3. Sentiment Analysis

Uses **DistilBERT** fine-tuned on sentiment:
- Classifies base sentiment (positive/negative)
- Maps to medical context (anxious/reassured)
- Combines keyword analysis for accuracy

### 4. Keyword Extraction

Uses **KeyBERT** for medical term extraction:
- Identifies 1-3 word medical phrases
- Ranks by relevance to conversation
- Filters out common stop words

### 5. SOAP Note Generation

Structured clinical documentation:
- **Subjective**: Patient's chief complaint and history
- **Objective**: Physical exam findings
- **Assessment**: Diagnosis and severity
- **Plan**: Treatment recommendations and follow-up


## Input File Format

Create a text file (`input.txt`) with the required conversation:

**Format Requirements:**
- Each line starts with either `Doctor:` or `Patient:`
- Multiple statements from the same person can be on separate lines
- No special formatting needed

## References

- **medspaCy**: [https://github.com/medspacy/medspacy](https://github.com/medspacy/medspacy)
- **scispaCy**: [https://allenai.github.io/scispacy/](https://allenai.github.io/scispacy/)
- **KeyBERT**: [https://github.com/MaartenGr/KeyBERT](https://github.com/MaartenGr/KeyBERT)
- **Transformers**: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)


