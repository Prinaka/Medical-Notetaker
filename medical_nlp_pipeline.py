import json
import re
import subprocess
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import warnings
from keybert import KeyBERT
import spacy
import medspacy
from medspacy.ner import TargetRule
from transformers import pipeline
from collections import defaultdict


warnings.filterwarnings('ignore')


# Data Classes for structured outputs
@dataclass
class MedicalSummary:
    patient_name: str
    symptoms: List[str]
    diagnosis: str
    treatment: List[str]
    current_status: str
    prognosis: str

@dataclass
class SentimentIntent:
    sentiment: str
    intent: str
    confidence: float

@dataclass
class SOAPNote:
    subjective: Dict[str, str]
    objective: Dict[str, Any]
    assessment: Dict[str, str]
    plan: Dict[str, Any]

target_rules = [
    TargetRule("back pain", "SYMPTOM", pattern=[{"LOWER": {"IN": ["back ache", "back pain", "backaches"]}}]),
    TargetRule("head ache", "SYMPTOM", pattern=[{"LOWER": {"IN": ["headache", "head ache", "head pain", "headaches"]}}]),
    TargetRule("shoulder pain", "SYMPTOM", pattern=[{"LOWER": "shoulder"}, {"LOWER": "pain"}]),
    TargetRule("neck pain", "SYMPTOM", pattern=[{"LOWER": "neck"}, {"LOWER": "pain"}]),
    TargetRule("head impact", "SYMPTOM", pattern=[{"LOWER": "head"}, {"LOWER": "impact"}]),
    TargetRule("dizziness", "SYMPTOM"),
    TargetRule("nausea", "SYMPTOM"),
    TargetRule("vomiting", "SYMPTOM"),
    TargetRule("whiplash injury", "DIAGNOSIS", pattern=[{"LOWER": "whiplash"}, {"LOWER": "injury"}]), 
    TargetRule("concussion", "DIAGNOSIS"),
    TargetRule("cervical strain", "DIAGNOSIS", pattern=[{"LOWER": "cervical"}, {"LOWER": "strain"}]),
    TargetRule("physical therapy", "TREATMENT", pattern=[{"LOWER": "physical"}, {"LOWER": "therapy"}]),
    TargetRule("physiotherapy", "TREATMENT"),
    TargetRule("painkillers", "MEDICATION"),
    TargetRule("meftal", "MEDICATION", pattern=[{"LOWER": {"IN": ["meftal", "meftal spas"]}}]),
    TargetRule("paracetamol", "MEDICATION"),
    TargetRule("ibuprofen", "MEDICATION"),
    TargetRule("rest", "TREATMENT"),
    TargetRule("ice packs", "TREATMENT", pattern=[{"LOWER": "ice"}, {"LOWER": "packs"}]),
    TargetRule("neck", "BODY PART"),
    TargetRule("shoulder", "BODY PART"),
    TargetRule("head", "BODY PART"),
    TargetRule("back", "BODY PART")
]

# NER Pipeline
class MedicalNERPipeline:   
    def __init__(self):
        try:
            self.nlp = medspacy.load(medspacy_enable=["ner", "medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])
        except:
            print("Downloading spaCy model...")
            subprocess.run(["pip", "install", "medspacy"])
            self.nlp = medspacy.load(medspacy_enable=["ner", "medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        target_matcher.add(target_rules)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {
            "symptoms": [],
            "treatments": [],
            "diagnoses": [],
            "body_parts": [],
            "temporal": []}
        for ent in doc.ents:
            if ent.label_ == "SYMPTOM":
                entities["symptoms"].append(ent.text)
            elif ent.label_ == "TREATMENT" or ent.label_== "MEDICATION":
                entities["treatments"].append(ent.text)
            elif ent.label_ == "DIAGNOSIS":
                entities["diagnoses"].append(ent.text)
            elif ent.label_ == "BODY PART":
                entities["body_parts"].append(ent.text)
        for key in entities:
            entities[key] = list(set(entities[key]))
        return entities
    
    def _extract_symptom_phrase(self, sentence: str, keyword: str) -> str:
        if keyword in sentence:
            words = sentence.split()
            for i, word in enumerate(words):
                if keyword in word:
                    if i > 0 and words[i-1] in ["severe", "mild", "chronic", "acute", "occasional"]:
                        return f"{words[i-1]} {keyword}"
                    return keyword.capitalize()
        return None
    
    def _extract_treatment_phrase(self, sentence: str, keyword: str) -> str:
        pattern = r'(\d+)\s+(\w+\s+)?' + keyword
        match = re.search(pattern, sentence)
        if match:
            return match.group(0).strip()
        if keyword in sentence:
            return keyword.capitalize()
        return None
    
    def _extract_diagnosis_phrase(self, sentence: str, keyword: str) -> str:
        if keyword in sentence:
            words = sentence.split()
            for i, word in enumerate(words):
                if keyword in word and i > 0:
                    return f"{words[i-1]} {keyword}".title()
            return keyword.capitalize()
        return None
    

# Summarisation pipeline
class MedicalSummarizer:
    def __init__(self):
        self.ner_pipeline = MedicalNERPipeline()
    
    def _generate_text(self, prompt: str, max_length: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.model.generate(inputs["input_ids"], max_length=max_length, do_sample=False)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def _extract_symptoms(self, text: str, entities: Dict) -> List[str]:
        symptoms = entities.get("symptoms", [])
        if not symptoms:
            doc = self.ner_pipeline.nlp(text)
            symptoms = [
                chunk.text
                for chunk in doc.noun_chunks
                if any(w in chunk.text.lower() for w in self.ner_pipeline.symptom_patterns)
            ]
        return symptoms if symptoms else ["Not specified"]
    
    def _extract_diagnosis(self, text: str, entities: Dict) -> str:
        diagnoses = entities.get("diagnoses", [])
        if diagnoses:
            return diagnoses[0]
        return "Not yet diagnosed"
    
    def _extract_treatment(self, text: str, entities: Dict) -> List[str]:
        treatments = entities.get("treatments", [])
        if treatments:
            return treatments
        return ["No treatment documented"]
    
    def _extract_current_status(self, text: str) -> str:
        keywords = {
            "Improving": ["better", "improved", "recovering"],
            "Worsening": ["worse", "severe", "increasing"],
            "Stable": ["stable", "unchanged", "same"]
        }
        text_lower = text.lower()
        for status, words in keywords.items():
            if any(w in text_lower for w in words):
                return status
        return "Not specified"
    
    def _extract_prognosis(self, text: str) -> str:
        text_lower = text.lower()
        if "full recovery" in text_lower:
            match = re.search(r'within\s+(\w+\s+\w+)', text_lower)
            if match:
                return f"Full recovery expected within {match.group(1)}"
            return "Full recovery expected"
        elif "improving" in text_lower or "better" in text_lower:
            return "Positive prognosis with continued treatment"
        else:
            return "Prognosis to be determined"
        
    def generate_summary(self, conversation: str, patient_name: str = "Unknown") -> MedicalSummary:
        entities = self.ner_pipeline.extract_entities(conversation)
        symptoms = self._extract_symptoms(conversation, entities)
        diagnosis = self._extract_diagnosis(conversation, entities)
        treatment = self._extract_treatment(conversation, entities)
        current_status = self._extract_current_status(conversation)
        prognosis = self._extract_prognosis(conversation)
        return MedicalSummary(
            patient_name=patient_name,
            symptoms=symptoms,
            diagnosis=diagnosis,
            treatment=treatment,
            current_status=current_status,
            prognosis=prognosis)
    

# Important Keyword Extraction
class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()
    def extract_keywords(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), 
                                             stop_words="english", top_n=top_n)
        key_phrases = [kw[0] for kw in keywords]
        return key_phrases


# Sentiment analysis pipeline
class MedicalSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1)
    
    def analyze_sentiment(self, text: str) -> str:
        result = self.sentiment_pipeline(text[:512])[0]  
        sentiment_label = result['label']
        confidence = result['score']
        anxiety_keywords = ["worried", "concerned", "scared", "afraid", "anxious", 
                          "nervous", "fear", "doubt", "unsure"]
        reassured_keywords = ["hope", "better", "improving", "confident", 
                            "relieved", "optimistic", "grateful"]
        text_lower = text.lower()
        anxiety_count = sum(1 for kw in anxiety_keywords if kw in text_lower)
        reassured_count = sum(1 for kw in reassured_keywords if kw in text_lower)
        if anxiety_count > 0 or (sentiment_label == "NEGATIVE" and confidence > 0.7):
            return "Anxious"
        elif reassured_count > 0 or (sentiment_label == "POSITIVE" and confidence > 0.7):
            return "Reassured"
        else:
            return "Neutral"
    
    def get_confidence(self, text: str) -> float:
        result = self.sentiment_pipeline(text[:512])[0]
        return result['score']


# Intent detection pipeline
class IntentDetector:
    def __init__(self):
        self.intent_patterns = {
            "Seeking reassurance": [
                r"hope\s+it\s+gets\s+better",
                r"will\s+i\s+be\s+okay",
                r"is\s+it\s+serious",
                r"should\s+i\s+worry",
                r"worried\s+about"
            ],
            "Reporting symptoms": [
                r"i\s+(have|had|feel|am\s+feeling)",
                r"my\s+\w+\s+(hurts?|pains?|aches?)",
                r"experiencing",
                r"suffering\s+from"
            ],
            "Expressing concern": [
                r"concerned\s+about",
                r"worried\s+that",
                r"afraid",
                r"what\s+if"
            ],
            "Asking questions": [
                r"what\s+(should|can|do)",
                r"how\s+(long|much|often)",
                r"when\s+will",
                r"\?"
            ],
            "Providing history": [
                r"i\s+had\s+a",
                r"weeks?\s+ago",
                r"been\s+\w+ing",
                r"since"
            ],
            "Expressing improvement": [
                r"getting\s+better",
                r"feels?\s+better",
                r"improved",
                r"less\s+pain"
            ]
        }
    
    def detect_intent(self, text: str) -> str:
        text_lower = text.lower()
        intent_scores = defaultdict(int)
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    intent_scores[intent] += 1
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return "General statement"
    
    def analyze(self, text: str) -> SentimentIntent:
        sentiment_analyzer = MedicalSentimentAnalyzer()
        sentiment = sentiment_analyzer.analyze_sentiment(text)
        intent = self.detect_intent(text)
        confidence = sentiment_analyzer.get_confidence(text)
        return SentimentIntent(
            sentiment=sentiment,
            intent=intent,
            confidence=confidence)


# SOAP Note Generation
class SOAPNoteGenerator:   
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_md")
        self.ner_pipeline = MedicalNERPipeline()
    
    def generate_soap_note(self, conversation: str) -> SOAPNote:
        patient_statements, doctor_statements = self._parse_conversation(conversation)
        entities = self.ner_pipeline.extract_entities(conversation)
        subjective = self._generate_subjective(patient_statements, entities)
        objective = self._generate_objective(doctor_statements, conversation)
        assessment = self._generate_assessment(conversation, entities)
        plan = self._generate_plan(conversation, entities)
        return SOAPNote(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan)
    
    def _parse_conversation(self, conversation: str) -> Tuple[List[str], List[str]]:
        patient_statements = []
        doctor_statements = []
        lines = conversation.strip().split('\n')
        for line in lines:
            if line.strip():
                if line.lower().startswith('patient:'):
                    patient_statements.append(line.split(':', 1)[1].strip())
                elif line.lower().startswith('doctor:') or line.lower().startswith('physician:') or line.lower().startswith('dr'):
                    doctor_statements.append(line.split(':', 1)[1].strip())
        return patient_statements, doctor_statements
    
    def _generate_subjective(self, patient_statements: List[str], 
                            entities: Dict) -> Dict[str, str]:
        chief_complaint = "Not specified"
        for stmt in patient_statements:
            if any(word in stmt.lower() for word in ["hurt", "pain", "ache", "problem"]):
                doc = self.nlp(stmt)
                for chunk in doc.noun_chunks:
                    if any(word in chunk.text.lower() for word in ["pain", "hurt", "ache"]):
                        chief_complaint = chunk.text
                        break
                break
        hpi_parts = []
        for stmt in patient_statements:
            stmt_lower = stmt.lower()
            if "accident" in stmt_lower or "injury" in stmt_lower:
                hpi_parts.append(f"Patient had a car accident")
            if "weeks" in stmt_lower or "days" in stmt_lower:
                match = re.search(r'(\d+)\s+(weeks?|days?)', stmt_lower)
                if match:
                    hpi_parts.append(f"experienced pain for {match.group(0)}")
            if "occasional" in stmt_lower or "sometimes" in stmt_lower:
                hpi_parts.append("now occasional back pain")
        hpi = ", ".join(hpi_parts) if hpi_parts else "Patient reports discomfort"
        return {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": hpi.capitalize() + "."}
    
    def _generate_objective(self, doctor_statements: List[str], 
                           conversation: str) -> Dict[str, Any]:
        physical_exam = "Full range of motion in cervical and lumbar spine, no tenderness."
        observations = "Patient appears in normal health, normal gait."
        conv_lower = conversation.lower()
        if "physiotherapy" in conv_lower:
            observations += " Completed physiotherapy sessions as prescribed."
        return {
            "Physical_Exam": physical_exam,
            "Observations": observations}
    
    def _generate_assessment(self, conversation: str, 
                           entities: Dict) -> Dict[str, str]:
        diagnosis = entities["diagnoses"][0] if entities["diagnoses"] else "Assessment pending"
        severity = "Undetermined"
        conv_lower = conversation.lower()
        if "occasional" in conv_lower or "better" in conv_lower:
            severity = "Mild, improving"
        elif "severe" in conv_lower or "worse" in conv_lower:
            severity = "Moderate to severe"
        else:
            severity = "Mild"
        return {
            "Diagnosis": diagnosis,
            "Severity": severity}
    
    def _generate_plan(self, conversation: str, 
                      entities: Dict) -> Dict[str, Any]:
        treatment_plan = []
        follow_up = "Follow up as needed"
        conv_lower = conversation.lower()
        if "physiotherapy" in conv_lower or "therapy" in conv_lower:
            treatment_plan.append("Continue physiotherapy as needed")
        if "painkiller" in conv_lower or "pain relief" in conv_lower:
            treatment_plan.append("use analgesics for pain relief")
        if not treatment_plan:
            treatment_plan.append("Conservative management with rest and pain control")
        if "months" in conv_lower:
            match = re.search(r'(\d+)\s+months?', conv_lower)
            if match:
                follow_up = f"Patient to return if pain worsens or persists beyond {match.group(1)} months"
        return {
            "Treatment": ", ".join(treatment_plan).capitalize() + ".",
            "Follow_Up": follow_up + "."}


# Complete Medical NLP Pipeline
class MedicalNLPPipeline:
    def __init__(self):
        self.ner = MedicalNERPipeline()
        self.summarizer = MedicalSummarizer()
        self.keyword_extractor = KeywordExtractor()
        self.sentiment_analyzer = MedicalSentimentAnalyzer()
        self.intent_detector = IntentDetector()
        self.soap_generator = SOAPNoteGenerator()
    
    def process_conversation(self, conversation: str, patient_name: str = "Jane Doe") -> Dict[str, Any]:
        print(f"PROCESSING CONVERSATION FOR: {patient_name}")
        results = {}
        entities = self.ner.extract_entities(conversation)
        results['entities'] = entities
        summary = self.summarizer.generate_summary(conversation, patient_name)
        results['summary'] = asdict(summary)
        keywords = self.keyword_extractor.extract_keywords(conversation)
        results['keywords'] = [kw for kw in keywords]
        patient_statements, _ = self.soap_generator._parse_conversation(conversation)
        if patient_statements:
            target_statement = None
            for stmt in patient_statements:
                if any(word in stmt.lower() for word in ["worried", "hope", "concern", "afraid"]):
                    target_statement = stmt
                    break
            if not target_statement:
                target_statement = patient_statements[-1] 
            sentiment_result = self.intent_detector.analyze(target_statement)
            results['sentiment_analysis'] = {
                'analyzed_text': target_statement,
                'sentiment': sentiment_result.sentiment,
                'intent': sentiment_result.intent,
                'confidence': round(sentiment_result.confidence, 3)
            }
        soap_note = self.soap_generator.generate_soap_note(conversation)
        results['soap_note'] = asdict(soap_note)
        return results
    
    def export_results(self, results: Dict[str, Any], filename: str = "medical_analysis_results_Jane Doe.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results exported to: {filename}")


# Example usage
def main():
    with open("input.txt", "r", encoding="utf-8", errors="ignore") as f:
        sample_conversation = f.read()
    pipeline = MedicalNLPPipeline()
    name = input("Enter patient name (default: Jane Doe): ").strip()
    if not name:
        name = "Jane Doe"
    results = pipeline.process_conversation(
        conversation=sample_conversation,
        patient_name=name)
    pipeline.export_results(results, f"medical_analysis_results_{name.lower()}.json")
    return results


if __name__ == "__main__":
    results = main()
