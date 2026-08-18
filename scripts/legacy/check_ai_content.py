#!/usr/bin/env python3
"""
AI Content Detection Analysis for Thesis
Analyzes LaTeX chapters for AI-generated content indicators
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

class AIContentAnalyzer:
    def __init__(self, chapters_dir: str):
        self.chapters_dir = chapters_dir
        self.chapters = self._load_chapters()
        self.scores = {}
        self.detailed_results = {}
        
    def _load_chapters(self) -> Dict[str, str]:
        """Load all chapter files from the chapters directory"""
        chapters = {}
        if not os.path.exists(self.chapters_dir):
            raise ValueError(f"Chapters directory not found: {self.chapters_dir}")
        
        for file in sorted(os.listdir(self.chapters_dir)):
            if file.endswith('.tex'):
                filepath = os.path.join(self.chapters_dir, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove LaTeX commands but keep text
                    text = self._extract_text_from_latex(content)
                    chapters[file] = text
        
        return chapters
    
    def _extract_text_from_latex(self, latex_content: str) -> str:
        """Extract readable text from LaTeX content"""
        # Remove commands
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', latex_content)
        text = re.sub(r'\\[a-zA-Z]+\[[^\]]*\]', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        # Remove environments
        text = re.sub(r'%.*', '', text)
        text = re.sub(r'\$.*?\$', '', text)
        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _analyze_chapter(self, chapter_name: str, text: str) -> Dict:
        """Analyze a single chapter for AI indicators"""
        
        # Normalize text
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {'chapter': chapter_name, 'ai_likelihood': 0.0, 'indicators': []}
        
        indicators = []
        scores = []
        
        # 1. Formal transition phrases (common in AI)
        formal_transitions = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin conclusion\b', r'\bin summary\b', r'\bimportantly\b',
            r'\bnotably\b', r'\bsignificantly\b', r'\bone might argue\b',
            r'\bit is worth noting\b', r'\bit should be noted\b'
        ]
        
        transition_count = sum(
            len(re.findall(transition, text.lower())) 
            for transition in formal_transitions
        )
        transition_score = min(1.0, transition_count / max(1, len(sentences) / 5))
        if transition_score > 0.3:
            indicators.append(f"High formal transitions: {transition_count}")
        scores.append(transition_score * 0.15)
        
        # 2. Lack of contractions (AI rarely uses them)
        contractions = [
            r"\bcan't\b", r"\bwon't\b", r"\bdon't\b", r"\bdidn't\b",
            r"\bi'm\b", r"\bit's\b", r"\bthat's\b", r"\bwe're\b"
        ]
        contraction_count = sum(
            len(re.findall(c, text.lower())) for c in contractions
        )
        contraction_score = 1.0 - min(1.0, contraction_count / max(1, len(sentences) / 10))
        if contraction_score > 0.6:
            indicators.append(f"Few contractions (very AI-like): only {contraction_count}")
        scores.append(contraction_score * 0.10)
        
        # 3. Overly perfect grammar and punctuation
        double_spaces = len(re.findall(r'  ', text))
        missing_articles = len(re.findall(r'\b(?:is|are|was|were)\s+(?:[A-Z][a-z]+\s+){2,}', text))
        grammar_score = max(0, (double_spaces + missing_articles) / max(1, len(sentences)))
        if grammar_score < 0.05:
            indicators.append("Perfect grammar (unnaturally consistent)")
        scores.append((1.0 - grammar_score) * 0.12)
        
        # 4. Repetitive sentence structure
        sentence_starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        unique_starts = len(set(sentence_starts)) / max(1, len(sentence_starts))
        repetition_score = 1.0 - unique_starts
        if repetition_score > 0.4:
            indicators.append(f"Repetitive sentence structure: {int(unique_starts*100)}% unique starts")
        scores.append(repetition_score * 0.15)
        
        # 5. Generic phrases (common in AI)
        generic_phrases = [
            r'\bin this paper\b', r'\bthe purpose of this.*is\b', r'\bthis work\b',
            r'\bthe results show\b', r'\bthe findings indicate\b', r'\bto summarize\b',
            r'\bas shown above\b', r'\ball in all\b', r'\bon the other hand\b'
        ]
        generic_count = sum(
            len(re.findall(phrase, text.lower())) for phrase in generic_phrases
        )
        generic_score = min(1.0, generic_count / max(1, len(sentences) / 4))
        if generic_score > 0.2:
            indicators.append(f"Generic phrases count: {generic_count}")
        scores.append(generic_score * 0.15)
        
        # 6. Vocabulary sophistication (too high = AI)
        complex_words = [
            r'\bphenomenon\b', r'\bfacilitate\b', r'\bdemonstrate\b',
            r'\bentail\b', r'\belucidation\b', r'\bprofound\b'
        ]
        complex_count = sum(
            len(re.findall(w, text.lower())) for w in complex_words
        )
        complexity_score = min(0.8, complex_count / max(1, len(sentences) / 5))
        if complexity_score > 0.3:
            indicators.append(f"High vocabulary complexity: {complex_count} complex words")
        scores.append(complexity_score * 0.10)
        
        # 7. Paragraph uniformity (AI tends to create uniform paragraphs)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            length_variance = sum((l - avg_sentence_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            uniformity_score = 1.0 - min(1.0, (length_variance ** 0.5) / 20)
        else:
            uniformity_score = 0
        if uniformity_score > 0.5:
            indicators.append(f"Uniform paragraph structure (variance: {length_variance:.1f})")
        scores.append(uniformity_score * 0.10)
        
        # 8. Lack of personal voice/opinions
        personal_markers = [
            r'\bi think\b', r'\bmy opinion\b', r'\bi believe\b', r'\bin my view\b',
            r'\bwe found\b', r'\bour approach\b', r'\bour contribution\b'
        ]
        personal_count = sum(
            len(re.findall(m, text.lower())) for m in personal_markers
        )
        personal_score = 1.0 - min(1.0, personal_count / max(1, len(sentences) / 10))
        if personal_score > 0.6:
            indicators.append(f"Minimal personal voice: only {personal_count} personal references")
        scores.append(personal_score * 0.08)
        
        # 9. Check for proper nouns and specific details (AI often lacks these)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|pp|epochs|samples|parameters)\b', text)
        specificity_score = 1.0 - min(1.0, (len(proper_nouns) + len(numbers)) / max(1, len(sentences)))
        if specificity_score > 0.5:
            indicators.append(f"Moderate specificity: {len(numbers)} quantified metrics")
        scores.append(specificity_score * 0.08)
        
        # Calculate final AI likelihood
        ai_likelihood = sum(scores)
        
        return {
            'chapter': chapter_name,
            'ai_likelihood': ai_likelihood,
            'indicators': indicators,
            'details': {
                'formal_transitions': (transition_score * 0.15, transition_count),
                'contractions': (contraction_score * 0.10, contraction_count),
                'grammar': ((1.0 - grammar_score) * 0.12, double_spaces),
                'repetition': (repetition_score * 0.15, f"{int(unique_starts*100)}%"),
                'generic_phrases': (generic_score * 0.15, generic_count),
                'vocabulary': (complexity_score * 0.10, complex_count),
                'uniformity': (uniformity_score * 0.10, f"{length_variance:.1f}"),
                'personal_voice': (personal_score * 0.08, personal_count),
                'specificity': (specificity_score * 0.08, f"{len(numbers)} metrics")
            }
        }
    
    def analyze_all(self) -> Dict:
        """Analyze all chapters"""
        total_score = 0
        results = []
        
        print("\n" + "="*70)
        print("AI CONTENT DETECTION ANALYSIS")
        print("="*70)
        
        for chapter_name, text in self.chapters.items():
            result = self._analyze_chapter(chapter_name, text)
            results.append(result)
            total_score += result['ai_likelihood']
            
            print(f"\n{chapter_name}: {result['ai_likelihood']:.1%} AI Likelihood")
            if result['indicators']:
                for indicator in result['indicators'][:3]:
                    print(f"  [WARNING] {indicator}")
        
        average_score = total_score / len(results) if results else 0
        
        print("\n" + "="*70)
        print(f"OVERALL AI LIKELIHOOD: {average_score:.1%}")
        print("="*70)
        
        if average_score < 0.30:
            print("✓ EXCELLENT: Content appears genuinely human-written (< 30%)")
            print("   Report is ready for PDF export and submission")
        elif average_score < 0.50:
            print("🟡 GOOD: Content appears mostly human-written (30-50%)")
            print("   Consider minor rephrasing for sensitive sections")
        else:
            print("🔴 CAUTION: Content shows significant AI markers (> 50%)")
            print("   Recommend rephasing suspected sections")
        
        return {
            'average': average_score,
            'chapters': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_summary_report(self) -> str:
        """Generate a text summary report"""
        results = self.analyze_all()
        
        report = f"""
AI CONTENT DETECTION REPORT
Generated: {results['timestamp']}

OVERALL SCORE: {results['average']:.1%} AI Likelihood

INTERPRETATION:
- 0-20%:   Highly likely human-written (excellent)
- 20-40%:  Likely human-written with minor AI assistance
- 40-60%:  Mixed or uncertain
- 60-80%:  Likely AI-generated with human editing
- 80-100%: Highly likely AI-generated

CHAPTER BREAKDOWN:
"""
        
        for chapter_result in results['chapters']:
            report += f"\n{chapter_result['chapter']}: {chapter_result['ai_likelihood']:.1%}\n"
            for indicator in chapter_result['indicators'][:2]:
                report += f"  • {indicator}\n"
        
        recommendation = (
            "\n✅ READY FOR SUBMISSION\n"
            if results['average'] < 0.30
            else "\n⚠️  REVIEW RECOMMENDED\n"
        )
        report += recommendation
        
        return report


def main():
    project_root = Path(__file__).parent.parent
    chapters_dir = project_root / "MSE-CAPSTONE-REPORT" / "chapters"
    
    print("Analyzing thesis for AI-generated content...")
    analyzer = AIContentAnalyzer(str(chapters_dir))
    results = analyzer.analyze_all()
    
    # Save results
    results_file = project_root / "ai_detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
        print(f"\n[OK] Results saved to: {results_file}")
    
    # Print summary
    print(analyzer.get_summary_report())
    
    return results['average']


if __name__ == "__main__":
    ai_score = main()
    exit(0 if ai_score < 0.30 else 1)
