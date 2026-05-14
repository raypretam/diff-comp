#!/usr/bin/env python3
"""
Fine-Grained Sanskrit Compound Type Identification using Gemini API

Identifies detailed Sanskrit compound subtypes based on traditional samāsa taxonomy.
Supports 50+ fine-grained compound classifications.
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Tuple
import re
import google.generativeai as genai


class FineGrainedCompoundIdentifier:
    """Identifies fine-grained Sanskrit compound types using Gemini API."""
    
    SYSTEM_PROMPT = """You are an expert Sanskrit grammarian specializing in fine-grained compound word (samāsa) analysis following Pāṇinian tradition. Your task is to identify ALL compounds in a segmented Sanskrit sentence with their PRECISE subtypes, including nested compounds.

**IMPORTANT: Use ONLY the English type codes (T6, Di, Bs5, K1, etc.) in the "type" field. DO NOT use Devanagari names.**

**FINE-GRAINED COMPOUND TYPES:**

**1. TATPURUSHA (Determinative Compounds) - First element modifies second:**

   **T6** - षष्ठीतत्पुरुष (Genitive): X-sya Y (Y of X)
      - Example: rāja-putra → rājñaḥ putraḥ (king's son)

   **T7** - सप्तमीतत्पुरुष (Locative): X-e Y (Y in/on X)
      - Example: grāma-vāsa → grāme vāsaḥ (living in village)

   **T5** - पञ्चमीतत्पुरुष (Ablative): X-āt Y (Y from X)
      - Example: svarga-patita → svargāt patitaḥ (fallen from heaven)

   **T4** - चतुर्थीतत्पुरुष (Dative): X-āya Y (Y for X)
      - Example: gurave-dakṣiṇā → gurave dakṣiṇā (offering for teacher)

   **T3** - तृतीयातत्पुरुष (Instrumental): X-ā Y (Y by/with X)
      - Example: khaḍga-kṛtta → khaḍgena kṛttaḥ (cut by sword)

   **T2** - द्वितीयातत्पुरुष (Accusative): X-am Y (Y towards X)
      - Example: grāma-gata → grāmam gataḥ (gone to village)

   **T1** - प्रथमातत्पुरुष (Nominative): X-aḥ Y (rare, used in specific contexts)

   **Tm** - मध्यमपदलोपी (Elided middle element)
      - Example: कार्य-कारण-भाव-अन्तरम् where भाव is the implicit middle element

   **Tn** - नञ्तत्पुरुष (Negative with na-/a-)
      - Example: a-dharma → na dharmaḥ (non-dharma)

   **Tp** - उपपदतत्पुरुष (With preverb/upasarga)
      - Example: su-putra → (good son), dur-jana → (bad person)

   **Tds/Tdt/Tdu** - दिक्समास variants (Directional compounds)

   **Tg** - गतिसमास (With motion verbs)

   **Tk** - कृत्समास (With kṛt suffixes)

**2. KARMADHARAYA (Descriptive Determinative) - Adjective + Noun or Noun + Noun in apposition:**

   **K1** - विशेषणपूर्वपदकर्मधारय (Adjective first)
      - Example: nīla-utpala → nīlam utpalam (blue lotus)

   **K2** - विशेषणोत्तरपदकर्मधारय (Adjective second)
      - Example: puruṣa-vyāghra → vyāghraḥ iva puruṣaḥ (tiger-like man)

   **K3** - विशेषणोभयपदकर्मधारय (Both adjectives)
      - Example: śīta-uṣṇa → śītam ca uṣṇam ca (cold and hot, both adjectives)

   **K4** - अवधारणापूर्वपदकर्मधारय (With limiting particle)

   **K5** - समानाधिकरणकर्मधारय (Same case ending, appositional)

   **K6** - उपमानपूर्वपदकर्मधारय (Simile, comparison first)
      - Example: puruṣa-siṃha → siṃhaḥ iva puruṣaḥ (lion-like man)

   **K7** - उपमानोत्तरपदकर्मधारय (Simile, comparison second)
      - Example: mukha-padma → padmam iva mukham (lotus-like face)

   **Km** - मध्यमपदलोपी कर्मधारय (Elided middle element in Karmadharaya)

**3. DVANDVA (Copulative/Coordinative) - X and Y:**

   **Di** - इतरेतरद्वन्द्व (Itaretara, mutual)
      - Example: rāma-kṛṣṇa → rāmaḥ ca kṛṣṇaḥ ca (Rama and Krishna)

   **Ds** - समाहारद्वन्द्व (Samāhāra, collective)
      - Example: pāṇi-pāda → pāṇī ca pādau ca (hands and feet, collectively)

   **D** - General Dvandva (when subtype unclear)

**4. BAHUVRIHI (Possessive/Exocentric) - Refers to entity outside compound:**

   **Bs6** - षष्ठीबहुव्रीहि (Genitive-based)
      - Example: pīta-ambara → yasya pītam ambaram (one who has yellow cloth)

   **Bs7** - सप्तमीबहुव्रीहि (Locative-based)
      - Example: cakra-pāṇi → yasya cakram pāṇau (one who has chakra in hand)

   **Bs5** - पञ्चमीबहुव्रीहि (Ablative-based)
      - Example: vīta-rāga → yasya rāgaḥ vīgataḥ (one from whom passion has gone)

   **Bs3** - तृतीयाबहुव्रीहि (Instrumental-based)

   **Bs2** - द्वितीयाबहुव्रीहि (Accusative-based)

   **Bsu** - उपमानबहुव्रीहि (Simile-based possessive)
      - Example: candra-mukha → yasya mukham candraḥ iva (whose face is like moon)

   **Bss** - सहबहुव्रीहि (With 'sa' = with/accompanied by)
      - Example: sa-putra → yasya putraḥ sahaḥ asti (one who has son with them)

   **BvS** - व्यधिकरणबहुव्रीहि (Different cases for constituents)

   **Bv** - General Bahuvrihi

   **Bvp** - प्रादिबहुव्रीहि (With preverb)

   **Bvs** - समानाधिकरणबहुव्रीहि (Same case)

   **Bsmn** - मध्यमपदलोपीबहुव्रीहि (Elided middle)

   **Bb** - ब्यधिकरणबहुव्रीहि (Spatial/locative possession)

   **Bsp** - नञ्बहुव्रीहि (Negative Bahuvrihi with na-)
      - Example: निर्-धन → yasya dhanam nāsti (one who has no wealth)

**5. DVIGU (Numerical Collective):**

   **U** - द्विगु (Numerical compound with collective sense)
      - Example: tri-loka → trayāṇām lokānām samāhāraḥ (collection of three worlds)
      - Also used for indeclinable compounds (अव्ययीभाव features)

**6. AVYAYIBHAVA (Adverbial/Indeclinable):**

   **U** - अव्ययीभाव (Forms adverb, indeclinable)
      - Example: yathā-śakti → śakti-anusāram (according to ability)
      - ā-mṛtyu → mṛtyu-paryantam (until death)

**7. SPECIAL CATEGORIES:**

   **A1** - अलुक्समास (No elision of case endings)
   **A2** - अलुक्तत्पुरुष
   **A7** - अलुक् variant

**CRITICAL INDEXING RULES:**

1. **0-based indexing, EXCLUSIVE end**:
   - [0:2] = words at indices 0 and 1 only
   - [2:5] = words at indices 2, 3, and 4

2. **Nested compounds** - List ALL levels:
   - INNER compounds FIRST, then OUTER
   - If [1:3] is a compound within [0:4], list [1:3] BEFORE [0:4]

3. **Output format**: Use ONLY English codes in "type" field
   - CORRECT: {"type": "T6", "text": "..."}
   - WRONG: {"type": "षष्ठीतत्पुरुष", "text": "..."}

4. **Output ONLY valid JSON** with all identified compounds

---

**EXAMPLE 1: Multiple Tatpurusha Subtypes**

Sentence: "हृत् अन्त स्थ संकट बोध"

Word indices: 0:हृत् 1:अन्त 2:स्थ 3:संकट 4:बोध

Total words: 5

Analysis:

Words 0–2 (हृत् अन्त) form T6 (षष्ठी): "in the heart" → हृदि अन्तः
Words 0–3 (हृत् अन्त स्थ) form U (अव्ययीभाव/द्विगु): "situated in the interior of heart"
Words 3–5 (संकट बोध) form T6 (षष्ठी): "awareness of danger"

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "T6", "text": "हृत् अन्त" },
{ "start": 0, "end": 3, "type": "U", "text": "हृत् अन्त स्थ" },
{ "start": 3, "end": 5, "type": "T6", "text": "संकट बोध" }
] }

---

**EXAMPLE 2: Bahuvrihi with Dvandva**

Sentence: "वीत राग भय क्रोधः"

Word indices: 0:वीत 1:राग 2:भय 3:क्रोधः

Total words: 4

Analysis:

Words 1–3 (राग भय) form Di (इतरेतरद्वन्द्व): "passion and fear"
Words 1–4 (राग भय क्रोधः) form Di: "passion, fear, and anger"
Words 0–4 (वीत राग भय क्रोधः) form Bs5 (पञ्चमीबहुव्रीहि): "one from whom passion, fear, and anger have departed"

Output JSON:

{ "compounds": [
{ "start": 1, "end": 3, "type": "Di", "text": "राग भय" },
{ "start": 1, "end": 4, "type": "Di", "text": "राग भय क्रोधः" },
{ "start": 0, "end": 4, "type": "Bs5", "text": "वीत राग भय क्रोधः" }
] }

---

**EXAMPLE 3: Complex Nested with Karmadharaya**

Sentence: "प्रथमा अन्त अर्थे कार्य कारण भाव अन्तरम्"

Word indices: 0:प्रथमा 1:अन्त 2:अर्थे 3:कार्य 4:कारण 5:भाव 6:अन्तरम्

Total words: 7

Analysis:

Words 0–2 (प्रथमा अन्त) form Bv (बहुव्रीहि): "having a primary end"
Words 0–3 (प्रथमा अन्त अर्थे) form K1: "for the purpose of the primary end"
Words 3–5 (कार्य कारण) form Di: "effect and cause"
Words 3–6 (कार्य कारण भाव) form T6: "the state of being effect-and-cause"
Words 3–7 (कार्य कारण भाव अन्तरम्) form Tm: "the difference in the state of effect-and-cause" (मध्यमपदलोपी with implicit middle element)

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "Bv", "text": "प्रथमा अन्त" },
{ "start": 0, "end": 3, "type": "K1", "text": "प्रथमा अन्त अर्थे" },
{ "start": 3, "end": 5, "type": "Di", "text": "कार्य कारण" },
{ "start": 3, "end": 6, "type": "T6", "text": "कार्य कारण भाव" },
{ "start": 3, "end": 7, "type": "Tm", "text": "कार्य कारण भाव अन्तरम्" }
] }

---

**EXAMPLE 4: Negative Compounds**

Sentence: "न क्षत शरीरः दुर् आत्मा"

Word indices: 0:न 1:क्षत 2:शरीरः 3:दुर् 4:आत्मा

Total words: 5

Analysis:

Words 0–2 (न क्षत) form Tn (नञ्तत्पुरुष): "not injured"
Words 0–3 (न क्षत शरीरः) form Bs6 (षष्ठीबहुव्रीहि): "one whose body is not injured"
Words 3–5 (दुर् आत्मा) form Bs6: "one who has a wicked soul" (दुर् is upasarga indicating negative)

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "Tn", "text": "न क्षत" },
{ "start": 0, "end": 3, "type": "Bs6", "text": "न क्षत शरीरः" },
{ "start": 3, "end": 5, "type": "Bs6", "text": "दुर् आत्मा" }
] }

---

**EXAMPLE 5: Upamāna (Simile) Compounds**

Sentence: "काक पक्ष धरः महेन्द्र करतलम्"

Word indices: 0:काक 1:पक्ष 2:धरः 3:महेन्द्र 4:करतलम्

Total words: 5

Analysis:

Words 0–2 (काक पक्ष) form T6: "crow's wing"
Words 0–3 (काक पक्ष धरः) form U (द्विगु/अव्ययीभाव): "one bearing crow's wing" (compound name/epithet)
Words 3–5 (महेन्द्र करतलम्) form T6: "Indra's palm"

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "T6", "text": "काक पक्ष" },
{ "start": 0, "end": 3, "type": "U", "text": "काक पक्ष धरः" },
{ "start": 3, "end": 5, "type": "T6", "text": "महेन्द्र करतलम्" }
] }

---

**EXAMPLE 6: Samāhāra Dvandva**

Sentence: "अल्प मध्यम उत्कृष्टतां शाप अनुग्रह समर्थः"

Word indices: 0:अल्प 1:मध्यम 2:उत्कृष्टतां 3:शाप 4:अनुग्रह 5:समर्थः

Total words: 6

Analysis:

Words 0–3 (अल्प मध्यम उत्कृष्टतां) form Ds (समाहारद्वन्द्व): "inferior, medium, and excellent (collectively)"
Words 3–5 (शाप अनुग्रह) form Di: "curse and blessing"
Words 3–6 (शाप अनुग्रह समर्थः) form T7: "capable of curse and blessing"

Output JSON:

{ "compounds": [
{ "start": 0, "end": 3, "type": "Ds", "text": "अल्प मध्यम उत्कृष्टतां" },
{ "start": 3, "end": 5, "type": "Di", "text": "शाप अनुग्रह" },
{ "start": 3, "end": 6, "type": "T7", "text": "शाप अनुग्रह समर्थः" }
] }

---

**ANALYSIS GUIDELINES:**

1. **Identify constituent relationships** - Determine semantic and grammatical connections
2. **Check for case relationships** - Identify which vibhakti (case) underlies the compound
3. **Determine head position** - Is the compound endocentric or exocentric?
4. **Look for nested structures** - Multiple layers of compounding
5. **Classify precisely** - Use the most specific subtype code
6. **ALWAYS use English codes** - type field must be T6, Di, Bs5, K1, etc. (NOT Devanagari)

Return ONLY valid JSON. Be thorough and identify ALL compounds at all nesting levels."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the fine-grained identifier.
        
        Args:
            api_key: Gemini API key
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY env variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def parse_conllu_sentence(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        """Parse CoNLL-U sentence to extract words."""
        words = []
        forms = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 2:
                continue
            
            if fields[1] == 'DUMMY':
                continue
            
            word_form = fields[1]
            clean_word = re.sub(r'_C[0-9]+$', '', word_form)
            
            words.append(clean_word)
            forms.append(word_form)
        
        return words, forms
    
    def read_conllu_file(self, filepath: str) -> List[List[str]]:
        """Read CoNLL-U file and split into sentences."""
        sentences = []
        current_sentence = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    current_sentence.append(line)
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences
    
    def create_prompt(self, words: List[str]) -> str:
        """Create prompt for fine-grained compound identification."""
        word_indices = " ".join([f"{i}:{word}" for i, word in enumerate(words)])
        sentence = " ".join(words)
        
        user_prompt = f"""
Sentence: "{sentence}"

Word indices: {word_indices}

Total words: {len(words)}

Analyze this Sanskrit sentence and identify ALL compounds with their PRECISE fine-grained types (T6, T7, Bs6, Di, Ds, K1, U, Tn, Tp, Tm, etc.), including nested ones. Return ONLY a valid JSON object."""
        
        return user_prompt
    
    def identify_compounds(self, words: List[str], temperature: float = 0.2,
                          max_retries: int = 3) -> Dict:
        """Identify fine-grained compound types."""
        user_prompt = self.create_prompt(words)
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=4096,
        )
        
        for attempt in range(max_retries):
            try:
                chat = self.model.start_chat(history=[])
                
                response = chat.send_message(
                    self.SYSTEM_PROMPT + "\n\n" + user_prompt,
                    generation_config=generation_config
                )
                
                response_text = response.text.strip()
                
                # Clean markdown
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                # Add metadata
                result['sentence'] = ' '.join(words)
                result['words'] = words
                result['total_words'] = len(words)
                
                return result
                
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    return {
                        "sentence": ' '.join(words),
                        "words": words,
                        "error": f"JSON parse failed after {max_retries} attempts: {e}",
                        "raw_response": response.text
                    }
                temperature = min(0.5, temperature + 0.1)
                
            except Exception as e:
                return {
                    "sentence": ' '.join(words),
                    "words": words,
                    "error": f"API error: {e}"
                }
        
        return {
            "sentence": ' '.join(words),
            "words": words,
            "error": "Max retries exceeded"
        }
    
    def process_conllu_file(self, filepath: str, temperature: float = 0.2,
                           num_samples: Optional[int] = None) -> List[Dict]:
        """Process entire CoNLL-U file with fine-grained analysis.

        Args:
            filepath: Path to CoNLL-U file
            temperature: Sampling temperature
            num_samples: Number of sentences to process (None for all)

        Returns:
            List of results for each sentence
        """
        sentences = self.read_conllu_file(filepath)

        # Limit number of sentences if specified
        if num_samples is not None:
            sentences = sentences[:num_samples]

        results = []

        for i, sentence_lines in enumerate(sentences, 1):
            words, forms = self.parse_conllu_sentence(sentence_lines)

            if not words:
                continue

            print(f"Processing sentence {i}/{len(sentences)}: {' '.join(words[:5])}...")

            result = self.identify_compounds(words, temperature)
            result['sentence_id'] = i
            result['original_forms'] = forms

            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-grained Sanskrit compound identification using Gemini API"
    )
    parser.add_argument(
        "input",
        help="Input CoNLL-U file or sentence string"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key"
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model (default: gemini-3-flash-preview)"
    )
    parser.add_argument(
        "--sentence",
        action="store_true",
        help="Treat input as sentence string"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=None,
        help="Number of sentences to process from the file (default: all)"
    )

    args = parser.parse_args()
    
    identifier = FineGrainedCompoundIdentifier(api_key=args.api_key, model_name=args.model)
    
    if args.sentence:
        words = args.input.split()
        result = identifier.identify_compounds(words, args.temperature)
        results = [result]
    else:
        results = identifier.process_conllu_file(args.input, args.temperature, args.num_samples)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "="*80)
        for result in results:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("="*80)


if __name__ == "__main__":
    main()