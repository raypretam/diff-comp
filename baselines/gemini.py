#!/usr/bin/env python3
"""
Sanskrit Compound Type Identification using Gemini API

Processes CoNLL-U format data to identify compound structures (samāsa) in Sanskrit.
Supports nested compounds with proper indexing.
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Tuple
import re
import google.generativeai as genai


class SanskritCompoundIdentifier:
    """Identifies Sanskrit compounds using Gemini API with specialized prompt."""
    
    SYSTEM_PROMPT = """You are an expert Sanskrit linguist specializing in compound word (samāsa) analysis. Your task is to identify ALL compounds in a segmented Sanskrit sentence, including nested compounds.

**COMPOUND TYPES:**

1. **Tatpurusha **: Determinative compound where first element modifies second
   - Test: Can be paraphrased as X-sya Y (Y of X)
   - Example: rāja-putra → rājñaḥ putraḥ (king's son)

2. **Bahuvrihi **: Possessive compound referring to an external entity
   - Test: Can be paraphrased as yasya X asti (one who has X)
   - Example: su-mukha → yasya sundaram mukham asti (one who has a beautiful face)

3. **Dvandva **: Copulative compound coordinating elements with "and"
   - Test: Can be paraphrased as X ca Y ca (X and Y)
   - Example: rāma-kṛṣṇa → rāmaḥ ca kṛṣṇaḥ ca (Rama and Krishna)

4. **Avyayibhava **: Adverbial compound with indeclinable prefix
   - Forms an adverb
   - Example: yathā-śakti → according to ability

**CRITICAL RULES:**

1. **Indexing**: Use 0-based indexing. End index is EXCLUSIVE.
   - [0:2] means words at positions 0 and 1 only
   - [2:5] means words at positions 2, 3, and 4

2. **Nested Compounds**: When compounds contain other compounds:
   - Identify ALL levels (inner and outer)
   - List INNER compounds BEFORE outer compounds
   - Example: If words 1-3 form a compound, and words 0-3 form another, list [1:3] first, then [0:3]

3. **Output Format**: Return ONLY valid JSON with all compounds

---

**EXAMPLE 1: Complex Mixed (Tatpurusha & Bahuvrihi)**

Sentence: "pAda raja upamAH giri nadI vega upamam"

Word indices: 0:pAda 1:raja 2:upamAH 3:giri 4:nadI 5:vega 6:upamam

Total words: 7

Analysis:

Words 0–2 (pAda raja) form Tatpurusha: "dust of feet".
Words 0–3 (pAda raja upamAH) form Bahuvrihi: "one who is like the dust of feet".
Words 3–5 (giri nadI) form Tatpurusha: "mountain river".
Words 3–6 (giri nadI vega) form Tatpurusha: "speed of mountain river".
Words 3–7 (giri nadI vega upamam) form Bahuvrihi: "one representing the speed of a mountain river".

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "Tatpurusha", "text": "pAda raja" },
{ "start": 0, "end": 3, "type": "Bahuvrihi", "text": "pAda raja upamAH" },
{ "start": 3, "end": 5, "type": "Tatpurusha", "text": "giri nadI" },
{ "start": 3, "end": 6, "type": "Tatpurusha", "text": "giri nadI vega" },
{ "start": 3, "end": 7, "type": "Bahuvrihi", "text": "giri nadI vega upamam" }
] }

---

**EXAMPLE 2: Dvandva and Tatpurusha**

Sentence: "SIta Atapa varzAByaH surakzA arTaM vetra kawaH"

Word indices: 0:SIta 1:Atapa 2:varzAByaH 3:surakzA 4:arTaM 5:vetra 6:kawaH

Total words: 7

Analysis:

Words 0–3 (SIta Atapa varzAByaH) form Dvandva: "cold, heat, and rain".
Words 3–5 (surakzA arTaM) form Tatpurusha: "for the purpose of protection".
Words 5–7 (vetra kawaH) form Tatpurusha: "mat made of cane".

Output JSON:

{ "compounds": [
{ "start": 0, "end": 3, "type": "Dvandva", "text": "SIta Atapa varzAByaH" },
{ "start": 3, "end": 5, "type": "Tatpurusha", "text": "surakzA arTaM" },
{ "start": 5, "end": 7, "type": "Tatpurusha", "text": "vetra kawaH" }
] }

---

**EXAMPLE 3: Bahuvrihi with Dvandva List**

Sentence: "sa sarzapaM tumburu DAnya vanyaM"

Word indices: 0:sa 1:sarzapaM 2:tumburu 3:DAnya 4:vanyaM

Total words: 5

Analysis:

Words 0–2 (sa sarzapaM) form Bahuvrihi: "accompanied by mustard" (Saha-Bahuvrihi).
Words 2–5 (tumburu DAnya vanyaM) form Dvandva: "tumburu, grain, and wild-grain".

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "Bahuvrihi", "text": "sa sarzapaM" },
{ "start": 2, "end": 5, "type": "Dvandva", "text": "tumburu DAnya vanyaM" }
] }

---

**EXAMPLE 4:**

Sentence: "tyAga saMnyAsa vikalpe yaTA darSite Barata sattama"

Word indices: 0:tyAga 1:saMnyAsa 2:vikalpe 3:yaTA 4:darSite 5:Barata 6:sattama

Total words: 7

Analysis:

Words 0–2 (tyAga saMnyAsa) form Dvandva: "renunciation and resignation".
Words 0–3 (tyAga saMnyAsa vikalpe) form Tatpurusha: "in the option of renunciation-resignation".
Words 3–5 (yaTA darSite) form Avyayibhava: "as shown".
Words 5–7 (Barata sattama) form Tatpurusha: "best of Bharatas".

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "Dvandva", "text": "tyAga saMnyAsa" },
{ "start": 0, "end": 3, "type": "Tatpurusha", "text": "tyAga saMnyAsa vikalpe" },
{ "start": 3, "end": 5, "type": "Avyayibhava", "text": "yaTA darSite" },
{ "start": 5, "end": 7, "type": "Tatpurusha", "text": "Barata sattama" }
] }

---

**EXAMPLE 5:**

Sentence: "A jIvana kAlam koza pfzWAni"

Word indices: 0:A 1:jIvana 2:kAlam 3:koza 4:pfzWAni

Total words: 5

Analysis:

Words 0–2 (A jIvana) form Avyayibhava: "throughout life / until life".
Words 0–3 (A jIvana kAlam) form Tatpurusha: "time of whole life".
Words 3–5 (koza pfzWAni) form Tatpurusha: "dictionary pages".

Output JSON:

{ "compounds": [
{ "start": 0, "end": 2, "type": "Avyayibhava", "text": "A jIvana" },
{ "start": 0, "end": 3, "type": "Tatpurusha", "text": "A jIvana kAlam" },
{ "start": 3, "end": 5, "type": "Tatpurusha", "text": "koza pfzWAni" }
] }"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the identifier.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable.
            model_name: Gemini model to use (default: gemini-1.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY env variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def parse_conllu_sentence(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        """
        Parse a CoNLL-U sentence to extract words.
        
        Args:
            lines: Lines of CoNLL-U format for one sentence
            
        Returns:
            Tuple of (word_list, form_list) where form_list contains the original forms
        """
        words = []
        forms = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 2:
                continue
            
            # Skip DUMMY nodes
            if fields[1] == 'DUMMY':
                continue
            
            # Extract word form (field 1), remove compound marker if present
            word_form = fields[1]
            # Remove _C2, _C3, _CN markers
            clean_word = re.sub(r'_C[0-9N]$', '', word_form)
            
            words.append(clean_word)
            forms.append(word_form)
        
        return words, forms
    
    def read_conllu_file(self, filepath: str) -> List[List[str]]:
        """
        Read CoNLL-U file and split into sentences.
        
        Args:
            filepath: Path to CoNLL-U file
            
        Returns:
            List of sentence line groups
        """
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
        """
        Create prompt for compound identification.
        
        Args:
            words: List of words in the sentence
            
        Returns:
            Formatted prompt string
        """
        # Create word indices string
        word_indices = " ".join([f"{i}:{word}" for i, word in enumerate(words)])
        sentence = " ".join(words)
        
        user_prompt = f"""
Sentence: "{sentence}"

Word indices: {word_indices}

Total words: {len(words)}

Analyze this sentence and identify ALL compounds, including nested ones. Return ONLY a valid JSON object with the format shown in the examples."""
        
        return user_prompt
    
    def identify_compounds(self, words: List[str], temperature: float = 0.2,
                          max_retries: int = 3) -> Dict:
        """
        Identify compounds in a sentence.
        
        Args:
            words: List of words in the sentence
            temperature: Sampling temperature
            max_retries: Number of retries for JSON parsing errors
            
        Returns:
            Dictionary containing compound analysis
        """
        user_prompt = self.create_prompt(words)
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        
        for attempt in range(max_retries):
            try:
                # Create chat with system instruction
                chat = self.model.start_chat(history=[])
                
                # Send system prompt first
                response = chat.send_message(
                    self.SYSTEM_PROMPT + "\n\n" + user_prompt,
                    generation_config=generation_config
                )
                
                response_text = response.text.strip()
                
                # Clean up response
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                # Parse JSON
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
                        "error": f"Failed to parse JSON after {max_retries} attempts: {e}",
                        "raw_response": response.text
                    }
                # Retry with slightly higher temperature
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
        """
        Process entire CoNLL-U file.

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
        description="Identify Sanskrit compounds from CoNLL-U format using Gemini API"
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
        help="Gemini API key (or set GEMINI_API_KEY env variable)"
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)"
    )
    parser.add_argument(
        "--sentence",
        action="store_true",
        help="Treat input as a sentence string instead of file path"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=None,
        help="Number of sentences to process from the file (default: all)"
    )

    args = parser.parse_args()
    
    identifier = SanskritCompoundIdentifier(api_key=args.api_key, model_name=args.model)
    
    if args.sentence:
        # Process single sentence
        words = args.input.split()
        result = identifier.identify_compounds(words, args.temperature)
        results = [result]
    else:
        # Process CoNLL-U file
        results = identifier.process_conllu_file(args.input, args.temperature, args.num_samples)
    
    # Output results
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