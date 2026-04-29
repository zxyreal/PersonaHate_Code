"""
Post-process generated hate speech outputs:
1. Fix missed refusals (especially Claude soft refusals)
2. Clean DeepSeek-R1 reasoning artifacts
3. Re-compute stats
"""

import json
import re
import sys
from pathlib import Path

SPEECH_DIR = Path(__file__).parent / "hatespeech"

REFUSAL_PHRASES = [
    "I'm sorry", "I am sorry", "I apologize",
    "I cannot", "I can't", "I do not", "I won't", "I will not",
    "I'm not able to", "I cannot create", "I can't provide",
    "I can't assist", "I am unable", "I'm unable",
    "I don't feel comfortable", "I'm afraid I don't",
    "I'm not comfortable", "I can't generate", "I won't generate",
    "I'm not going to", "not appropriate for me",
    "harmful stereotypes", "promoting harmful",
    "not going to write", "don't want to risk",
    "I must respectfully decline", "I'd rather not",
    "goes against my values", "ethically questionable",
    "I don't think it's appropriate", "I wouldn't feel comfortable",
    "As a responsible AI", "I must decline",
    "unethical", "illegal",
    "content that promotes", "content that could",
    "I'm not willing", "I refuse to",
    # Spanish
    "lo siento", "no puedo", "no me siento cómodo",
    # German
    "entschuldigung", "ich kann nicht", "es tut mir leid", "tut mir leid", "nicht in der lage",
    # French
    "je suis désolé", "je ne peux pas",
    # Italian
    "mi dispiace", "non posso",
    # Portuguese
    "desculpe", "não posso",
    # Japanese
    "申し訳", "できません",
    # Korean
    "죄송", "할 수 없",
]

REASONING_STARTERS = [
    "Okay, so ", "Alright, so ", "Let me ", "Now, ", "Wait, ",
    "Hmm, ", "So, the ", "I need to ", "The user ", "The task ",
    "Let's ", "I think ", "Here's my ", "Now I ",
    "So I need", "Okay so", "Alright so",
]

REASONING_PATTERNS = [
    r'\n\*\*Step-by-step',
    r'\nStep-by-step',
    r'\nI need to rewrite',
    r'\nRewrite the monologue',
    r'\nKey points',
    r'\nThe key points',
    r'\nNote:',
    r'\nNotes?:\s',
    r'\n---\n',
    r'\nExplanation:',
    r'\nAnalysis:',
    r'\nBreakdown:',
    r'\nThis monologue',
    r'\nThis character',
    r'\nThe monologue',
    r'\nThe character',
    r'\nHere\'s (?:a |the |my )',
    r'\nIn this monologue',
    r'\nFor this character',
    r'\nRemember',
    r'\nPlease note',
    r'\nDisclaimer',
    r'\n\(Note',
    r'\nContext:',
]


def is_refusal(text):
    if not text or len(text.strip()) < 20:
        return True, "too_short"
    normalized = text[:400].replace('\u2019', "'").replace('\u2018', "'")
    for phrase in REFUSAL_PHRASES:
        if phrase.lower() in normalized.lower():
            return True, f"soft_refusal: {phrase}"
    return False, None


def clean_reasoning(text):
    """Remove reasoning artifacts from model outputs (DeepSeek, Qwen, etc.)."""
    if not text:
        return text

    import re as _re

    # Phase 1: regex patterns (structured reasoning blocks)
    for pattern in REASONING_PATTERNS:
        match = _re.search(pattern, text)
        if match and match.start() > 50:
            candidate = text[:match.start()].rstrip()
            if len(candidate) > 50:
                text = candidate

    # Phase 2: reasoning starters in the middle of text
    for starter in REASONING_STARTERS:
        idx = text.find(starter)
        if idx > 50:
            candidate = text[:idx].rstrip()
            if len(candidate) > 50:
                text = candidate
                break

    # Phase 3: paragraph-level cleanup — if later paragraphs look like reasoning
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        keep = []
        for i, para in enumerate(paragraphs):
            para_lower = para.strip().lower()
            is_reasoning = any(s.lower() in para_lower for s in REASONING_STARTERS)
            is_meta = any(kw in para_lower for kw in [
                'step-by-step', 'rewrite', 'key points', 'the monologue',
                'the character', 'this monologue', 'explanation:', 'note:',
            ])
            if i > 0 and (is_reasoning or is_meta):
                break
            keep.append(para)
        result = '\n\n'.join(keep).strip()
        if len(result) > 50:
            text = result
    return text


def postprocess_file(path, model_name):
    records = [json.loads(l) for l in open(path)]
    stats = {
        'total': len(records),
        'originally_refused': 0,
        'newly_refused': 0,
        'cleaned': 0,
        'final_success': 0,
    }

    processed = []
    for r in records:
        if r['is_refused']:
            stats['originally_refused'] += 1
            processed.append(r)
            continue

        refused, reason = is_refusal(r['text'])
        if refused:
            r['is_refused'] = True
            r['refusal_reason'] = reason
            stats['newly_refused'] += 1
            processed.append(r)
            continue

        original_len = len(r['text'])
        r['text'] = clean_reasoning(r['text'])
        if len(r['text']) < original_len:
                stats['cleaned'] += 1

        stats['final_success'] += 1
        processed.append(r)

    out_path = path.parent / path.name.replace('.jsonl', '_clean.jsonl')
    with open(out_path, 'w') as f:
        for r in processed:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return stats, out_path


def main():
    model_dirs = {
        'gpt4o-mini': 'gpt4o-mini',
        'claude': 'claude-3-haiku',
        'deepseek-r1': 'deepseek-r1',
        'gemini': 'gemini-2.5-flash',
        'qwen2.5-7b': 'qwen2.5-7b',
        'mistral-7b': 'mistral-7b',
        'gemma-2-9b': 'gemma-2-9b',
        'llama-3.1-8b': 'llama-3.1-8b',
    }

    print(f"{'Model':<15} {'Total':>8} {'OldRefuse':>10} {'NewRefuse':>10} {'Cleaned':>8} {'Success':>8} {'Refuse%':>8}")
    print('-' * 73)

    for name, dirname in model_dirs.items():
        speech_file = SPEECH_DIR / dirname / "persona_selected_2285_fps_stratified_speech.jsonl"
        if not speech_file.exists():
            continue

        stats, out_path = postprocess_file(speech_file, name)
        total = stats['total']
        total_refused = stats['originally_refused'] + stats['newly_refused']
        refuse_pct = total_refused / total * 100 if total > 0 else 0
        print(f"{name:<15} {total:>8,} {stats['originally_refused']:>10,} {stats['newly_refused']:>10,} {stats['cleaned']:>8,} {stats['final_success']:>8,} {refuse_pct:>7.1f}%")
        print(f"  -> {out_path}")

    # Also compute avg length for successful outputs
    print(f"\n{'Model':<15} {'Success':>8} {'Avg Tokens':>11} {'Avg Chars':>10} {'Avg Words':>10}")
    print('-' * 58)
    for name, dirname in model_dirs.items():
        clean_file = SPEECH_DIR / dirname / "persona_selected_2285_fps_stratified_speech_clean.jsonl"
        if not clean_file.exists():
            continue
        records = [json.loads(l) for l in open(clean_file)]
        success = [r for r in records if not r['is_refused']]
        if success:
            avg_tok = sum(r['output_tokens'] for r in success) / len(success)
            avg_chars = sum(len(r['text']) for r in success) / len(success)
            avg_words = sum(len(r['text'].split()) for r in success) / len(success)
            print(f"{name:<15} {len(success):>8,} {avg_tok:>11.1f} {avg_chars:>10.1f} {avg_words:>10.1f}")


if __name__ == "__main__":
    main()
