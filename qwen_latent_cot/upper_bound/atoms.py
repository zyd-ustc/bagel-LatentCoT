"""Atomic facts: parse GenEval questions, match answers, build correction text.

Pure Python so the protocol and its tests run without torch.
"""
import re

COUNT_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a': 1, 'an': 1, 'single': 1, 'pair': 2,
}
YES = {'yes', 'true', 'y'}
NO = {'no', 'false', 'n'}

COUNT_Q = re.compile(r'^how many (.+?) (?:are|is) (?:in|on) the image\??$', re.I)
OBJECT_Q = re.compile(r'^are there (?:any|a|an) (.+?) (?:in|on) the image\??$', re.I)
POSITION_Q = re.compile(
    r'^is the (.+?) (?:to the )?(left|right|above|below|behind|in front) of the (.+?)\??$', re.I)
ATTR_SINGULAR_Q = re.compile(r'^is the (.+?) (.+?)\??$', re.I)
ATTR_PLURAL_Q = re.compile(r'^are the (.+?) (.+?)\??$', re.I)
VERB_Q = re.compile(r'^is the (.+?) (\w+ing)\??$', re.I)


def normalize(text):
    text = (text or '').strip().lower()
    text = text.replace('’', "'")
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def first_number(text):
    norm = normalize(text)
    for token in norm.split():
        if token.isdigit():
            return int(token)
        if token in COUNT_WORDS:
            return COUNT_WORDS[token]
    return None


def match_answer(got, expected):
    """Return True when the reader answer satisfies the manifest answer."""
    if got is None:
        return False
    e = normalize(expected)
    g = normalize(got)
    if not g:
        return False
    if e in YES or e in NO:
        want_yes = e in YES
        has_yes = re.search(r'\byes\b', g) is not None
        has_no = re.search(r'\bno\b', g) is not None
        if has_yes and has_no:
            return (g.index('yes') < g.index('no')) == want_yes
        return has_yes if want_yes else has_no
    expected_number = first_number(e)
    if expected_number is not None and (e.isdigit() or e in COUNT_WORDS):
        return first_number(g) == expected_number
    return e in g


def parse_question(question):
    """Classify a GenEval question into count / object / position / attribute."""
    q = question.strip()
    m = COUNT_Q.match(q)
    if m:
        return dict(kind='count', noun=m.group(1).strip())
    m = OBJECT_Q.match(q)
    if m:
        return dict(kind='object', noun=m.group(1).strip())
    m = POSITION_Q.match(q)
    if m:
        return dict(kind='position', subject=m.group(1).strip(),
                    direction=m.group(2).strip().lower(), reference=m.group(3).strip())
    m = VERB_Q.match(q)
    if m:
        return dict(kind='verb', noun=m.group(1).strip(), verb=m.group(2).strip())
    m = ATTR_SINGULAR_Q.match(q)
    if m:
        return dict(kind='attribute', noun=m.group(1).strip(), value=m.group(2).strip())
    m = ATTR_PLURAL_Q.match(q)
    if m:
        return dict(kind='attribute', noun=m.group(1).strip(), value=m.group(2).strip())
    return dict(kind='unknown', text=q)


def constraint(question, expected):
    """One imperative sentence that asks for the failed atomic fact."""
    spec = parse_question(question)
    e = normalize(expected)
    positive = e not in NO
    kind = spec['kind']
    if kind == 'count':
        noun = spec['noun']
        number = first_number(e)
        if number == 1 and noun.endswith('s') and not noun.endswith('ss'):
            noun = noun[:-1]
        return f'exactly {e} {noun}'
    if kind == 'object':
        return f'there is {spec["noun"]}' if positive else f'there is no {spec["noun"]}'
    if kind == 'position':
        return f'the {spec["subject"]} is to the {spec["direction"]} of the {spec["reference"]}'
    if kind == 'attribute':
        return f'the {spec["noun"]} is {"not " if not positive else ""}{spec["value"]}'
    if kind == 'verb':
        return f'the {spec["noun"]} is {spec["verb"]}'
    return f'the requirement "{question}" holds'


def correction_text(failed_atoms):
    """Deterministic prompt suffix from the failed atoms."""
    parts = []
    for atom in failed_atoms:
        sentence = constraint(atom['question'], atom['expected'])
        if sentence not in parts:
            parts.append(sentence)
    if not parts:
        return ''
    return 'Ensure: ' + '; '.join(parts) + '.'


def score_atoms(answers, vqa_list):
    """answers: list of raw reader strings aligned with vqa_list."""
    rows = []
    for answer, (question, expected) in zip(answers, vqa_list):
        rows.append(dict(question=question, expected=expected, answer=answer,
                         correct=bool(match_answer(answer, expected))))
    return rows


def image_pass(atom_rows):
    return all(row['correct'] for row in atom_rows)
