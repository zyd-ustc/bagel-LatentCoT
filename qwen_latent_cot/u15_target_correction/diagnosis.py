"""Frozen UND question decomposition, visual scoring, and correction conditions."""

import math
import re
import torch
from torch.nn import functional as F
from .runtime import autocast


DECOMPOSE_RULES = (
    "Use only directly visible subjects, properties, props and setting. Replace invisible "
    "professional or family identities with 'person'; a person without a defining uniform "
    "or prop must never be called an astronomer, doctor or brother in a question. "
    "Check identifying props and setting separately.\n"
    "Every question must be self-contained: an action or spatial relation must be false "
    "if either participating subject or required prop is absent.\n"
    "Never generate mutually contradictory or duplicate questions. Preserve the exact "
    "direction of every relation in the request: older than, behind, blocking, and so on. "
    "Do not ask the reversed relation.\n"
    "Ask an exact count ONLY when the request explicitly gives a number or a singular "
    "article for that subject. For a bare plural such as 'acrobats', ask 'two or more', "
    "never invent an exact number.\n"
    "Do not ask about position, background, setting, or an action unless the user request "
    "itself contains that spatial preposition, setting, or action verb. Never invent "
    "behind, in front of, left, right, on, under, a plate, a wall, or a background.\n"
    "Do not combine text spelling with placement or other properties in one question. "
    "Dedicated exact-spelling questions will be appended separately."
)

DECOMPOSE_SYSTEM_V2 = (
    "Decompose the user's image request into 3 to 5 atomic scene-verification questions "
    "answerable Yes or No from a single still image. Phrase each so Yes means that the "
    "request is satisfied. Cover subject existence, permitted count constraints, key "
    "attributes and props, setting, actions, and the requested spatial or occlusion "
    "relations.\nStrict rules:\n"
    + DECOMPOSE_RULES
    + "\nOutput ONLY questions, one per line, starting each line with 'Q: '."
)


QUESTION_SYSTEM = DECOMPOSE_SYSTEM_V2.replace("3 to 5", "3 to 4") + (
    "\nKeep at most four questions. Relation questions are allowed only when the request "
    "contains a spatial preposition or an action verb. "
    "For unnumbered plural subjects use two or more, never an invented exact count."
)

def clean_questions(raw, body):
    questions = []
    edits = []
    relations = {}
    core = 0
    separation = False
    for line in re.split(r"\r?\n|(?<=[?？])\s*(?=Q\d*[:：])", body):
        q = re.sub(r"^\s*(?:Q\d*[:：.]|-|\d+[.)])\s*", "", line).strip()
        if not q or not q.rstrip("\"'”’)） ").endswith(("?", "？")):
            continue
        # Only rewrite explicit counts of a plural noun actually present bare in raw.
        for m in list(
            re.finditer(
                r"\b(?:exactly\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+([a-z]+s)\b",
                q,
                re.I,
            )
        ):
            number, noun = m.groups()
            if re.search(r"\b" + re.escape(noun) + r"\b", raw, re.I) and not re.search(
                r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                + re.escape(noun)
                + r"\b",
                raw,
                re.I,
            ):
                revised = q.replace(m.group(0), "two or more " + noun)
                edits.append(
                    dict(original=q, replacement=revised, reason="unnumbered_plural")
                )
                q = revised
        normalized = re.sub(r"\s+", " ", q).casefold()
        if any(normalized == re.sub(r"\s+", " ", x).casefold() for x in questions):
            edits.append(dict(original=q, reason="duplicate"))
            continue
        rel = re.fullmatch(
            r"Is (.+?) (older than|younger than|behind|in front of|blocking the view of) (.+?)\?",
            q,
            re.I,
        )
        if rel:
            left, relation, right = rel.groups()
            normalize = lambda x: re.sub(r"^(?:a|an|the)\s+", "", x.strip().casefold())
            left, right = normalize(left), normalize(right)
            relation = relation.casefold()
            inverse = {
                "older than": "younger than",
                "younger than": "older than",
                "behind": "in front of",
                "in front of": "behind",
            }
            if (right, relation, left) in relations or (
                left,
                inverse.get(relation, ""),
                right,
            ) in relations:
                edits.append(
                    dict(original=q, reason="opposed_relation_to_prior_question")
                )
                continue
            relations[(left, relation, right)] = q
        is_separation = bool(
            re.search(
                r"fus(?:ed|ion)|physically distinct|separate (?:objects|entities)",
                q,
                re.I,
            )
        )
        if is_separation:
            if separation:
                continue
            separation = True
        else:
            if core >= 4:
                continue
            core += 1
        questions.append(q)
    for match in re.finditer(
        r'(?<!\w)(?:"([^"]{2,})"|\x27([^\x27]{2,})\x27|“([^”]{2,})”|‘([^’]{2,})’)', raw
    ):
        phrase = next(x for x in match.groups() if x is not None)
        q = f'Is the exact text "{phrase}" clearly legible and spelled accurately without garbled letters?'
        if q not in questions:
            questions.append(q)
    if not questions:
        raise ValueError("No parseable questions: " + body)
    return questions, edits

@torch.no_grad()
def decompose(rt, raw, cfg):
    result = semantic_plan(
        rt,
        raw,
        cfg,
        system_message=QUESTION_SYSTEM,
        question="USER REQUEST:\n" + raw + "\nFINAL RULES:\n" + DECOMPOSE_RULES,
    )
    questions, edits = clean_questions(raw, result["plan"])
    questions, dropped = drop_ungrounded(raw, questions)
    if dropped:
        edits.append(dict(dropped=dropped, reason="not_in_raw_prompt"))
    return questions, dict(response=result["response"], edits=edits)


@torch.no_grad()
def semantic_plan(rt, raw, config, *, system_message=None, question):
    """Text-only UND planning; identical information is available at deployment."""
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    model, tokenizer = rt.model, rt.tokenizer
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    query = model._build_t2i_query(
        question, system_message=system_message, append_text="<think>\n\n</think>\n\n"
    )
    inputs = tokenizer(query, return_tensors="pt")
    separator = get_conv_template(model.template).sep.strip()
    eos = tokenizer.convert_tokens_to_ids(separator)
    with autocast(rt.device):
        output = model.generate(
            input_ids=inputs["input_ids"].to(rt.device),
            attention_mask=inputs["attention_mask"].to(rt.device),
            max_new_tokens=config["und_max_new_tokens"],
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    response = (
        tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        .split(separator)[0]
        .strip()
    )
    plan = response.rsplit("</think>", 1)[-1].strip()
    if not plan or "<think>" in plan:
        raise ValueError("UND did not finish its semantic plan: " + response)
    return dict(raw_prompt=raw, response=response, plan=plan)


YES_VARIANTS = ("Yes", "yes", "YES", " Yes", " yes")

NO_VARIANTS = ("No", "no", "NO", " No", " no")

def _single_token_ids(tokenizer, candidates):
    ids = []
    for word in candidates:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    if not ids:
        raise ValueError(f"No single-token encoding found for {candidates}")
    return ids


def _image_inputs(rt, image):
    """The image tag and understanding-side pixels for one protocol canvas."""
    m, tok = rt.model, rt.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = ((image.detach().float().clamp(-1, 1) + 1) / 2 - mean) / std
    gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
    grid = torch.tensor([[gh, gw]], device=rt.device)
    pixels = (
        m.patchify(rgb, m.patch_size, channel_first=True)
        .reshape(gh * gw, -1)
        .to(m.vision_model.embeddings.patch_embedding.weight.dtype)
    )
    tag = "<img>" + "<IMG_CONTEXT>" * rt.tokens + "</img>"
    return grid, pixels, tag

@torch.no_grad()
def und_log_reward(rt, image, questions, yes_ids=None, no_ids=None):
    """Single-pass prefill Yes/No log-sigmoid conjunction score in (-inf, 0]."""
    m, tok = rt.model, rt.tokenizer
    yes_ids = yes_ids or _single_token_ids(tok, YES_VARIANTS)
    no_ids = no_ids or _single_token_ids(tok, NO_VARIANTS)
    grid, pixels, img_tag = _image_inputs(rt, image)

    with autocast(rt.device):
        # One understanding-side ViT pass per preview image.
        vit_embeds = m.extract_feature(pixels, gen_model=False, grid_hw=grid)
        embed_tokens = m.language_model.get_input_embeddings()
        per_q = []
        for q in questions:
            prompt = (
                f"{img_tag}\nInspect the image carefully and answer strictly Yes or No.\n"
                f"Question: {q}"
            )
            query = m._build_t2i_query(prompt, append_text="<think>\n\n</think>\n\n")
            enc = tok(query, return_tensors="pt")
            input_ids = enc["input_ids"].to(rt.device)
            attn_mask = enc["attention_mask"].to(rt.device)
            indexes = m.get_thw_indexes(input_ids[0], grid)

            flat_ids = input_ids.reshape(-1)
            embeds = embed_tokens(input_ids).reshape(-1, vit_embeds.shape[-1]).clone()
            selected = flat_ids == m.img_context_token_id
            embeds[selected] = vit_embeds.reshape(-1, embeds.shape[-1]).to(embeds.dtype)
            embeds = embeds.unsqueeze(0)

            out = m.language_model(
                inputs_embeds=embeds,
                indexes=indexes,
                attention_mask=attn_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            last_logits = out.logits[0, -1].float()
            l_yes = torch.logsumexp(last_logits[yes_ids], dim=0)
            l_no = torch.logsumexp(last_logits[no_ids], dim=0)
            margin = l_yes - l_no
            per_q.append(
                dict(
                    margin=float(margin),
                    p_yes=float(torch.sigmoid(margin)),
                    log_p_yes=float(F.logsigmoid(margin)),
                )
            )

    log_r = sum(item["log_p_yes"] for item in per_q) / len(per_q)
    return log_r, per_q


def summarize_question_scores(per_q):
    logs = [q["log_p_yes"] for q in per_q]
    mean = sum(logs) / len(logs)
    worst = min(logs)
    return dict(
        log_r_mean=mean,
        min_log_p=worst,
        min_p_yes=min(q["p_yes"] for q in per_q),
        r_conj=0.5 * mean + 0.5 * worst,
        questions=per_q,
    )

@torch.no_grad()
def ask_with_image(rt, image, question, cfg, system_message=None, max_new_tokens=None):
    """Text answer conditioned on one image; same canvas and normalization as scoring."""
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    m, tok = rt.model, rt.tokenizer
    grid, pixels, img_tag = _image_inputs(rt, image)
    query = m._build_t2i_query(
        img_tag + "\n" + question,
        system_message=system_message,
        append_text="<think>\n\n</think>\n\n",
    )
    encoded = tok(query, return_tensors="pt")
    separator = get_conv_template(m.template).sep.strip()
    eos = tok.convert_tokens_to_ids(separator)
    with autocast(rt.device):
        vit = m.extract_feature(pixels, gen_model=False, grid_hw=grid)
        output = m.generate(
            pixel_values=pixels,
            input_ids=encoded["input_ids"].to(rt.device),
            grid_hw=grid,
            attention_mask=encoded["attention_mask"].to(rt.device),
            visual_features=vit,
            max_new_tokens=int(max_new_tokens or cfg["und_max_new_tokens"]),
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    text = tok.batch_decode(output, skip_special_tokens=True)[0].split(separator)[0].strip()
    return text.rsplit("</think>", 1)[-1].strip()


@torch.no_grad()
def und_answer_image(rt, image, questions, cfg, max_new_tokens=6):
    """Ask each question and read the first word of the reply.

    ``und_log_reward`` reads the logits after an empty ``<think>`` block, which is
    where generation would start thinking rather than answering. On the pooled
    text-to-image images it returned p_yes = 0.000 for questions the model had
    itself just answered as true, so scoring now uses the generated answer, the
    same interface ``ask_with_image`` already uses for describing.
    """
    per_q = []
    for question in questions:
        answer = ask_with_image(
            rt, image,
            "Look at the image and answer with Yes or No only.\nQuestion: " + question,
            cfg, max_new_tokens=max_new_tokens,
        )
        head = (answer or "").strip().lower()[:12]
        if head.startswith("yes"):
            p_yes, label = 0.99, "yes"
        elif head.startswith("no"):
            p_yes, label = 0.01, "no"
        else:
            p_yes, label = None, "unparsed"
        per_q.append(dict(p_yes=p_yes, label=label, answer=(answer or "")[:40]))
    return per_q


@torch.no_grad()
def und_score_image(rt, image, questions, cfg=None, yes_ids=None, no_ids=None):
    """Per-question scores. Uses the generated answer when a config is supplied.

    ``und_log_reward`` stays available for callers that need the prefill margins,
    but it is no longer what decides acceptance: on the pooled text-to-image
    images it reported p_yes = 0.000 for facts the model had just described.
    """
    if cfg is not None:
        per = und_answer_image(rt, image, questions, cfg)
        answered = [q["p_yes"] for q in per if q["p_yes"] is not None]
        return dict(questions=per, min_p_yes=min(answered) if answered else None)
    _, scores = und_log_reward(rt, image, questions, yes_ids, no_ids)
    return summarize_question_scores(scores)


_SPATIAL_Q = re.compile(
    r"\b(behind|in front of|to the left of|to the right of|left of|right of|under|underneath|"
    r"above|below|on top of|next to|beside|between|inside|near|foreground|background|"
    r"located on|positioned|on the)\b",
    re.I,
)
_SPATIAL_PROMPT = re.compile(
    r"\b(behind|in front of|to the left of|to the right of|left of|right of|under|above|below|"
    r"on top of|next to|beside|between|inside|near|over|on)\b",
    re.I,
)
_ACTION = re.compile(
    r"\b(playing with|chasing|jumping over|holding|riding|wearing|carrying|sitting|standing|"
    r"running|flying|eating|looking at)\b",
    re.I,
)
_SETTING_Q = re.compile(
    r"\b(setting|background|environment|surface|plate|sky|outdoor|stone wall|tree)\b",
    re.I,
)
_COLOR = (
    "red|green|blue|yellow|black|white|brown|pink|purple|orange|grey|gray|"
    "metal|metallic|wooden|stone|plastic|spotted|striped"
)
_COUNT_WORD = {
    "a": "one", "an": "one", "one": "one", "two": "two", "three": "three", "four": "four",
    "five": "five", "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "ten": "ten",
}
_RELATIONS = ("in front of", "behind", "on top of", "next to", "beside", "under", "above")
_RELATION_WORDS = (
    "in front of", "to the left of", "to the right of", "on top of",
    "next to", "beside", "behind", "under", "above", "below",
    "left of", "right of", "on",
)
_RELATION_ALTERNATION = "|".join(re.escape(word) for word in _RELATION_WORDS)
_RELATION_INVERSE = {
    "behind": "in front of",
    "in front of": "behind",
    "to the left of": "to the right of",
    "to the right of": "to the left of",
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
    "on top of": "under",
    "under": "on top of",
}
_COUNT_NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _same_subject(left, right):
    left, right = left.lower().strip(), right.lower().strip()
    return left.rstrip("s") == right.rstrip("s") or left in right or right in left


def _clean_entity(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip(" .,")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    text = re.sub(
        r"\s+(?:visible|present|shown|in the (?:scene|image))\b.*$", "", text, flags=re.I
    )
    text = re.sub(r"\b(?:is|are|was|were|sits?|stands?|located|positioned)\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" .,")


def _relation_claim(question):
    text = question.strip().rstrip("?？")
    match = re.match(
        r"(?:Is|Are)\s+(?:the\s+)?(.+?)\s+(?:positioned\s+|located\s+)?"
        r"(" + _RELATION_ALTERNATION + r")\s+(?:the\s+)?(.+?)$",
        text,
        re.I,
    )
    if not match:
        match = re.match(
            r"(?:Is|Are) there\s+(?:(?:a|an|the)\s+)?(.+?)\s+"
            r"(" + _RELATION_ALTERNATION + r")\s+(?:(?:a|an|the)\s+)?(.+?)$",
            text,
            re.I,
        )
    if not match:
        return None
    return _clean_entity(match.group(1)), match.group(2).lower(), _clean_entity(match.group(3))


def raw_relations(raw):
    """Extract explicit relation triples from the source prompt."""
    text = re.sub(r"\s+", " ", str(raw or "").strip().rstrip("."))
    pattern = re.compile(
        r"(?:(?:the|a|an)\s+)?([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,4}?)\s+"
        r"(?:is\s+|are\s+|was\s+|were\s+)?(" + _RELATION_ALTERNATION + r")\s+"
        r"(?:(?:the|a|an)\s+)?([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,4}?)"
        r"(?=\s*(?:[,.;]| and |$))",
        re.I,
    )
    return [
        (_clean_entity(match.group(1)), match.group(2).lower(), _clean_entity(match.group(3)))
        for match in pattern.finditer(text)
    ]


def _count_claim(question):
    text = re.sub(r"\s+", " ", question.strip().rstrip("?？"))
    exact = bool(re.search(r"\bexactly\b", text, re.I))
    lower = bool(re.search(r"\bor more\b", text, re.I))
    match = re.match(
        r"Are there\s+(?:exactly\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
        r"(?:\s+or more)?\s+(.+?)(?:\s+(?:visible|present|shown)\b.*)?$",
        text,
        re.I,
    )
    if match:
        kind = "exact" if exact else "at_least" if lower else "exact"
        subject = _clean_entity(match.group(2))
        return kind, int(_COUNT_NUMBER.get(match.group(1).lower(), match.group(1))), subject
    exists = re.match(
        r"(?:Is|Are) there\s+(a|an|any)\s+(.+?)(?:\s+(?:visible|present|shown)\b.*)?$",
        text,
        re.I,
    )
    if exists:
        return "exists", 1, _clean_entity(exists.group(2))
    return None


def _raw_mentions(raw, subject):
    words = [word for word in re.findall(r"[a-z][a-z\-]*", subject.lower()) if word not in {"green", "red", "blue", "black", "white", "brown", "yellow", "pink", "purple", "orange", "grey", "gray"}]
    if not words:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in words[-2:]) + r"s?\b"
    return bool(re.search(pattern, raw, re.I))


def parse_entities(raw):
    """Bind each count, color, and material to the noun phrase that carries it."""
    text = re.sub(r"\s+", " ", raw.strip().rstrip("."))
    pieces = re.split(r"\s+\b(?:" + "|".join(_RELATIONS) + r")\b\s+", text, flags=re.I)
    entities = []
    for piece in pieces:
        for part in re.split(r"\s*(?:,| and )\s*", piece):
            match = re.match(
                r"(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(.+)$",
                part.strip(),
                re.I,
            )
            if not match:
                continue
            words = match.group(2).split()
            color = material = None
            if words and re.fullmatch(_COLOR, words[0], re.I):
                token = words[0].lower()
                if token in {"metal", "metallic", "wooden", "stone", "plastic"}:
                    material = token
                else:
                    color = token
                words = words[1:]
            if not words:
                continue
            entities.append({
                "subject": " ".join(words),
                "count": _COUNT_WORD[match.group(1).lower()],
                "color": color,
                "material": material,
            })
    return entities


def _color_claim(question):
    match = re.match(
        rf"(?:Is|Are) (?:each |every |the )?(?!there\b)(.+?) ({_COLOR})(?: in color)?$",
        question.strip().rstrip("?？"),
        re.I,
    )
    if not match:
        return None
    return match.group(1).strip(), match.group(2).lower()


def question_grounded(raw, question):
    """Bind relations and counts to the entities and direction in the raw prompt."""
    relation = _relation_claim(question)
    if relation is not None:
        left, relation_name, right = relation
        licensed = False
        for raw_left, raw_relation, raw_right in raw_relations(raw):
            if (
                _same_subject(left, raw_left)
                and _same_subject(right, raw_right)
                and relation_name == raw_relation
            ):
                licensed = True
                break
            if (
                _same_subject(left, raw_right)
                and _same_subject(right, raw_left)
                and relation_name == _RELATION_INVERSE.get(raw_relation)
            ):
                licensed = True
                break
        if not licensed:
            return False
    elif _SPATIAL_Q.search(question) and not _SPATIAL_PROMPT.search(raw):
        return False

    count = _count_claim(question)
    if count is not None:
        kind, number, subject = count
        matching = [
            entity for entity in parse_entities(raw)
            if _same_subject(subject, entity["subject"])
        ]
        if kind == "exact":
            if not matching or not any(
                int(_COUNT_NUMBER.get(entity["count"], -1)) == number for entity in matching
            ):
                return False
        elif kind == "at_least":
            if matching and not any(
                int(_COUNT_NUMBER.get(entity["count"], 0)) >= number for entity in matching
            ):
                return False
            if not matching and not _raw_mentions(raw, subject):
                return False
        elif not matching and not _raw_mentions(raw, subject):
            return False

    if _ACTION.search(question) and not _ACTION.search(raw):
        return False
    if _SETTING_Q.search(question) and not _SETTING_Q.search(raw):
        return False
    if re.search(r"\b(person|people|man|woman|boy|girl|child)\b", question, re.I) and not re.search(
        r"\b(person|people|man|woman|boy|girl|child)\b", raw, re.I
    ):
        return False
    claim = _color_claim(question)
    if claim is not None:
        subject, color = claim
        owners = [
            entity["subject"] for entity in parse_entities(raw)
            if entity.get("color") == color or entity.get("material") == color
        ]
        if not any(_same_subject(subject, owner) for owner in owners):
            return False
    return True


def drop_ungrounded(raw, questions):
    kept, dropped = [], []
    for question in questions:
        (kept if question_grounded(raw, question) else dropped).append(question)
    return kept, dropped


def _imperative(question):
    text = question.strip().rstrip("?？")
    if re.match(r"(?:Is|Are) the .+ an? ", text, re.I):
        return ""
    count = re.search(r"exactly\s+(\w+)\s+(.+?)\s+visible", text, re.I)
    if count:
        noun = re.sub(r"\s+", " ", count.group(2)).strip()
        return f"Change the number of {noun} to exactly {count.group(1)}."
    relation = re.match(
        r"(?:Is|Are) the (.+?) (?:positioned |located )?(behind|in front of|to the left of|"
        r"to the right of|under|above|on top of|next to|beside) the (.+)",
        text,
        re.I,
    )
    if relation:
        subject, rel, obj = relation.groups()
        return f"Put the {subject} {rel.lower()} the {obj}."
    action = re.match(
        r"(?:Is|Are) the (.+?) (playing with|chasing|jumping over|holding|riding|wearing|carrying) the (.+)",
        text,
        re.I,
    )
    if action:
        return f"Make the {action.group(1)} {action.group(2).lower()} the {action.group(3)}."
    existence = re.match(
        r"(?:Is|Are) there (a|an|any) (.+?)(?: present| visible)?(?: in the (?:scene|image))?$",
        text,
        re.I,
    )
    if existence:
        article = "" if existence.group(1).lower() == "any" else existence.group(1).lower() + " "
        return f"Add {article}{existence.group(2).strip()}."
    if re.search(r"\b(distinct|fused)\b", text, re.I):
        return "Separate the subjects so they are not fused."
    claim = _color_claim(text)
    if claim:
        return f"Make the {claim[0]} {claim[1]}."
    return ""


def _attribute_keep(question):
    match = re.match(
        rf"(?:Is|Are) (?:each |every |the )?(.+?) ({_COLOR})(?: in color)?$",
        question.strip().rstrip("?？"),
        re.I,
    )
    if not match:
        return ""
    noun = match.group(1).strip()
    if noun.lower().startswith("there "):
        return ""
    return f"Keep the {noun} {match.group(2).lower()}."


_UNACTIONABLE = re.compile(
    r"\b(other|besides|only subjects?|without any occlusion|partially obscured)\b", re.I
)


def feedback_from_questions(questions, failed):
    """Split a diagnosis into an edit request, a preserve list, and rejections.

    The request is the imperative form of failed questions. Preserve only
    carries confirmed attributes that the image already satisfies. A failure
    without a concrete imperative form is rejected rather than smuggled into
    the edit.
    """
    failed_set = set(failed).intersection(questions)
    changes, rejected = [], []
    for question in questions:
        if question not in failed_set:
            continue
        if _UNACTIONABLE.search(question):
            rejected.append(question)
            continue
        line = _imperative(question)
        if line:
            changes.append(line)
        else:
            rejected.append(question)
    keeps = []
    for question in questions:
        if question in failed_set:
            continue
        keep = _attribute_keep(question)
        if keep and keep not in keeps:
            keeps.append(keep)
    return " ".join(changes), " ".join(keeps), rejected


def classify_question(question):
    text = question.lower()
    # Binary relations take precedence over an embedded existence phrase.
    if re.search(
        r"behind|in front of|next to|near\b|older than|younger than|positioned|"
        r"located|parked|balanced|blocking|whispering|physically distinct|"
        r"coherent entities|fused|morphing|stitching|slicing|mounted|filled|"
        r"位于|前面|后面|旁边|安装|装满|连接|遮挡|独立|融合|清灰|架柴|引燃",
        text,
    ):
        return "relation"
    if re.match(r"^(is|are) there ", text) or re.search(
        r"\b(present|visible|shown|exists?)\b|是否(?:包含|有|存在)", text
    ):
        return "entity"
    return "other"

def entailment_cap(score, questions):
    """Cap relation evidence by the weakest explicit entity-existence question."""
    classes = [classify_question(q) for q in questions]
    entities = [i for i, kind in enumerate(classes) if kind == "entity"]
    relations = [i for i, kind in enumerate(classes) if kind == "relation"]
    per_q = [dict(q) for q in score["questions"]]
    if entities:
        cap = min(per_q[i]["log_p_yes"] for i in entities)
        for i in relations:
            q = per_q[i]
            q.update(
                raw_p_yes=q["p_yes"],
                raw_log_p_yes=q["log_p_yes"],
                existence_dependencies=entities,
            )
            q["log_p_yes"] = min(q["log_p_yes"], cap)
            q["p_yes"] = math.exp(q["log_p_yes"])
    return dict(
        summarize_question_scores(per_q),
        raw_score=score,
        entity_indices=entities,
        relation_indices=relations,
    )


@torch.no_grad()
def review_preview(rt, image, raw, cfg):
    """Decompose the request, score a preview, and turn failures into an edit."""
    questions, decomposition = decompose(rt, raw, cfg)
    preview_score = (
        entailment_cap(und_score_image(rt, image, questions), questions)
        if questions
        else {"questions": []}
    )
    failed = [
        question
        for question, score in zip(questions, preview_score["questions"])
        if score["p_yes"] < float(cfg["pass_threshold"])
    ]
    edit_request, preserve, rejected = feedback_from_questions(questions, failed)
    return dict(
        questions=questions,
        decomposition=decomposition,
        preview_score=preview_score,
        failed=failed,
        edit_request=edit_request,
        preserve=preserve,
        rejected=rejected,
    )


def review_atomic_edit(rt, image, raw, cfg):
    """Propose one visible correction using the training instruction format."""
    from .edit_request import SCHEMA, parse_edit

    prompt = (
        "Inspect this draft against the original prompt. Identify at most one visible "
        "violation. Use the target quantity, spatial relation, or attribute from the "
        "prompt. If an object is present but has the wrong attribute, request an "
        "attribute change. Treat ambiguous or already satisfied requirements as none. "
        + SCHEMA + "\nOriginal prompt: " + raw
    )
    response = ask_with_image(rt, image, prompt, cfg)
    try:
        edit = parse_edit(response)
        status = "edit" if edit else "no_failure"
    except ValueError:
        edit, status = None, "failed_unconverted"
    return dict(questions=[], decomposition={"response": response}, preview_score={"questions": []},
                failed=[], rejected=[], edit_request=edit["instruction"] if edit else "",
                preserve=edit["preserve"] if edit else "", atomic_edit=edit, diagnosis_outcome=status)
