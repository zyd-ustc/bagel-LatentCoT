"""Frozen UND question decomposition, visual scoring, and correction conditions."""

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
    "If there are two or more main entities, include a SEPARATE question checking that "
    "they are physically distinct, coherent entities, not fused into a single object. "
    "Allow the contact, overlap and occlusion explicitly requested; distinct does not "
    "mean fully exposed or non-touching.\n"
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
    "\nKeep at most four core scene questions plus one dedicated distinct-entity question. "
    "Check the direction of asymmetric relations against the raw request before returning. "
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
    return questions, dict(response=result["response"], edits=edits)


@torch.no_grad()
def semantic_plan(rt, raw, config, *, system_message=None, question=None):
    """Text-only UND planning; identical information is available at deployment."""
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    default_question = (
        "Turn the user's image request into a concise visual construction plan. "
        "Resolve subject identity, exact counts, who is in front of or behind whom, "
        "and the main spatial arrangement. Preserve the requested style and exact quoted text. "
        "Describe one complete coherent intended image in positive concrete terms. "
        "Do not invent extra objects or claim to have inspected an image. "
        "Do not write an editing critique, a caption of an existing image, or reasoning. "
        "Use at most 80 words. Return only the construction plan.\nUSER REQUEST:\n"
        + raw
    )
    question = default_question if question is None else question
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
    return dict(
        raw_prompt=raw,
        response=response,
        plan=plan,
        condition=raw + "\nVisual construction plan:\n" + plan,
    )


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

@torch.no_grad()
def und_log_reward(rt, image, questions, yes_ids=None, no_ids=None):
    """Single-pass prefill Yes/No log-sigmoid conjunction score in (-inf, 0]."""
    m, tok = rt.model, rt.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    yes_ids = yes_ids or _single_token_ids(tok, YES_VARIANTS)
    no_ids = no_ids or _single_token_ids(tok, NO_VARIANTS)

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
    img_tag = "<img>" + "<IMG_CONTEXT>" * rt.tokens + "</img>"

    with autocast(rt.device):
        # 同一张预览图只跑 1 次理解侧 ViT
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
def und_score_image(rt, image, questions, yes_ids=None, no_ids=None):
    _, scores = und_log_reward(rt, image, questions, yes_ids, no_ids)
    return summarize_question_scores(scores)


def semantic_correction_condition(raw, clauses):
    return raw + "\n[Semantic Corrections]:\n" + "\n".join(clauses) if clauses else raw

def affirmative_constraints(questions, allowed_negations=frozenset()):
    clauses = []
    rejected = []
    for question in questions:
        quoted = re.match(r'Is the exact text "(.+)" clearly legible', question)
        if quoted:
            clauses.append(
                f'Display the exact text "{quoted.group(1)}", clearly legible and accurately spelled.'
            )
            continue
        text = question.strip().rstrip("?？")
        text = re.sub(
            r"\b(?:and )?not fused(?: into (?:a |one |the )?(?:single |same )?object)?",
            "and physically distinct",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bwithout garbled letters\b", "with accurate lettering", text, flags=re.I
        )
        negatives = set(
            re.findall(r"\b(no|not|missing|garbled|without)\b", text.lower())
        )
        if negatives - allowed_negations:
            rejected.append(question)
            continue
        if "是否" in text:
            clauses.append(text.replace("是否", "", 1) + "。")
            continue
        if re.match(r"^(Is|Are) there ", text, re.I):
            text = re.sub(r"^(Is|Are) there ", "There must be ", text, flags=re.I)
        else:
            sentence = re.match(
                r"^(Is|Are) (.+?) (present|visible|positioned|located|parked|balanced|"
                r"older|younger|physically|morphing|holding|blocking|whispering|stitching|not)(.*)$",
                text,
                re.I,
            )
            if sentence:
                verb, subject, predicate, rest = sentence.groups()
                text = f"{subject[0].upper() + subject[1:]} {verb.lower()} {predicate}{rest}"
            else:
                text = "Show " + re.sub(r"^(Is|Are) ", "", text, flags=re.I)
        clauses.append(text + ".")
    return clauses, rejected

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
