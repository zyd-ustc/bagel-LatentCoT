"""Offline external VLM; credentials remain in the configured environment variable."""
import base64,json,mimetypes,os
from pathlib import Path


def _object(text):
    raw=(text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lines=raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines=lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines=lines[:-1]
            return json.loads("\n".join(lines).strip())
        raise


def judge(requests,output,base_url,model,key_env):
    from openai import OpenAI
    client=OpenAI(base_url=base_url,api_key=os.environ[key_env])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=Path(requests).resolve()
    for line in source.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        # Output filenames are positional to keep external UIDs out of paths.
        name=out/(str(row['index']).zfill(6)+'.json')
        if name.exists():continue
        content=[dict(type='text',text=row['instruction']+"\nReturn JSON with requirements (entity-bound verdicts), structural_integrity, preservation, edit_instruction, and accepted. Inspect object fusion, missing/misconnected parts and unintended changes. Do not infer correctness from style or aesthetics.")]
        for item in row['images']:
            path=source.parent/item['path']
            content.append(dict(type='text',text=item['role']))
            content.append(dict(type='image_url',image_url=dict(url='data:'+mimetypes.guess_type(path)[0]+';base64,'+base64.b64encode(path.read_bytes()).decode())))
        response=client.chat.completions.create(model=model,messages=[dict(role='user',content=content)],response_format={'type':'json_object'})
        result=_object(response.choices[0].message.content)
        name.write_text(json.dumps(dict(uid=row['uid'],request=row,model=model,response=result,usage=response.usage.model_dump() if response.usage else None),ensure_ascii=False,indent=2))
