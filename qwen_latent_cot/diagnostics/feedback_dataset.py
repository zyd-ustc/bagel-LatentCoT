"""Export native baseline states and fixed text teachers from checked E1 rows."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .explicit_feedback import Feedback, correction_source
from .feedback_source import prompt_source, write_feedback
from .feedback_restart import initial_noise
from .flow_training import TrainingState, conditional_gen
from .schedule import native_times
from .data import tensor_hash, file_hash
from .record import write_json, load_checkpoint, manifest


@torch.no_grad()
def collect_feedback_states(runtime, rows, output, nodes):
    """Feedback describes the completed E1 image; it is not a noisy-image caption."""
    times = native_times(runtime.config); cfg = runtime.config
    if not nodes or min(nodes) < 0 or max(nodes) >= len(times)-1:
        raise ValueError('export nodes must be native Euler update indexes')
    output.mkdir(parents=True, exist_ok=True); exported = []
    families = {}
    for row in rows:
        if row.get('verified') is not True or row.get('split') not in ('train','val','test') or not row.get('family_id'):
            raise ValueError('checked feedback export needs verified labels, split and family_id')
        if families.setdefault(row['family_id'], row['split']) != row['split']:
            raise ValueError('feedback family leaks across splits')
        feedback = Feedback(**row['feedback'])
        if feedback.origin != 'manual_checked':
            raise ValueError('export checked E1 feedback separately from automatic feedback')
        flow, lengths, positions = runtime.flow_inputs(row['prompt'])
        x = initial_noise(runtime, flow, row['noise_seed'])
        if row.get('initial_state'):
            saved = torch.load(row['initial_state'], map_location=runtime.device, weights_only=True)
            sampler = {k:cfg[k] for k in ('resolution','sampler_steps','timestep_shift','cfg_text_scale','cfg_renorm_min','cfg_renorm_type')}
            if saved['prompt'] != row['prompt'] or saved['noise_seed'] != row['noise_seed'] or saved['model_files'] != runtime.native_files or saved['sampler_config'] != sampler:
                raise ValueError('E1 initial state differs from the native model/prompt/sampler')
            x = saved['x_initial'].to(runtime.device)
            if tensor_hash(x) != saved['initial_noise_hash']:
                raise ValueError('E1 initial noise hash differs')
        p = prompt_source(flow, lengths, positions)
        teacher_bank = None
        if not feedback.satisfied:
            correction = correction_source(runtime, flow, lengths, positions, feedback.correction)[0]
            teacher_bank = write_feedback(runtime, p, correction)
        for index,(t,s) in enumerate(zip(times,times[1:])):
            if index in nodes:
                name = f"{row['sample_id']}_n{index:02d}"
                metadata = dict(state_source='trajectory', prompt=row['prompt'], t=t,
                                step_index=index, noise_seed=row['noise_seed'], x_t_hash=tensor_hash(x),
                                sampler_steps=cfg['sampler_steps'], timestep_shift=cfg['timestep_shift'],
                                native_files=runtime.native_files)
                state = TrainingState(x,None,None,t,flow,lengths,positions,metadata)
                if teacher_bank is not None:
                    target = conditional_gen(runtime,state,teacher_bank,capture=False).velocity
                else:
                    from dataclasses import replace
                    from .explicit_feedback import shifted_layout
                    target = conditional_gen(runtime,replace(state,flow=shifted_layout(flow,16)),capture=False).velocity
                torch.save(dict(metadata,x_t=x.cpu()),output/(name+'_state.pt'))
                torch.save(dict(metadata,velocity=target.cpu(),basis='conditional_pre_cfg',
                                text_teacher_hash=runtime.text_teacher_hash,
                                correction=feedback.correction,satisfied=feedback.satisfied,
                                bank=None if teacher_bank is None else [dict(layer=kv.layer,
                                    keys=kv.keys.cpu(),values=kv.values.cpu(),lengths=kv.lengths,rotated=False)
                                    for kv in teacher_bank.layers]),output/(name+'_teacher.pt'))
                exported.append(dict(sample_id=name,scene_id=row.get('scene_id',row['family_id']),
                    family_id=row['family_id'],split=row['split'],prompt=row['prompt'],t=t,
                    noise_seed=row['noise_seed'],state=name+'_state.pt',teacher=name+'_teacher.pt',
                    label_kind='feedback',label_type='observed',verified=True,facts=[],feedback=asdict(feedback),
                    feedback_scope='checked complete E1 image; target for this baseline state'))
            if index >= max(nodes):
                break
            x = x-(t-s)*runtime.velocity(x,t,flow)[1]
    (output/'feedback.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exported))
    return exported


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',required=True,help='checked E1 template with split/family_id/verified fields')
    parser.add_argument('--text-checkpoint',required=True)
    parser.add_argument('--config',default='configs/visual_feedback.json')
    parser.add_argument('--nodes',nargs='+',type=int,default=[0,5,10,15,19])
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    from .feedback_restart_cli import load_rows
    rows=load_rows(args.data);cfg=json.loads(Path(args.config).read_text())
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    runtime=DiagnosticRuntime(load_native(args.model,args.device,cfg['timestep_shift']),cfg,'C')
    run=manifest(runtime,args,args.data,rows=rows);runtime.native_files=run['model_files']
    load_checkpoint(args.text_checkpoint,runtime,('C1','C1-read'))
    runtime.writer.eval().requires_grad_(False);runtime.text_teacher_hash=file_hash(args.text_checkpoint)
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh feedback export directory')
    records=collect_feedback_states(runtime,rows,output,args.nodes)
    write_json(output/'manifest.json',dict(run,exported_samples=len(records),nodes=args.nodes,
                                          text_teacher_hash=runtime.text_teacher_hash,status='exported checked E1 states'))


if __name__ == '__main__': main()
