"""Same-state visual feedback; no sampler/cache mutation inside a round."""
from dataclasses import dataclass
import torch
from .flow_training import conditional_gen, FlowOutput
from .feedback_source import prompt_source


@dataclass
class LoopOutput:
    baseline: FlowOutput
    banks: list
    exits: list[FlowOutput]
    gen_calls: int

    @property
    def feedback_updates(self):
        return len(self.banks)

    @property
    def passes(self):
        return self.feedback_updates+1


def memory_loop(student, state, rounds=1, *, feedback='new', donor_source=None,
                diagnostics=False, checkpoint_activations=False, donor_metadata=None):
    if rounds not in (1, 2) or feedback not in ('new', 'old', 'fixed', 'swap'):
        raise ValueError('support one/two feedback updates and new/old/fixed/swap')
    runtime = student.runtime
    if feedback == 'swap':
        if not donor_metadata or (donor_metadata.get('prompt'), donor_metadata.get('t')) != (state.metadata['prompt'], state.t):
            raise ValueError('swap feedback requires verified same-prompt/t donor metadata')
    cross_round = runtime.config.get('cross_round_gradients', False)
    with torch.no_grad():
        baseline = conditional_gen(runtime, state, round_index=0)
    source = baseline.source
    prompt = state.prompt_kv if state.prompt_kv is not None else prompt_source(state.flow, state.lengths, state.positions)
    banks = []; exits = []
    for index in range(rounds):
        if index and feedback == 'fixed':
            bank = banks[0]
        else:
            if index:
                source = (baseline.source if feedback == 'old' else
                          donor_source if feedback == 'swap' else exits[-1].source)
                if source is None:
                    raise ValueError('swap feedback requires a same-prompt/t donor source')
            # write always resets the SAME neutral seeds. M1 is not a writer input.
            bank = student.write(source, state.t, prompt=prompt, detach_source=not cross_round)
        output = conditional_gen(runtime, state, bank, capture=index < rounds-1 or diagnostics,
                                 round_index=index+1, detach_source=not cross_round,
                                 checkpoint_activations=checkpoint_activations)
        banks.append(bank); exits.append(output)
    return LoopOutput(baseline, banks, exits, 1+rounds)
