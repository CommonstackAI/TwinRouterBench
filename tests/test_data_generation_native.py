"""Native bridge regressions using real local HTTP and resettable state."""
import copy
import json
import urllib.request
from dataclasses import replace
from types import SimpleNamespace

import pytest

from twinrouterbench.data_generation.adapters import get_adapter
from twinrouterbench.data_generation.harness_proxy import HarnessProxyBackend, completion_envelope
from twinrouterbench.data_generation.interactive import InteractiveBackend, Observation, OfficialVerdict
from twinrouterbench.data_generation.text_loaders import mtrag_filter_reason, QMSumLoader
from twinrouterbench.data_generation.types import Assignment
from twinrouterbench.data_generation.source import SourceSpec


class Client:
    generation_parameters = {}
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def _chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        message = next(self.responses)
        self.last_completion = completion_envelope(message, kwargs['model'])
        return copy.deepcopy(message)


class Environment:
    all = []
    drift = False
    def __init__(self, task):
        self.state = 0
        self.closed = False
        self.all.append(self)
    def step(self, response):
        if response['content'] == 'set':
            self.state = 7
            return Observation(({'role': 'user', 'content': 'changed' if self.drift else 'state=7'},))
        return Observation(done=True)
    def evaluate(self):
        return OfficialVerdict(self.state == 7, 'state checked', {'state': self.state})
    def close(self):
        self.closed = True


def task():
    original = get_adapter('swebench').load_tasks()[0]
    return replace(original, steps=original.steps[:2])


def message(content):
    return {'role': 'assistant', 'content': content}


def test_swe_incompatible_harness_fails_before_allocating_resources(monkeypatch, tmp_path):
    from twinrouterbench.data_generation.swebench_backend import SwebenchEnvironments

    monkeypatch.setattr('importlib.metadata.version',
        lambda name: '2.2.8' if name == 'mini-swe-agent' else '5.0.2')
    work_dir = tmp_path / 'native-work'
    with pytest.raises(RuntimeError, match=r'swebench==3\.0\.15'):
        SwebenchEnvironments(work_dir)
    assert not work_dir.exists()


def test_interactive_reexecutes_state_and_only_bills_suffix():
    client = Client([message('set'), message('submit'), message('submit')])
    backend = InteractiveBackend(client, Environment)
    seed = backend.run_seed(task(), 'strong')
    assert seed.passed
    result = backend.run_mixed(task(), [Assignment('high', 'strong'), Assignment('low', 'cheap')],
        target_step=2, generation_parameters={}, locked_execution=seed)
    assert result.passed and result.prefixes == seed.prefixes
    assert result.responses[0] == seed.responses[0]
    assert len(client.calls) == 3
    assert Environment.all[-1].state == 7 and Environment.all[-1].closed
    backend.close()  # Class-based factories must not call an unbound close().


def test_interactive_drift_fails_before_paid_suffix():
    client = Client([message('set'), message('submit')])
    backend = InteractiveBackend(client, Environment)
    seed = backend.run_seed(task(), 'strong')
    class Drifting(Environment):
        drift = True
    backend.environment_factory = Drifting
    result = backend.run_mixed(task(), [Assignment('high', 'strong'), Assignment('low', 'cheap')],
        target_step=2, generation_parameters={}, locked_execution=seed)
    assert not result.passed and result.verification['prefix_replay_matches'] is False
    assert len(client.calls) == 2 and Environment.all[-1].closed


def test_termination_is_not_a_pass_and_errors_close_environment():
    backend = InteractiveBackend(Client([message('submit')]), Environment)
    assert not backend.run_seed(task(), 'strong').passed
    backend.client = Client([])
    with pytest.raises(StopIteration):
        backend.run_seed(task(), 'strong')
    assert Environment.all[-1].closed


def test_native_proxy_freezes_scaffold_without_replacing_feedback():
    calls = []
    def runner(task, base, nonce):
        initial = [{'role':'system','content':f'clock={len(calls)}'}, {'role':'user','content':'task'}]
        calls.append(base)
        messages = copy.deepcopy(initial)
        for i in range(2):
            body = {'model':'virtual','messages':messages,'tools':[],'stream':False,'max_tokens':100}
            request = urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                assistant = json.load(response)['choices'][0]['message']
            messages.extend([assistant, {'role':'user','content':'actual-tool-result'}])
        return OfficialVerdict(True,'native grader',{})
    client = Client([message('first'),message('last'),message('cheap-last')])
    backend = HarnessProxyBackend(client,runner)
    t = replace(task(),metadata={'max_steps':2,'freeze_initial_prefix':True})
    seed = backend.run_seed(t,'strong')
    candidate = backend.run_mixed(t,[Assignment('high','strong'),Assignment('low','cheap')],
        target_step=2,generation_parameters={},locked_execution=seed)
    assert candidate.passed and candidate.prefixes == seed.prefixes
    assert len(client.calls)==3 and client.calls[-1]['messages'][-1]['content']=='actual-tool-result'
    assert candidate.verification['native_initial_sha256'] != seed.verification['native_initial_sha256']


@pytest.mark.parametrize('change,reason', [
    ({'Answerability':['UNANSWERABLE']},'not_answerable'),
    ({'contexts':[]},'no_contexts'),
    ({'contexts':[{'text':'evidence'}]},'missing_human_relevance'),
    ({'contexts':[{'text':'evidence','feedback':{'relevant':{'human':{'value':'no'}}}}]},'irrelevant_context'),
])
def test_mtrag_filter_requires_answerability_and_human_relevance(change,reason):
    row={'Answerability':['ANSWERABLE'],'contexts':[{'text':'evidence','feedback':{'relevant':{'human':{'value':'yes'}}}}]}
    assert not mtrag_filter_reason(row)
    row.update(change)
    assert mtrag_filter_reason(row)==reason


def test_mtrag_historical_filter_is_explicit():
    row={'Answerability':['ANSWERABLE'],'contexts':[{'text':'evidence'}]}
    assert not mtrag_filter_reason(row,require_human_feedback=False)


def test_qmsum_specific_spans_are_inclusive_and_bounds_checked(tmp_path):
    path=tmp_path/'input.json'
    row={'meeting_id':'test','meeting_transcripts':[{'speaker':'A','content':str(i)} for i in range(4)],
        'specific_query_list':[{'query':'what','answer':'yes','relevant_text_span':[['1','2']]}]}
    path.write_text(json.dumps([row]))
    loader=QMSumLoader({})
    tasks=loader.load_tasks(get_adapter('qmsum'),SourceSpec.parse('local://'+str(path)))
    content=tasks[0].initial_messages[-1]['content']
    assert 'A: 1\nA: 2' in content and 'A: 0' not in content and 'A: 3' not in content
    row['specific_query_list'][0]['relevant_text_span']=[['1','9']]
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='outside'):
        loader.load_tasks(get_adapter('qmsum'),SourceSpec.parse('local://'+str(path)))


def test_swe_grader_uses_a_container_name_distinct_from_work_environment(monkeypatch,tmp_path):
    from twinrouterbench.data_generation.swebench_backend import SwebenchEnvironment
    from swerouter.harness import container_runner
    env=object.__new__(SwebenchEnvironment)
    env.run_id='owned-work-container'
    env.task=SimpleNamespace(instance_id='real-instance')
    env.test_spec=object()
    env.factory=SimpleNamespace(eval_timeout=10)
    env.patch='nonempty patch'
    env.submitted=True
    def native_grade(**kwargs):
        if kwargs['run_id']==env.run_id:
            raise RuntimeError('Docker 409: work container still exists')
        return SimpleNamespace(resolved=True,patch_applied=True,error=None,report_path=tmp_path/'report.json')
    monkeypatch.setattr(container_runner,'run_upstream_eval',native_grade)
    monkeypatch.setattr('importlib.metadata.version',lambda name:'pinned-test-version')
    assert env.evaluate().passed


def test_native_proxy_keeps_actual_auxiliary_context():
    requests=[]
    def runner(task,base,nonce):
        for content in ['task conversation','auxiliary context']:
            messages=[{'role':'system','content':content},{'role':'user','content':'work'}]
            request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps({'messages':messages,'max_tokens':10}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=5) as response:
                json.load(response)
            requests.append(messages)
        return OfficialVerdict(True,'native grader',{})
    client=Client([message('a'),message('b')])
    backend=HarnessProxyBackend(client,runner)
    t=replace(task(),metadata={'max_steps':2,'freeze_initial_prefix':True})
    seed=backend.run_seed(t,'strong')
    assert seed.prefixes[1][0]['content']=='auxiliary context'
    assert client.calls[1]['messages']==requests[1]


def test_native_output_limit_is_forwarded_without_injecting_a_second_limit(monkeypatch):
    import io
    from twinrouterbench.data_generation.openai_backend import OpenAICompatibleBackend
    from twinrouterbench.data_generation.pipeline import load_model_pool
    from twinrouterbench.data_generation.budget import request_reservation
    _,pool=load_model_pool()
    client=OpenAICompatibleBackend(pool,api_key='local-test',base_url='http://unused')
    bodies=[]
    def transport(request,**kwargs):
        bodies.append(json.loads(request.data))
        return io.BytesIO(json.dumps(completion_envelope(message('ok'),'model')).encode())
    monkeypatch.setattr(urllib.request,'urlopen',transport)
    client._chat(model='model',messages=[{'role':'user','content':'test'}],
        generation_parameters={'max_completion_tokens':8192})
    assert bodies[0]['max_completion_tokens']==8192 and 'max_tokens' not in bodies[0]
    with pytest.raises(ValueError,match='only one'):
        client._chat(model='model',messages=[],generation_parameters={'max_tokens':1,'max_completion_tokens':8192})
    price={'prompt':0.000001,'completion':0.00001}
    assert request_reservation(bodies[0],price)>8192*price['completion']
    with pytest.raises(ValueError,match='ambiguous'):
        request_reservation({**bodies[0],'max_tokens':1},price)


def test_optional_json_mode_is_recorded_and_used_for_judge(monkeypatch):
    from twinrouterbench.data_generation.openai_backend import OpenAICompatibleBackend
    from twinrouterbench.data_generation.pipeline import load_model_pool
    from twinrouterbench.data_generation.backends import MockBackend
    _,pool=load_model_pool()
    client=OpenAICompatibleBackend(pool,api_key='local-test',base_url='http://unused',
        json_response_format={'type':'json_object'})
    calls=[]
    verdict={'passed':False,'faithfulness':False,'appropriateness':False,'completeness':False,'uncertain':False,'evidence_conflict':True}
    def chat(**kwargs):
        calls.append(kwargs)
        return message(json.dumps(verdict))
    monkeypatch.setattr(client,'_chat',chat)
    t=get_adapter('qmsum').load_tasks()[0]
    result=client.judge_open_ended(t,MockBackend(pool).run_seed(t,pool['high'][0]))
    assert result.passed is False
    assert calls[0]['generation_parameters']['response_format']=={'type':'json_object'}
    assert client.prompt_artifacts['open_ended_judge']['response_format']=={'type':'json_object'}


def test_compatible_provider_prose_can_precede_one_explicit_json_verdict():
    from twinrouterbench.data_generation.openai_backend import OpenAICompatibleBackend
    verdict={'passed':False,'faithfulness':False,'appropriateness':False,'completeness':False,'uncertain':False,'evidence_conflict':True}
    parsed=OpenAICompatibleBackend._json_response(message('Analysis of the evidence.\n\n'+json.dumps(verdict)))
    assert parsed==verdict
    assert OpenAICompatibleBackend._json_response(message('Analysis.\n```json\n'+json.dumps(verdict)+'\n```'))==verdict
    for invalid in [json.dumps(verdict)+'\n'+json.dumps({'passed':True}),
                    'Analysis. '+json.dumps(verdict)+' trailing opinion',
                    'Analysis {malformed}\n'+json.dumps(verdict),
                    'Analysis.\n```json\n'+json.dumps(verdict)+'\n```\nTrailing opinion',
                    'Analysis.\n```json\n'+json.dumps(verdict)+'\n```\n```json\n{}\n```',
                    'I think the answer passed.']:
        with pytest.raises(RuntimeError,match='invalid JSON'):
            OpenAICompatibleBackend._json_response(message(invalid))


def test_judge_output_budget_is_configurable_and_recorded(monkeypatch):
    from twinrouterbench.data_generation.openai_backend import OpenAICompatibleBackend
    from twinrouterbench.data_generation.pipeline import load_model_pool
    from twinrouterbench.data_generation.backends import MockBackend
    _,pool=load_model_pool()
    client=OpenAICompatibleBackend(pool,api_key='test',base_url='http://unused',judge_max_tokens=2048)
    calls=[]
    def chat(**kwargs):
        calls.append(kwargs)
        return message(json.dumps({'passed':False,'faithfulness':False,'appropriateness':False,'completeness':False,'uncertain':True,'evidence_conflict':False}))
    monkeypatch.setattr(client,'_chat',chat)
    t=get_adapter('mtrag').load_tasks()[0]
    verdict=client.judge_open_ended(t,MockBackend(pool).run_seed(t,pool['high'][0]))
    assert verdict.uncertain and not verdict.passed
    assert calls[0]['generation_parameters']['max_tokens']==2048
    assert client.prompt_artifacts['open_ended_judge']['max_tokens']==2048
    with pytest.raises(ValueError,match='positive integer'):
        OpenAICompatibleBackend(pool,api_key='test',base_url='http://unused',judge_max_tokens=0)
