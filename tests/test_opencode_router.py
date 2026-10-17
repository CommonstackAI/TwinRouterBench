import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from twinrouterbench.opencode_router.proxy import RouterServer, RoutingPolicy, usage_cost
from twinrouterbench.opencode_router import proxy

HERE=Path(proxy.__file__).resolve().parent


class FakeUpstream:
    def __init__(self, raw, content_type='application/json'):
        import io
        self.buffer=io.BytesIO(raw)
        self.status=200
        self.headers={'Content-Type':content_type}
    def read(self):return self.buffer.read()
    def read1(self, n):return self.buffer.read(min(n,7))
    def __enter__(self):return self
    def __exit__(self,*args):pass


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=json.loads((HERE/'router.example.json').read_text())
        self.server=RouterServer(('127.0.0.1',0),self.config,mode='mock',ledger=Path(self.temp.name)/'events.jsonl')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def post(self, body, headers=None):
        return urllib.request.urlopen(urllib.request.Request(self.url+'/v1/chat/completions', data=json.dumps(body).encode(),headers={'Content-Type':'application/json',**(headers or {})}),timeout=5)

    def test_each_request_routes_current_prefix(self):
        first={'model':'auto','messages':[{'role':'user','content':'list files'}]}
        with self.post(first) as r:one=json.load(r)
        later=copy.deepcopy(first)
        later['messages'] += [{'role':'assistant','tool_calls':[{'id':'a','type':'function','function':{'name':'bash','arguments':'{}'}}]}, {'role':'tool','tool_call_id':'a','content':'Traceback: test failed'}]
        with self.post(later) as r:two=json.load(r)
        self.assertEqual(one['model'],self.config['models']['low'])
        self.assertEqual(two['model'],self.config['models']['high'])
        self.assertEqual(first['model'],'auto')

    def test_client_json_wrapped_text_is_normalized_for_policy_only(self):
        request={'model':'auto','messages':[{'role':'user','content':'"list files"'}]}
        original=copy.deepcopy(request)
        decision=self.server.policy.select(request,None)
        self.assertEqual(decision['model_id'],self.config['models']['low'])
        self.assertEqual(request,original)

    def test_mock_stream_and_ledger_have_no_prompt_or_key(self):
        body={'model':'auto','stream':True,'messages':[{'role':'user','content':'PRIVATE_PROMPT_DO_NOT_LOG'}]}
        with self.post(body,{'x-twinrouter-session':'private-session'}) as r:
            text=r.read().decode();self.assertIn('data: [DONE]',text)
            self.assertEqual(r.headers['Content-Type'],'text/event-stream')
        self.server.shutdown()
        text=(Path(self.temp.name)/'events.jsonl').read_text();event=json.loads(text)
        self.assertEqual(event['actual_cost_usd'],0.)
        self.assertNotIn('PRIVATE_PROMPT_DO_NOT_LOG',text)
        self.assertNotIn('private-session',text)

    def test_invalid_pool_or_virtual_model_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post({'model':'other','messages':[{'role':'user','content':'hi'}]})
        self.assertEqual(e.exception.code,400)
        with tempfile.TemporaryDirectory() as d:
            Path(d,'policy.py').write_text("def select_model(request, context): return 'outside-pool'\n")
            config=dict(self.config,policy_module='policy.py')
            with self.assertRaises(ValueError):RoutingPolicy(config,d).select({'messages':[]},None)

    def test_custom_policy_receives_tools_and_cannot_mutate_forwarded_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'policy.py').write_text(
                "def select_model(request, context):\n"
                "    assert request['tools'][0]['function']['name'] == 'read'\n"
                "    assert context['session_id'] == 'hashed-session'\n"
                "    request['messages'][0]['content'] = 'changed locally'\n"
                "    return context['tier_to_model']['low']\n"
            )
            request = {'messages': [{'role': 'user', 'content': 'original'}],
                       'tools': [{'type': 'function', 'function': {'name': 'read'}}]}
            original = copy.deepcopy(request)
            policy = RoutingPolicy(dict(self.config, policy_module='policy.py'), d)
            self.assertEqual(policy.select(request, 'hashed-session')['model_id'], self.config['models']['low'])
            self.assertEqual(request, original)

    def test_optional_local_token_is_checked_before_routing(self):
        self.server.local_token = 'local-fixture-token'
        body = {'model': 'auto', 'messages': [{'role': 'user', 'content': 'list files'}]}
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post(body)
        self.assertEqual(error.exception.code, 401)
        with self.post(body, {'Authorization': 'Bearer local-fixture-token'}) as response:
            self.assertEqual(response.status, 200)

    def test_live_requires_explicit_key_but_mock_does_not(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):RouterServer(('127.0.0.1',0),self.config,mode='live')

    def test_json_forward_preserves_tools_and_accounts_provider_cost(self):
        self.server.mode='live';self.server.upstream_key='SECRET_TEST_KEY'
        body={'model':'auto','messages':[{'role':'user','content':'list files'}],
              'tools':[{'type':'function','function':{'name':'bash','parameters':{'type':'object'}}}], 'tool_choice':'auto'}
        result={'model':self.config['models']['low'],'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'x','function':{'name':'bash','arguments':'{}'}}]}}], 'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':.001}}
        original_open=urllib.request.urlopen
        seen=[]
        def fake_open(request,*args,**kwargs):
            if request.full_url.startswith(self.url):return original_open(request,*args,**kwargs)
            seen.append(request)
            return FakeUpstream(json.dumps(result).encode())
        with patch('urllib.request.urlopen',side_effect=fake_open):
            with self.post(body) as r:self.assertEqual(json.load(r),result)
        sent=json.loads(seen[0].data)
        self.assertEqual(sent['model'],self.config['models']['low'])
        self.assertEqual(sent['messages'],body['messages']);self.assertEqual(sent['tools'],body['tools'])
        self.server.shutdown()
        event=json.loads((Path(self.temp.name)/'events.jsonl').read_text())
        self.assertEqual(event['actual_cost_usd'],.001)
        self.assertNotIn('SECRET_TEST_KEY',json.dumps(event))

    def test_stream_forward_preserves_tool_fragments_and_usage(self):
        self.server.mode='live';self.server.upstream_key='SECRET_TEST_KEY'
        chunks=[{'model':'selected-real-model','choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call1','type':'function','function':{'name':'bash','arguments':'{"command":'}}]}}]},
                {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"pwd"}'}}]},'finish_reason':'tool_calls'}]},
                {'choices':[],'usage':{'prompt_tokens':200,'completion_tokens':20,'cost':.002}}]
        raw=(''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode()
        original_open=urllib.request.urlopen
        def fake_open(request,*args,**kwargs):
            if request.full_url.startswith(self.url):return original_open(request,*args,**kwargs)
            self.assertTrue(json.loads(request.data)['stream_options']['include_usage'])
            return FakeUpstream(raw,'text/event-stream')
        with patch('urllib.request.urlopen',side_effect=fake_open):
            with self.post({'model':'auto','stream':True,'messages':[{'role':'user','content':'hi'}]}) as r:self.assertEqual(r.read(),raw)
        self.server.shutdown()
        event=json.loads((Path(self.temp.name)/'events.jsonl').read_text())
        self.assertEqual(event['actual_cost_usd'],.002)
        self.assertEqual(event['provider_model'],'selected-real-model')

    def test_missing_or_invalid_cost_is_not_zero(self):
        for usage in [None,{}, {'cost':'0.1'},{'cost':float('nan')},{'cost':-1}]:self.assertIsNone(usage_cost(usage))
        self.assertEqual(usage_cost({'cost':0}),0.)

    def test_optional_session_plugin_uses_the_published_hook(self):
        text=(HERE/'session_header_plugin.mjs').read_text()
        self.assertIn('"chat.headers"',text);self.assertIn('input.sessionID',text)


if __name__=='__main__':unittest.main()
