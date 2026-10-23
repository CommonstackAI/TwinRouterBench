#!/usr/bin/env python3
"""Run real OpenCode against a LOCAL synthetic tool-loop provider; no model API.

Usage: python -m twinrouterbench.opencode_router.smoke --binary /path/to/opencode
The only emitted tool command prints a marker. All model responses are fixtures.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from .proxy import RouterHandler, RouterServer, HERE


class ToolLoopFixture(RouterHandler):
    def mock_reply(self,body,decision,event):
        self.server.smoke_requests.append({'roles':[m.get('role') for m in body['messages']],
            'last_message_preview':str(body['messages'][-1].get('content'))[:2000]})
        has_tool_result=any(m.get('role')=='tool' for m in body['messages'])
        tools={t.get('function',{}).get('name') for t in body.get('tools',[])}
        if 'bash' not in tools or has_tool_result:
            return super().mock_reply(body,decision,event)
        call={'id':'call_local_smoke_only','type':'function','function':{
            'name':'bash','arguments':json.dumps({'command':"printf 'ROUTER_SMOKE_OBSERVATION\\n'",'description':'Print a local smoke-test marker'})}}
        usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'cost':0}
        message={'role':'assistant','content':'','tool_calls':[call]}
        common={'id':'chatcmpl-'+event['request_id'],'created':int(time.time()),'model':decision['model_id']}
        if body.get('stream'):
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers();self._stream_started=True
            delta=copy.deepcopy(message);delta['tool_calls'][0]['index']=0
            chunks=[dict(common,object='chat.completion.chunk',choices=[{'index':0,'delta':delta,'finish_reason':None}]),
                    dict(common,object='chat.completion.chunk',choices=[{'index':0,'delta':{},'finish_reason':'tool_calls'}],usage=usage)]
            for chunk in chunks:self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();self.close_connection=True
        else:
            self.send_json(200,dict(common,object='chat.completion',choices=[{'index':0,'message':message,'finish_reason':'tool_calls'}],usage=usage))
        event.update(status='mock_tool_call_fixture',usage=usage,actual_cost_usd=0.)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('opencode_smoke_result.json'));args=p.parse_args()
    binary=args.binary.resolve()
    config=json.loads((HERE/'router.example.json').read_text())
    config['local_token_env']='TWINROUTER_SMOKE_UNUSED_TOKEN'
    with tempfile.TemporaryDirectory(prefix='opencode-twinrouter-smoke-') as d:
        root=Path(d);ledger=root/'events.jsonl'
        server=RouterServer(('127.0.0.1',0),config,mode='mock',ledger=ledger)
        server.smoke_requests=[]
        server.RequestHandlerClass=ToolLoopFixture
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=json.loads((HERE/'opencode.example.json').read_text())
        client['provider']['twinrouter']['options']['baseURL']='http://127.0.0.1:'+str(server.server_port)+'/v1'
        client['permission']={'*':'deny','bash':'allow'}
        config_path=root/'opencode.json';config_path.write_text(json.dumps(client))
        env=dict(os.environ)
        for key,sub in [('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('XDG_STATE_HOME','state'),('XDG_CACHE_HOME','cache'),('OPENCODE_CONFIG_DIR','config-dir')]:
            path=root/sub;path.mkdir();env[key]=str(path)
        env['PWD']=str(root)
        env.pop('TWINROUTER_SMOKE_UNUSED_TOKEN',None)
        env.update(OPENCODE_CONFIG=str(config_path),OPENCODE_DISABLE_AUTOUPDATE='true',OPENCODE_DISABLE_DEFAULT_PLUGINS='true',MODELS_DEV_DISABLE_REMOTE='true')
        try:
            proc=subprocess.run([str(binary),'run','--model','twinrouter/auto','--format','json','list files'],cwd=root,env=env,capture_output=True,text=True,timeout=150)
        finally:
            server.shutdown();server.server_close();thread.join()
        events=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        models=[e['selected_model'] for e in events]
        tool_emitted=any(e['status']=='mock_tool_call_fixture' for e in events)
        tool_observed='ROUTER_SMOKE_OBSERVATION' in proc.stdout
        result={'opencode_binary':str(binary),'exit_code':proc.returncode,'mode':'local_synthetic_fixture_only',
                'new_model_api_calls':0,'selected_models':models,'provider_requests':len(events),
                'tool_call_emitted':tool_emitted,'tool_output_observed':tool_observed,
                'automatic_model_change_observed':len(set(models))>=2,
                'request_summaries':server.smoke_requests,
                'passed':proc.returncode==0 and tool_emitted and tool_observed and len(set(models))>=2,
                'stdout':proc.stdout[-20000:],'stderr':proc.stderr[-12000:]}
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['stdout','stderr']},ensure_ascii=False,indent=2))
        if not result['passed']:print(proc.stderr[-3000:]);raise SystemExit(1)


if __name__=='__main__':main()
