#!/usr/bin/env python3
"""OpenAI-compatible, per-request routing provider for OpenCode (stdlib only).

Run without a paid API: python -m twinrouterbench.opencode_router --mode mock
Real forwarding is opt-in: --mode live (requires configured API key env var).
This is a routing transport with an explicitly unvalidated rule baseline, not
the trained router from the paper. A local Python policy can replace the rules.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent


def text_content(value):
    if isinstance(value, str):
        # Some client serializers wrap a text part as a JSON string literal.
        # Normalize for the local policy only; forwarded messages stay untouched.
        if value.startswith('"') and value.endswith('"'):
            try:
                decoded=json.loads(value)
                if isinstance(decoded,str):return decoded
            except ValueError:
                pass
        return value
    if isinstance(value, list):
        return '\n'.join(str(b.get('text', '')) for b in value if isinstance(b, dict))
    return ''


def validate_config(config):
    models = config.get('models')
    if not isinstance(models, dict) or not models or any(not isinstance(m, str) or not m for m in models.values()):
        raise ValueError('models must map tier names to concrete model IDs')
    if config.get('default_tier') not in models:
        raise ValueError('default_tier must exist in models')
    for rule in config.get('rules', []):
        if rule['tier'] not in models or rule.get('field') not in ('last_message', 'last_user', 'full_context'):
            raise ValueError('Invalid routing rule field or tier')
        re.compile(rule['pattern'])
    return config


class RoutingPolicy:
    def __init__(self, config, config_dir=HERE):
        self.config = validate_config(config)
        self.custom = None
        if config.get('policy_module'):
            path = (Path(config_dir)/config['policy_module']).resolve()
            spec = importlib.util.spec_from_file_location('twinrouter_local_policy', path)
            if spec is None or spec.loader is None:
                raise ValueError('Cannot load policy_module')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.custom = module.select_model

    def select(self, body, session_id):
        messages = body['messages']
        if self.custom:
            result = self.custom(copy.deepcopy(body), {
                'session_id': session_id,
                'available_models': tuple(self.config['models'].values()),
                'tier_to_model': dict(self.config['models']),
            })
            if isinstance(result, str):
                result = {'model_id': result, 'reason': 'custom policy'}
        else:
            fields = {
                'last_message': text_content(messages[-1].get('content')) if messages else '',
                'last_user': next((text_content(m.get('content')) for m in reversed(messages) if m.get('role')=='user'), ''),
                'full_context': '\n'.join(text_content(m.get('content')) for m in messages),
            }
            selected = next((r for r in self.config.get('rules', []) if re.search(r['pattern'], fields[r['field']], re.I)), None)
            tier = selected['tier'] if selected else self.config['default_tier']
            result = {'model_id': self.config['models'][tier], 'reason': selected.get('name', 'rule') if selected else 'default '+tier}
        if not isinstance(result, dict) or result.get('model_id') not in self.config['models'].values():
            raise ValueError('Routing policy returned a model outside the configured pool')
        return {'model_id': result['model_id'], 'reason': str(result.get('reason', 'custom policy'))[:300]}


def usage_cost(usage):
    cost = usage.get('cost') if isinstance(usage, dict) else None
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
        return float(cost)
    return None


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, config, *, mode='mock', ledger=None, config_dir=HERE):
        self.config = validate_config(config)
        self.mode = mode
        self.policy = RoutingPolicy(config, config_dir)
        self.ledger = Path(ledger) if ledger else None
        self.ledger_lock = threading.Lock()
        self.upstream_key = ''
        self.local_token = os.environ.get(config.get('local_token_env', 'TWIN_ROUTER_TOKEN'), '')
        if mode == 'live':
            self.upstream_key = os.environ.get(config.get('api_key_env', 'ROUTER_UPSTREAM_API_KEY'), '')
            if not self.upstream_key:
                raise ValueError('Live mode requires the configured upstream API key environment variable')
            url = config.get('upstream_base_url', '')
            if not url.startswith('https://'):
                raise ValueError('Live upstream_base_url must use HTTPS')
        elif mode != 'mock':
            raise ValueError('Unknown mode')
        super().__init__(address, RouterHandler)

    def record(self, event):
        if self.ledger:
            with self.ledger_lock:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                # No prompts, tool arguments, authorization headers, or API keys.
                with self.ledger.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(event, ensure_ascii=False)+'\n')


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send_json(self, status, value, headers=None):
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/health':
            return self.send_json(200, {'status': 'ok', 'mode': self.server.mode})
        if self.path == '/v1/models':
            return self.send_json(200, {'object': 'list', 'data': [{'id': self.server.config.get('virtual_model', 'auto'), 'object': 'model', 'owned_by': 'twinrouterbench'}]})
        return self.send_json(404, {'error': {'message': 'Unknown endpoint'}})

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            return self.send_json(404, {'error': {'message': 'Use /v1/chat/completions'}})
        if self.server.local_token and self.headers.get('Authorization') != 'Bearer '+self.server.local_token:
            return self.send_json(401, {'error': {'message': 'Invalid local router token'}})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16*1024*1024:
                return self.send_json(413, {'error': {'message': 'Invalid request size'}})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body.get('messages'), list) or not body['messages'] or any(not isinstance(m, dict) or 'role' not in m for m in body['messages']):
                raise ValueError('messages must be a nonempty array of chat messages')
            if body.get('model') != self.server.config.get('virtual_model', 'auto'):
                raise ValueError('Use the configured virtual model name')
            if not isinstance(body.get('stream', False), bool):
                raise ValueError('stream must be boolean')
            session = self.headers.get('x-twinrouter-session')
            # Without the optional plugin there is no trustworthy session identity.
            session_hash = hashlib.sha256(session.encode()).hexdigest() if session else None
            decision = self.server.policy.select(body, session_hash)
        except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
            return self.send_json(400, {'error': {'message': 'Invalid chat request or routing decision'}})
        event = {'request_id': str(uuid.uuid4()), 'timestamp': datetime.now(timezone.utc).isoformat(),
                 'mode': self.server.mode, 'session_hash': session_hash,
                 'selected_model': decision['model_id'], 'policy_reason': decision['reason'],
                 'messages_count': len(body['messages']), 'usage': None, 'actual_cost_usd': None,
                 'status': 'started'}
        start = time.monotonic()
        try:
            if self.server.mode == 'mock':
                self.mock_reply(body, decision, event)
            else:
                self.forward(body, decision, event)
        except (BrokenPipeError, ConnectionResetError):
            event['status'] = 'client_disconnected_cost_may_be_unknown'
        except urllib.error.HTTPError as exc:
            event['status'] = 'upstream_http_'+str(exc.code)
            # Upstream error bodies can echo request secrets. Never relay/log them.
            self.send_json(502, {'error': {'message': 'Upstream request failed', 'upstream_status': exc.code}})
        except (OSError, ValueError, TimeoutError):
            event['status'] = 'upstream_error_cost_may_be_unknown'
            if not getattr(self, '_stream_started', False):
                self.send_json(502, {'error': {'message': 'Upstream connection or response failed'}})
            else:
                self.close_connection = True
        finally:
            event['latency_ms'] = round((time.monotonic()-start)*1000, 2)
            self.server.record(event)

    def mock_reply(self, body, decision, event):
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost': 0}
        message = {'role': 'assistant', 'content': '[MOCK ONLY] Selected '+decision['model_id']+'. No model API was called.'}
        response = {'id': 'chatcmpl-'+event['request_id'], 'object': 'chat.completion', 'created': int(time.time()),
                    'model': decision['model_id'], 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}], 'usage': usage}
        if body.get('stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.send_header('X-TwinRouter-Model', decision['model_id'])
            self.end_headers()
            self._stream_started = True
            for chunk in [dict(response, object='chat.completion.chunk', choices=[{'index':0, 'delta':message, 'finish_reason':None}]),
                          dict(response, object='chat.completion.chunk', choices=[{'index':0, 'delta':{}, 'finish_reason':'stop'}])]:
                self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
            self.close_connection = True
        else:
            self.send_json(200, response, {'X-TwinRouter-Model': decision['model_id']})
        event.update(status='mock_complete', usage=usage, actual_cost_usd=0.)

    def forward(self, body, decision, event):
        body = copy.deepcopy(body)
        body['model'] = decision['model_id']
        if body.get('stream'):
            body['stream_options'] = dict(body.get('stream_options') or {}, include_usage=True)
        request = urllib.request.Request(self.server.config['upstream_base_url'].rstrip('/')+'/chat/completions',
                    data=json.dumps(body).encode(), method='POST',
                    headers={'Authorization':'Bearer '+self.server.upstream_key, 'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=self.server.config.get('timeout_seconds', 180)) as upstream:
            if not body.get('stream'):
                raw = upstream.read()
                result = json.loads(raw)
                event['usage'] = result.get('usage')
                event['actual_cost_usd'] = usage_cost(event['usage'])
                event['provider_model'] = result.get('model')
                self.send_response(upstream.status)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)))
                self.send_header('X-TwinRouter-Model',decision['model_id'])
                self.end_headers()
                self.wfile.write(raw)
            else:
                if 'text/event-stream' not in upstream.headers.get('Content-Type', ''):
                    raise ValueError('Expected upstream SSE response')
                self.send_response(upstream.status)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Connection','close')
                self.send_header('X-TwinRouter-Model',decision['model_id'])
                self.end_headers()
                self._stream_started = True
                buffer = b''
                while True:
                    chunk = upstream.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n',1)
                        if line.startswith(b'data:'):
                            try:
                                item = json.loads(line[5:])
                                if item.get('usage'):
                                    event['usage'] = item['usage']
                                    event['actual_cost_usd'] = usage_cost(item['usage'])
                                if item.get('model'):
                                    event['provider_model'] = item['model']
                                if item.get('error'):
                                    event['stream_error'] = True
                            except (ValueError, AttributeError):
                                pass
                    if len(buffer) > 8*1024*1024:
                        raise ValueError('Upstream SSE event too large')
                self.close_connection = True
        event['status'] = 'upstream_stream_error' if event.get('stream_error') else 'forwarded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'router.example.json')
    parser.add_argument('--mode', choices=['mock','live'], default='mock')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--ledger', type=Path, default=Path('routing_events.jsonl'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    server = RouterServer(('127.0.0.1',args.port),config,mode=args.mode,ledger=args.ledger,config_dir=args.config.parent)
    print(json.dumps({'url':'http://127.0.0.1:'+str(server.server_port)+'/v1', 'mode':args.mode, 'policy':'custom' if config.get('policy_module') else 'unvalidated_rules'},ensure_ascii=False),flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
