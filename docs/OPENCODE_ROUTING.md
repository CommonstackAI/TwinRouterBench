# Per-call routing with OpenCode

OpenCode can use the local routing provider as a custom OpenAI-compatible
provider. For every model request, the provider gives the latest messages and
tool definitions to a routing policy and forwards the request to its selected
model. After a tool returns, the next request is routed again.

The included rules demonstrate the interface. To use an existing router, supply
a Python policy as described below. The example rules and policy have no reported
SWE-bench resolve rate or cost reduction.

## Try the interface without model API charges

From the repository root, install the package and start the mock provider:

```sh
pip install -e .
python -m twinrouterbench.opencode_router --mode mock
```

In another terminal, point OpenCode at the example provider configuration:

```sh
OPENCODE_CONFIG="$PWD/twinrouterbench/opencode_router/opencode.example.json" \
  opencode run --model twinrouter/auto 'list files'
```

The mock provider reports its selected model without calling an upstream model.
For an actual OpenCode tool round trip using synthetic model responses, run:

```sh
python -m twinrouterbench.opencode_router.smoke --binary /path/to/opencode
```

The smoke test creates a temporary project and isolated OpenCode configuration,
emits a shell tool call that prints a marker, checks the returned observation,
and checks that the subsequent request selects a different model. The test
writes `opencode_smoke_result.json` in the current directory; use `--output` to
choose another path. OpenCode may also issue auxiliary requests such as title
generation; these appear in the provider ledger too.

The tool round trip was verified with OpenCode 1.19.2. A separate live API
connectivity check verified a model switch across a tool response while reading
and writing temporary files. These checks exercise the interface and continued
interaction; running an end-to-end benchmark requires its agent/environment
adapter and official task evaluator.

## Forward to a real provider

Real model calls incur upstream API charges. Copy `router.example.json` to a
local configuration, set its HTTPS `upstream_base_url` and concrete model IDs,
and make the named `api_key_env` available in the provider process. The default
environment variable is `ROUTER_UPSTREAM_API_KEY`; credentials stay out of JSON
configuration and logs.

```sh
python -m twinrouterbench.opencode_router \
  --config /path/to/router.local.json --mode live
```

The server listens on `127.0.0.1` and exposes `/health`, `/v1/models`, and
`/v1/chat/completions`. OpenCode continues to select the virtual model
`twinrouter/auto`. The provider selects the concrete model separately on each
request, preserving messages, tool definitions, tool choices, and tool-call
responses. Both JSON and streaming SSE responses are supported.

Check that every model in the configured pool supports the client's context
length, output limit, and tools. The client advertises one context/output limit
for the virtual model; this must fit the models your policy can select.
The provider does not truncate conversations or translate provider-specific
reasoning formats.

## Supply a routing policy

Add `"policy_module": "my_policy.py"` to the local router configuration. Paths
are resolved relative to that configuration file. The module defines:

```python
def select_model(request, context):
    # request contains the latest messages, tools, and other request fields.
    # Replace this example with your classifier or routing implementation.
    return {
        "model_id": context["tier_to_model"]["high"],
        "reason": "example always-high policy",
    }
```

`context` contains `available_models`, `tier_to_model`, and `session_id`.
The return value can also be a model ID string. A selected model must belong
to the configured pool. The provider supplies a copy of the request so policy
changes cannot modify the forwarded conversation. Example policies run locally;
an LLM-based policy can add its own model calls and associated costs.

Without `policy_module`, ordered rules match `last_message`, `last_user`, or
`full_context`, then fall back to `default_tier`.

An optional `session_header_plugin.mjs` uses OpenCode's `chat.headers` hook to
send a session identifier. Follow the [OpenCode plugin configuration](https://opencode.ai/docs/plugins/)
to load its absolute file path. The provider hashes this identifier before
passing it to the policy and recording it. Without the plugin, `session_id`
is `None`; the provider still routes every request using its full prefix.

## Logs and local authentication

`routing_events.jsonl` records the chosen model, policy reason, request latency,
status, and returned usage. A numeric, finite, nonnegative upstream `usage.cost`
is recorded as `actual_cost_usd`; missing cost remains `null`. This ledger does
not estimate cache savings or substitute a catalog price for an actual charge.
Prompts, tool arguments, and authorization headers are excluded from the
provider's ledger. A custom policy should keep its `reason` free of sensitive
content. Consult the upstream provider's usage records for final billing.

For optional local authentication, set `TWIN_ROUTER_TOKEN` in the provider
environment and set the OpenCode provider `apiKey` to the same value. Otherwise
the local endpoint accepts requests from clients that can reach its loopback
port. The upstream key is used only for forwarding to the configured provider.

## Tests

```sh
python -m unittest discover -s tests -p test_opencode_router.py -v
```

The tests cover per-request model selection after tool feedback, request
preservation, JSON and streaming tool calls, custom-policy pool validation,
cost accounting, and ledger contents. They do not make paid model requests.

OpenCode configuration references:
[providers](https://opencode.ai/docs/providers/),
[models](https://opencode.ai/docs/models/),
[plugins](https://opencode.ai/docs/plugins/).
