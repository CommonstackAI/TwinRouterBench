"""Optional policy interface. Set policy_module in a local copy of the config.

Replace this function with your existing trained/local router. Do not call an
LLM merely to select a model unless you explicitly want that extra API cost.
"""
def select_model(request, context):
    # Full current prefix, including tool results, is available in request.
    messages = request['messages']
    assert messages
    return {'model_id': context['tier_to_model']['high'],
            'reason': 'Example always-high policy; replace with actual trained router'}
