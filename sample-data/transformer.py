import json
import trace_explorer.transformer


class Transformer(trace_explorer.transformer.Transformer):
    def columns(self):
        return ['scan', 'join', 'filter']

    def transform(self, content: str):
        obj = json.loads(content)
        return [obj['scan'], obj['join'], obj['filter']]
