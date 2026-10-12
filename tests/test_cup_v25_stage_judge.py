"""V25 malformed model output must remain evidence before any parser failure."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/deployment'))
from cup_v25_compact_judge import OpenRouterJudge, JudgeError

class Response:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, limit): return self.body

@pytest.mark.parametrize('raw', [b'not envelope JSON', b'{"choices":[{"message":{"content":"not judgement JSON"}}]}'])
def test_raw_saved_before_envelope_or_content_parse(tmp_path, raw):
    key = tmp_path / 'key'; key.write_text('test-secret')
    observed = []
    judge = OpenRouterJudge(key, maximum_calls=1, opener=lambda *args, **kwargs:Response(raw), response_recorder=observed.append)
    compact = SimpleNamespace(context={'requested_facts':['holding'], 'frames':[]}, images=[], local_bindings={'frames':[], 'request_sha256':'bound'})
    with pytest.raises(JudgeError): judge.ask(compact)
    assert len(observed) == 1
    assert observed[0]['http_response_utf8'] == raw.decode()
    assert observed[0]['recorded_before_json_parsing'] is True
    assert observed[0]['secret_present_after_redaction'] is False
    assert judge.calls == 1
