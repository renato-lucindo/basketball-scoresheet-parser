import unittest

from sumula_reader.jev import (
    JevDecisionEngine,
    TypeSafeAPIError,
    TypeSafeClient,
    TypeSafeConfig,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class JevTests(unittest.TestCase):
    def test_list_models_uses_bearer_token(self):
        session = FakeSession(
            [
                FakeResponse(
                    payload={
                        "models": [
                            {
                                "name": "jev-latest",
                                "description": "General model",
                                "release_date": "2026-09-15",
                            }
                        ]
                    }
                )
            ]
        )
        client = TypeSafeClient(
            TypeSafeConfig(api_key="secret"),
            session=session,
        )

        models = client.list_models()

        self.assertEqual(models[0].name, "jev-latest")
        method, url, kwargs = session.calls[0]
        self.assertEqual(method, "GET")
        self.assertEqual(url, "https://api.typesafe.ai/v1/models")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")

    def test_choice_request_is_typed_and_parsed(self):
        session = FakeSession(
            [
                FakeResponse(
                    payload={
                        "model": "jev-2026-09-15",
                        "answers": {
                            "team_foul_kind": {
                                "type": "choice",
                                "choice": "x",
                                "confidence": 0.91,
                                "probabilities": {
                                    "x": 0.91,
                                    "unused": 0.04,
                                    "ambiguous": 0.05,
                                },
                            }
                        },
                        "usage": {"input_tokens": 80, "output_tokens": 8},
                    }
                )
            ]
        )
        engine = JevDecisionEngine(
            TypeSafeClient(
                TypeSafeConfig(api_key="secret"),
                session=session,
            )
        )

        answer = engine.classify_team_foul(
            period=4,
            slot=3,
            evidence={"ink_ratio": 0.03, "diagonal_lines": 2},
            context={"slot_4": "x"},
        )

        self.assertEqual(answer.choice, "x")
        self.assertEqual(answer.confidence, 0.91)
        _, url, kwargs = session.calls[0]
        self.assertEqual(url, "https://api.typesafe.ai/v1/systemone")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "jev-latest")
        question = payload["questions"]["team_foul_kind"]
        self.assertEqual(question["type"], "choice")
        self.assertIn("ambiguous", question["criteria"])

    def test_http_error_does_not_leak_bearer_token(self):
        session = FakeSession([FakeResponse(status_code=401, text="invalid key")])
        client = TypeSafeClient(
            TypeSafeConfig(api_key="do-not-leak"),
            session=session,
        )

        with self.assertRaises(TypeSafeAPIError) as caught:
            client.list_models()

        self.assertNotIn("do-not-leak", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
