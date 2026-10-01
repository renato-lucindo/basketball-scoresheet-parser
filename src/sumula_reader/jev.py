from __future__ import annotations

import getpass
import os
from typing import Any, Literal, Mapping

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError


DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"


class TypeSafeAPIError(RuntimeError):
    pass


class TypeSafeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str = Field(min_length=1, repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = Field(default=20.0, gt=0)

    @classmethod
    def from_env(cls) -> "TypeSafeConfig":
        api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        if not api_key:
            raise TypeSafeAPIError(
                "TYPESAFE_API_KEY nao foi definida no ambiente"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("TYPESAFE_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("TYPESAFE_MODEL", DEFAULT_MODEL),
        )


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelListResponse(BaseModel):
    models: list[ModelMetadata]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: Any | None = None
    criteria: dict[str, Any]


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    model: str
    questions: dict[str, ChoiceQuestion]


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: Usage

    def choice_answer(self, name: str) -> ChoiceAnswer:
        try:
            payload = self.answers[name]
        except KeyError as exc:
            raise TypeSafeAPIError(
                f"Resposta TypeSafe nao contem a pergunta {name!r}"
            ) from exc
        return ChoiceAnswer.model_validate(payload)


class TypeSafeClient:
    def __init__(
        self,
        config: TypeSafeConfig,
        *,
        session: requests.Session | Any | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "TypeSafeClient":
        return cls(TypeSafeConfig.from_env())

    def list_models(self) -> list[ModelMetadata]:
        payload = self._request_json("GET", "/v1/models")
        try:
            return ModelListResponse.model_validate(payload).models
        except ValidationError as exc:
            raise TypeSafeAPIError(
                "TypeSafe retornou uma lista de modelos em formato inesperado"
            ) from exc

    def choose(
        self,
        *,
        state: str | dict[str, Any] | list[Any],
        name: str,
        criteria: Mapping[str, Any],
        instructions: Any | None = None,
        model: str | None = None,
    ) -> ChoiceAnswer:
        request = SystemOneRequest(
            state=state,
            model=model or self.config.model,
            questions={
                name: ChoiceQuestion(
                    instructions=instructions,
                    criteria=dict(criteria),
                )
            },
        )
        payload = self._request_json(
            "POST",
            "/v1/systemone",
            json=request.model_dump(exclude_none=True),
        )
        try:
            response = SystemOneResponse.model_validate(payload)
            return response.choice_answer(name)
        except ValidationError as exc:
            raise TypeSafeAPIError(
                "TypeSafe retornou uma resposta System One em formato inesperado"
            ) from exc

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Accept": "application/json",
        }
        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                json=json,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TypeSafeAPIError(f"Falha de rede ao chamar TypeSafe: {exc}") from exc

        if not 200 <= response.status_code < 300:
            body = response.text.strip()
            if len(body) > 500:
                body = body[:500] + "..."
            raise TypeSafeAPIError(
                f"TypeSafe retornou HTTP {response.status_code}: {body}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise TypeSafeAPIError("TypeSafe retornou JSON invalido") from exc
        if not isinstance(payload, dict):
            raise TypeSafeAPIError("TypeSafe retornou payload inesperado")
        return payload


class JevDecisionEngine:
    TEAM_FOUL_CRITERIA = {
        "x": "Ha um X que representa uma falta coletiva registrada nesta caixa.",
        "unused": (
            "A caixa foi inutilizada/fechada por tracos horizontais e nao "
            "representa uma nova falta."
        ),
        "ambiguous": "A evidencia nao permite distinguir com seguranca X e fechamento.",
    }
    SCORING_CRITERIA = {
        "free_throw": "A marca representa um lance livre convertido, valendo 1 ponto.",
        "two_point": "A marca representa uma cesta de 2 pontos.",
        "three_point": "A marca representa uma cesta de 3 pontos.",
        "closure_stroke": (
            "O traco pertence ao fechamento/encerramento da contagem e nao e uma cesta."
        ),
        "ambiguous": "A evidencia nao permite classificar a marca com seguranca.",
    }

    def __init__(self, client: TypeSafeClient) -> None:
        self.client = client

    @classmethod
    def from_env(cls) -> "JevDecisionEngine":
        return cls(TypeSafeClient.from_env())

    @classmethod
    def from_env_or_prompt(
        cls,
        *,
        model: str | None = None,
    ) -> "JevDecisionEngine":
        api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        if not api_key:
            api_key = getpass.getpass("TypeSafe API key: ").strip()
        if not api_key:
            raise TypeSafeAPIError("Chave TypeSafe nao informada")
        selected_model = (
            model
            or os.getenv("TYPESAFE_MODEL", "").strip()
            or DEFAULT_MODEL
        )
        return cls(
            TypeSafeClient(
                TypeSafeConfig(api_key=api_key, model=selected_model)
            )
        )

    def classify_team_foul(
        self,
        *,
        period: int,
        slot: int,
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> ChoiceAnswer:
        state = {
            "field": "team_foul_cell",
            "period": period,
            "slot": slot,
            "evidence": dict(evidence),
            "context": dict(context or {}),
        }
        return self.client.choose(
            state=state,
            name="team_foul_kind",
            instructions=(
                "Classifique a caixa usando somente as evidencias fornecidas. "
                "Marcas fracas ou contraditorias devem permanecer ambiguous."
            ),
            criteria=self.TEAM_FOUL_CRITERIA,
        )

    def classify_scoring_mark(
        self,
        *,
        team: str,
        running_score: int,
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> ChoiceAnswer:
        state = {
            "field": "scoring_mark",
            "team": team.upper(),
            "running_score": running_score,
            "evidence": dict(evidence),
            "context": dict(context or {}),
        }
        return self.client.choose(
            state=state,
            name="scoring_kind",
            instructions=(
                "Classifique a marca da sumula usando somente as evidencias "
                "fornecidas e respeite a semantica da contagem corrente."
            ),
            criteria=self.SCORING_CRITERIA,
        )


def main() -> None:
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("TypeSafe API key: ").strip()
    if not api_key:
        raise SystemExit("Chave TypeSafe nao informada")

    client = TypeSafeClient(TypeSafeConfig(api_key=api_key))
    for model in client.list_models():
        print(f"{model.name}\t{model.release_date}\t{model.description}")


if __name__ == "__main__":
    main()
