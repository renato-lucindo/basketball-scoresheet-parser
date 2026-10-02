from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class NormalizedRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Coordenadas normalizadas devem estar entre 0 e 1")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("A regiao ultrapassa os limites normalizados")

    def pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        left = round(self.x * image_width)
        top = round(self.y * image_height)
        right = round((self.x + self.width) * image_width)
        bottom = round((self.y + self.height) * image_height)
        return left, top, right, bottom


@dataclass(slots=True)
class TemplateSpec:
    template_id: str
    canonical_width: int
    canonical_height: int
    regions: dict[str, NormalizedRect] = field(default_factory=dict)

    def region(self, name: str) -> NormalizedRect:
        try:
            return self.regions[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.regions))
            raise KeyError(f"Regiao desconhecida: {name}. Disponiveis: {available}") from exc


FECABA_V1 = TemplateSpec(
    template_id="fecaba_v1",
    canonical_width=2480,
    canonical_height=3508,
    # Coordenadas medidas dentro da borda externa da sumula branca FECABA.
    # A normalizacao geometrica mapeia essa borda para o canvas canonico,
    # portanto as ROIs continuam estaveis mesmo com margens diferentes.
    regions={
        "team_a_roster": NormalizedRect(0.000, 0.212, 0.492, 0.232),
        "team_a_jersey": NormalizedRect(0.320, 0.212, 0.042, 0.232),
        "team_a_participation": NormalizedRect(0.362, 0.212, 0.022, 0.232),
        "team_a_player_fouls": NormalizedRect(0.384, 0.212, 0.108, 0.232),
        "team_a_team_fouls": NormalizedRect(0.276, 0.166, 0.216, 0.045),
        "team_b_roster": NormalizedRect(0.000, 0.561, 0.492, 0.222),
        "team_b_jersey": NormalizedRect(0.320, 0.561, 0.042, 0.222),
        "team_b_participation": NormalizedRect(0.362, 0.561, 0.022, 0.222),
        "team_b_player_fouls": NormalizedRect(0.384, 0.561, 0.108, 0.222),
        "team_b_team_fouls": NormalizedRect(0.276, 0.466, 0.216, 0.045),
        "scoring_table": NormalizedRect(0.511, 0.151, 0.489, 0.645),
        "period_scores": NormalizedRect(0.000, 0.783, 0.492, 0.060),
        "final_score": NormalizedRect(0.511, 0.783, 0.489, 0.060),
        "apontador": NormalizedRect(0.000, 0.843, 0.492, 0.023),
    },
)


TEMPLATES: dict[str, TemplateSpec] = {
    FECABA_V1.template_id: FECABA_V1,
}


def get_template(template_id: str) -> TemplateSpec:
    try:
        return TEMPLATES[template_id]
    except KeyError as exc:
        available = ", ".join(sorted(TEMPLATES))
        raise ValueError(
            f"Template desconhecido: {template_id}. Disponiveis: {available}"
        ) from exc
