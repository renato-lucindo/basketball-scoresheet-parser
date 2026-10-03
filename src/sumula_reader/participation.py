from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imaging import crop_region
from .models import DecisionStatus, ParticipantMark, PlayerResult
from .template import FECABA_V1, TemplateSpec


@dataclass(slots=True)
class ParticipationDetection:
    mark: ParticipantMark
    participated: bool
    starter: bool
    participation_confidence: float
    starter_confidence: float
    status: DecisionStatus
    blue_pixel_ratio: float
    red_pixel_ratio: float
    red_ring_ratio: float


@dataclass(frozen=True, slots=True)
class ParticipationGridSpec:
    header_fraction: float = 0.085
    roster_rows: int = 12


PARTICIPATION_GRID = ParticipationGridSpec()


def _as_rgb_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("A imagem deve ter formato HxWx3 (RGB)")
    return arr[..., :3].astype(np.int16, copy=False)


def _color_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    colored = saturation >= 35

    blue_mask = colored & (blue >= red + 20) & (blue >= green + 10) & (blue >= 70)
    red_mask = colored & (red >= green + 25) & (red >= blue + 20) & (red >= 80)
    return blue_mask, red_mask


def detect_participation(
    image: np.ndarray,
    *,
    min_ink_ratio: float = 0.008,
    starter_ring_ratio: float = 0.015,
) -> ParticipationDetection:
    """Classifica a celula de participacao do atleta.

    Regras do MVP:
    - X azul ou vermelho: participou;
    - X azul envolvido por circulo vermelho: titular;
    - sem tinta suficiente: nao participou.

    O detector usa apenas distribuicao de cor nesta primeira versao. A etapa de
    geometria do template garantira que o crop recebido contenha somente a
    celula de participacao.
    """
    rgb = _as_rgb_array(image)
    blue_mask, red_mask = _color_masks(rgb)

    h, w = blue_mask.shape
    total = float(h * w)
    blue_ratio = float(blue_mask.sum() / total)
    red_ratio = float(red_mask.sum() / total)

    yy, xx = np.ogrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    nx = (xx - cx) / max(w / 2.0, 1.0)
    ny = (yy - cy) / max(h / 2.0, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    ring = (radius >= 0.42) & (radius <= 0.92)
    ring_pixels = max(int(ring.sum()), 1)
    red_ring_ratio = float((red_mask & ring).sum() / ring_pixels)

    has_blue = blue_ratio >= min_ink_ratio
    has_red = red_ratio >= min_ink_ratio
    participated = has_blue or has_red
    starter = has_blue and has_red and red_ring_ratio >= starter_ring_ratio

    if starter:
        mark = ParticipantMark.BLUE_X_RED_CIRCLE
    elif has_blue and not has_red:
        mark = ParticipantMark.BLUE_X
    elif has_red and not has_blue:
        mark = ParticipantMark.RED_X
    elif has_blue and has_red:
        mark = ParticipantMark.AMBIGUOUS
    else:
        mark = ParticipantMark.NONE

    participation_confidence = min(1.0, max(blue_ratio, red_ratio) / max(min_ink_ratio * 4.0, 1e-9)) if participated else min(1.0, 1.0 - max(blue_ratio, red_ratio) / max(min_ink_ratio, 1e-9))
    starter_confidence = min(1.0, red_ring_ratio / max(starter_ring_ratio * 2.0, 1e-9)) if has_blue else 0.0

    status = DecisionStatus.ACCEPTED
    if mark is ParticipantMark.AMBIGUOUS:
        status = DecisionStatus.REVIEW

    return ParticipationDetection(
        mark=mark,
        participated=participated,
        starter=starter,
        participation_confidence=round(float(participation_confidence), 4),
        starter_confidence=round(float(starter_confidence), 4),
        status=status,
        blue_pixel_ratio=round(blue_ratio, 6),
        red_pixel_ratio=round(red_ratio, 6),
        red_ring_ratio=round(red_ring_ratio, 6),
    )


def extract_players(
    normalized_image: np.ndarray,
    *,
    team: str,
    jerseys: list[int],
    template: TemplateSpec = FECABA_V1,
    grid: ParticipationGridSpec = PARTICIPATION_GRID,
) -> list[PlayerResult]:
    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team deve ser A ou B")
    if len(jerseys) > grid.roster_rows:
        raise ValueError(
            f"O template suporta no maximo {grid.roster_rows} linhas de jogadores"
        )

    block = crop_region(
        normalized_image,
        template.region(f"team_{side.lower()}_participation"),
    )
    data_top = round(block.shape[0] * grid.header_fraction)
    data = block[data_top:]
    row_height = data.shape[0] / grid.roster_rows
    players: list[PlayerResult] = []

    for index, jersey in enumerate(jerseys):
        top = round(index * row_height)
        bottom = round((index + 1) * row_height)
        detection = detect_participation(data[top:bottom])
        players.append(
            PlayerResult(
                jersey=jersey,
                participation_mark=detection.mark,
                participated=detection.participated,
                starter=detection.starter,
                participation_confidence=detection.participation_confidence,
                starter_confidence=detection.starter_confidence,
                participation_status=detection.status,
                starter_status=detection.status,
            )
        )
    return players
