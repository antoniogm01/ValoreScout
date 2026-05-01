"""
models/risk_model.py
====================
ValoreScout — Modelo de Riesgo de Fichajes

Arquitectura:
  Capa 1: Valor de mercado dinámico via GBM + saltos de Poisson (lesión)
  Capa 2: Performance score ajustado por posición y minutos
  Capa 3: Fit score club-jugador (similitud táctica + restricciones)
  Capa 4: Simulación Monte Carlo → ROI, VaR, probabilidad de éxito

Uso:
    python models/risk_model.py --player "Kylian Mbappé" --fee 180 --salary 25 --years 4
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("valorescout.risk")

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "master_players.parquet"

# ---------------------------------------------------------------------------
# Parámetros de mercado — calibrados sobre datos históricos de Transfermarkt
# ---------------------------------------------------------------------------

# Drift anual por grupo de edad (apreciación/depreciación esperada del valor)
AGE_DRIFT = {
    (16, 21): +0.20,   # jóvenes: alta apreciación esperada
    (22, 25): +0.10,   # prime ascendente
    (26, 28): +0.02,   # prime estable
    (29, 31): -0.10,   # declive moderado
    (32, 40): -0.22,   # declive acelerado
}

# Volatilidad anual del valor de mercado por posición
SIGMA_BY_POS = {
    "FW":      0.35,
    "MF":      0.28,
    "DF":      0.22,
    "GK":      0.18,
    "default": 0.28,
}

# Intensidad de lesiones graves (eventos Poisson por año) por historial
INJURY_LAMBDA = {
    "low":    0.08,   # historial limpio
    "medium": 0.18,   # alguna lesión previa
    "high":   0.35,   # historial preocupante
}

# Impacto de lesión grave sobre el valor de mercado
INJURY_IMPACT_MEAN  = -0.35   # -35% de media
INJURY_IMPACT_STD   =  0.12   # con incertidumbre


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlayerProfile:
    """Perfil completo de un jugador para el modelo de riesgo."""
    name:              str
    age:               int
    position:          str          # FW, MF, DF, GK
    market_value_m:    float        # valor de mercado actual en M€
    performance_score: float        # score normalizado [-3, +3]
    availability_rate: float        # fracción de minutos jugados [0,1]
    injury_history:    str = "medium"   # low / medium / high
    league_origin:     str = "unknown"

    def __post_init__(self):
        self.position = self.position.upper().split(",")[0].strip()
        if self.position not in SIGMA_BY_POS:
            self.position = "default"


@dataclass
class TransferDeal:
    """Parámetros económicos del fichaje."""
    fee_m:        float   # coste del traspaso en M€
    salary_m:     float   # salario anual en M€
    contract_years: int   # duración del contrato

    @property
    def total_cost_m(self) -> float:
        return self.fee_m + self.salary_m * self.contract_years


@dataclass
class ClubProfile:
    """Perfil táctico y económico del club comprador."""
    name:                str
    avg_performance_score: float   # score medio del equipo
    salary_cap_m:          float   # salario máximo que puede pagar
    tactical_style:        str     # attacking / balanced / defensive
    squad_age:             float   # edad media del equipo


@dataclass
class RiskResult:
    """Resultado completo del análisis de riesgo."""
    player_name:    str
    club_name:      str
    fee_m:          float
    fair_value_m:   float
    expected_roi_m: float
    var_95_m:       float       # VaR al 95% (pérdida máxima esperada)
    prob_positive:  float       # P(ROI > 0)
    prob_recoup:    float       # P(recuperar inversión total)
    fit_score:      float       # [0, 1]
    verdict:        str         # COMPRAR / NEUTRAL / EVITAR
    simulated_outcomes: np.ndarray = field(repr=False, default=None)


# ---------------------------------------------------------------------------
# Capa 1: Modelo de valor dinámico — GBM + Poisson
# ---------------------------------------------------------------------------

class PlayerValueModel:
    """
    Modela la evolución del valor de mercado V(t) de un jugador como
    GBM con saltos de Poisson representando lesiones graves:

        dV = μ·V·dt + σ·V·dW + J·V·dN

    donde:
        μ  = drift calibrado por edad y posición
        σ  = volatilidad calibrada por posición
        dW = incremento Browniano
        J  = impacto de lesión ~ N(μ_J, σ_J)
        dN = proceso de Poisson con intensidad λ(historial)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _get_drift(self, age: int) -> float:
        for (lo, hi), drift in AGE_DRIFT.items():
            if lo <= age <= hi:
                return drift
        return -0.15

    def _get_sigma(self, position: str) -> float:
        return SIGMA_BY_POS.get(position, SIGMA_BY_POS["default"])

    def _get_lambda(self, injury_history: str) -> float:
        return INJURY_LAMBDA.get(injury_history, INJURY_LAMBDA["medium"])

    def simulate(
        self,
        player:   PlayerProfile,
        deal:     TransferDeal,
        n_paths:  int = 100_000,
        n_steps:  int = 52,        # pasos semanales
    ) -> np.ndarray:
        """
        Simula N trayectorias del valor de mercado del jugador
        durante la duración del contrato.

        Retorna array de shape (n_paths,) con valores terminales en M€.
        """
        T        = deal.contract_years
        dt       = T / n_steps
        mu       = self._get_drift(player.age)
        sigma    = self._get_sigma(player.position)
        lam      = self._get_lambda(player.injury_history)
        V0       = player.market_value_m

        # GBM — simulación exacta por pasos
        V = np.full(n_paths, V0, dtype=float)

        for _ in range(n_steps):
            # Componente GBM
            Z   = self.rng.standard_normal(n_paths)
            gbm = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

            # Componente Poisson — lesiones
            n_events = self.rng.poisson(lam * dt, n_paths)
            injury_impact = np.where(
                n_events > 0,
                np.exp(self.rng.normal(INJURY_IMPACT_MEAN, INJURY_IMPACT_STD, n_paths)),
                1.0
            )

            V = V * gbm * injury_impact

        # Ajuste por rendimiento — jugadores con alto performance score
        # tienen menor probabilidad de depreciación extrema
        perf_adj = 1.0 + 0.05 * np.clip(player.performance_score, -2, 2)
        V *= perf_adj

        # Ajuste por disponibilidad — jugadores lesionables deprecian más
        avail_adj = 0.7 + 0.3 * player.availability_rate
        V *= avail_adj

        return np.maximum(V, 0)


# ---------------------------------------------------------------------------
# Capa 3: Fit Score
# ---------------------------------------------------------------------------

class FitScorer:
    """
    Calcula el fit score club-jugador en [0, 1].

    Dimensiones:
      1. Rendimiento relativo: ¿el jugador mejora el nivel del equipo?
      2. Salarial: ¿el salario pedido es asumible para el club?
      3. Edad: ¿encaja en la pirámide de edad del equipo?
      4. Táctica: proxy por estilo de juego vs posición del jugador
    """

    def score(
        self,
        player: PlayerProfile,
        deal:   TransferDeal,
        club:   ClubProfile,
    ) -> tuple[float, dict]:

        scores = {}

        # 1. Rendimiento relativo
        delta_perf = player.performance_score - club.avg_performance_score
        scores["performance"] = float(norm.cdf(delta_perf, loc=0, scale=0.5))

        # 2. Salarial
        salary_ratio = deal.salary_m / max(club.salary_cap_m, 0.1)
        scores["salary"] = float(np.clip(1.0 - salary_ratio, 0, 1))

        # 3. Edad — penalizar jugadores mayores de 30 para contratos largos
        age_penalty = max(0, (player.age - 28) * 0.05 * deal.contract_years)
        scores["age"] = float(np.clip(1.0 - age_penalty, 0, 1))

        # 4. Táctica — simplificado: attacking clubs quieren FW/MF
        tactical_map = {
            ("attacking",  "FW"): 1.0,
            ("attacking",  "MF"): 0.8,
            ("attacking",  "DF"): 0.5,
            ("balanced",   "FW"): 0.8,
            ("balanced",   "MF"): 0.9,
            ("balanced",   "DF"): 0.8,
            ("defensive",  "FW"): 0.5,
            ("defensive",  "MF"): 0.8,
            ("defensive",  "DF"): 1.0,
        }
        scores["tactical"] = tactical_map.get(
            (club.tactical_style, player.position), 0.7
        )

        # 5. Disponibilidad histórica
        scores["availability"] = player.availability_rate

        # Fit score ponderado
        weights = {
            "performance":  0.30,
            "salary":       0.25,
            "age":          0.20,
            "tactical":     0.15,
            "availability": 0.10,
        }
        fit = sum(weights[k] * scores[k] for k in weights)

        return round(fit, 4), scores


# ---------------------------------------------------------------------------
# Capa 4: Motor de riesgo Monte Carlo
# ---------------------------------------------------------------------------

class TransferRiskEngine:
    """
    Motor principal de valoración y riesgo de fichajes.

    Para cada simulación i:
        Outcome(i) = V_terminal(i) + Performance_value(i) - Total_cost

    donde Performance_value = performance_score × salary × years × multiplier
    """

    def __init__(self, n_simulations: int = 100_000, seed: int = 42):
        self.n   = n_simulations
        self.val = PlayerValueModel(seed=seed)
        self.fit = FitScorer()

    def analyse(
        self,
        player: PlayerProfile,
        deal:   TransferDeal,
        club:   ClubProfile,
    ) -> RiskResult:

        log.info(f"Analizando fichaje: {player.name} → {club.name}")
        log.info(f"  Fee: {deal.fee_m}M€ | Salario: {deal.salary_m}M€/año | Contrato: {deal.contract_years} años")
        log.info(f"  Coste total: {deal.total_cost_m:.1f}M€")

        # Simulación de valor terminal
        V_terminal = self.val.simulate(player, deal, n_paths=self.n)

        # Valor por rendimiento deportivo (beneficio intangible)
        # Modelado como función del performance score y la disponibilidad
        perf_multiplier = 1.0 + 0.15 * np.clip(player.performance_score, -2, 2)
        perf_value = (
            player.performance_score
            * deal.salary_m
            * deal.contract_years
            * perf_multiplier
            * player.availability_rate
        )

        # Outcome total: valor de reventa + valor deportivo - coste total
        outcomes = V_terminal + max(perf_value, 0) - deal.total_cost_m

        # Métricas de riesgo
        fair_value    = float(V_terminal.mean())
        expected_roi  = float(outcomes.mean())
        var_95        = float(np.percentile(outcomes, 5))   # VaR al 95%
        prob_positive = float((outcomes > 0).mean())
        prob_recoup   = float((V_terminal >= deal.fee_m).mean())

        # Fit score
        fit, fit_breakdown = self.fit.score(player, deal, club)

        # Veredicto
        verdict = self._verdict(expected_roi, var_95, fit, deal)

        result = RiskResult(
            player_name         = player.name,
            club_name           = club.name,
            fee_m               = deal.fee_m,
            fair_value_m        = fair_value,
            expected_roi_m      = expected_roi,
            var_95_m            = var_95,
            prob_positive       = prob_positive,
            prob_recoup         = prob_recoup,
            fit_score           = fit,
            verdict             = verdict,
            simulated_outcomes  = outcomes,
        )

        self._print_report(result, deal, fit_breakdown)
        return result

    def _verdict(
        self,
        roi:  float,
        var:  float,
        fit:  float,
        deal: TransferDeal,
    ) -> str:
        # Normalizar por tamaño del deal
        roi_ratio = roi / max(deal.total_cost_m, 1)
        var_ratio = var / max(deal.total_cost_m, 1)

        if roi_ratio > 0.10 and var_ratio > -0.40 and fit > 0.60:
            return "✅ COMPRAR"
        elif roi_ratio < -0.15 or var_ratio < -0.65 or fit < 0.35:
            return "❌ EVITAR"
        else:
            return "⚠️  NEUTRAL"

    def _print_report(self, r: RiskResult, deal: TransferDeal, fit_breakdown: dict):
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  VALORESCOUT — INFORME DE RIESGO DE FICHAJE")
        print(sep)
        print(f"  Jugador      : {r.player_name}")
        print(f"  Club         : {r.club_name}")
        print(f"  Fee          : {r.fee_m:.1f}M€")
        print(f"  Coste total  : {deal.total_cost_m:.1f}M€  "
              f"(salario {deal.salary_m}M€/año × {deal.contract_years} años)")
        print(sep)
        print(f"  Fair value   : {r.fair_value_m:.1f}M€")
        print(f"  ROI esperado : {r.expected_roi_m:+.1f}M€")
        print(f"  VaR 95%      : {r.var_95_m:+.1f}M€")
        print(f"  P(ROI > 0)   : {r.prob_positive:.1%}")
        print(f"  P(recuperar fee): {r.prob_recoup:.1%}")
        print(sep)
        print(f"  Fit score    : {r.fit_score:.2f} / 1.00")
        for k, v in fit_breakdown.items():
            bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
            print(f"    {k:14s} [{bar}] {v:.2f}")
        print(sep)
        print(f"  VEREDICTO    : {r.verdict}")
        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Cargador de datos reales
# ---------------------------------------------------------------------------

def load_player_from_data(name: str) -> PlayerProfile | None:
    """Carga un jugador del dataset maestro por nombre (búsqueda aproximada)."""
    if not DATA_PATH.exists():
        log.error(f"Dataset no encontrado: {DATA_PATH}")
        return None

    df = pd.read_parquet(DATA_PATH)

    # Búsqueda case-insensitive
    mask = df["player"].str.contains(name, case=False, na=False)
    matches = df[mask]

    if matches.empty:
        log.warning(f"Jugador '{name}' no encontrado. Jugadores disponibles con nombre similar:")
        similar = df[df["player"].str.contains(name.split()[0], case=False, na=False)]["player"].tolist()
        for p in similar[:5]:
            print(f"  → {p}")
        return None

    row = matches.iloc[0]
    log.info(f"Jugador encontrado: {row['player']} ({row.get('team','?')}, {row.get('league','?')})")

    # Estimar valor de mercado desde performance score (proxy hasta tener TM)
    perf  = float(row.get("performance_score", 0))
    v_est = max(5.0, 20.0 + perf * 15.0)   # proxy muy simplificado

    pos_raw = str(row.get("pos", "MF"))
    pos     = pos_raw.split(",")[0].upper().strip()

    return PlayerProfile(
        name              = row["player"],
        age               = int(row.get("age", 25)),
        position          = pos,
        market_value_m    = v_est,
        performance_score = perf,
        availability_rate = float(row.get("availability_rate", 0.7)),
        injury_history    = "medium",
        league_origin     = str(row.get("league", "unknown")),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ValoreScout — análisis de riesgo de fichaje")
    p.add_argument("--player",  type=str,   required=True,  help="Nombre del jugador")
    p.add_argument("--club",    type=str,   default="Mi Club")
    p.add_argument("--fee",     type=float, required=True,  help="Fee de traspaso en M€")
    p.add_argument("--salary",  type=float, required=True,  help="Salario anual en M€")
    p.add_argument("--years",   type=int,   default=4,      help="Años de contrato")
    p.add_argument("--injury",  type=str,   default="medium",
                   choices=["low", "medium", "high"], help="Historial de lesiones")
    p.add_argument("--tactical", type=str,  default="balanced",
                   choices=["attacking", "balanced", "defensive"])
    p.add_argument("--n",       type=int,   default=100_000, help="Número de simulaciones")
    return p.parse_args()


def main():
    args = parse_args()

    # Cargar jugador
    player = load_player_from_data(args.player)
    if player is None:
        return

    # Aplicar historial de lesiones desde CLI
    player.injury_history = args.injury

    # Deal
    deal = TransferDeal(
        fee_m           = args.fee,
        salary_m        = args.salary,
        contract_years  = args.years,
    )

    # Club (simplificado — en el dashboard será interactivo)
    club = ClubProfile(
        name                  = args.club,
        avg_performance_score = 0.0,
        salary_cap_m          = args.salary * 1.5,
        tactical_style        = args.tactical,
        squad_age             = 26.0,
    )

    # Análisis
    engine = TransferRiskEngine(n_simulations=args.n)
    engine.analyse(player, deal, club)


if __name__ == "__main__":
    main()