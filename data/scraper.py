"""
data/scraper.py
===============
Pipeline de datos para ValoreScout.

Fuentes:
  - FBref via soccerdata  → métricas avanzadas de rendimiento
  - Transfermarkt         → valores de mercado históricos y fees

Uso:
    python data/scraper.py --league "Big 5" --season 2024
"""

import os
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import soccerdata as sd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("valorescout")

RAW_DIR  = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Ligas soportadas (notación soccerdata/FBref)
LEAGUES = {
    "ESP-La Liga":        "La Liga",
    "ENG-Premier League": "Premier League",
    "GER-Bundesliga":     "Bundesliga",
    "ITA-Serie A":        "Serie A",
    "FRA-Ligue 1":        "Ligue 1"
}


# ---------------------------------------------------------------------------
# FBref — métricas avanzadas
# ---------------------------------------------------------------------------

class FBrefPipeline:
    """
    Descarga métricas avanzadas de jugadores desde FBref via soccerdata.

    Métricas recogidas por jugador y temporada:
      - Goles, asistencias, xG, xAG
      - Progressive passes, progressive carries, progressive receptions
      - Tackles, interceptions, blocks (defensive)
      - Minutos jugados, partidos
    """

    def __init__(self, leagues: list[str], seasons: list[int]):
        self.leagues = leagues
        self.seasons = seasons
        self.fbref   = sd.FBref(leagues=leagues, seasons=seasons)

    def fetch_shooting(self) -> pd.DataFrame:
        log.info("FBref → shooting stats...")
        try:
            df = self.fbref.read_player_season_stats(stat_type="shooting")
            return df
        except Exception as e:
            log.error(f"Error fetching shooting: {e}")
            return pd.DataFrame()

    def fetch_passing(self) -> pd.DataFrame:
        log.info("FBref → passing stats...")
        try:
            df = self.fbref.read_player_season_stats(stat_type="passing")
            return df
        except Exception as e:
            log.error(f"Error fetching passing: {e}")
            return pd.DataFrame()

    def fetch_defense(self) -> pd.DataFrame:
        log.info("FBref → defensive stats...")
        try:
            df = self.fbref.read_player_season_stats(stat_type="defense")
            return df
        except Exception as e:
            log.error(f"Error fetching defense: {e}")
            return pd.DataFrame()

    def fetch_misc(self) -> pd.DataFrame:
        log.info("FBref → misc stats (cards, fouls, aerial)...")
        try:
            df = self.fbref.read_player_season_stats(stat_type="misc")
            return df
        except Exception as e:
            log.error(f"Error fetching misc: {e}")
            return pd.DataFrame()

    def fetch_all(self) -> pd.DataFrame:
        """Descarga y une todas las tablas por jugador/temporada/equipo."""
        dfs = {}
        for stat in ["shooting", "passing", "defense", "misc", "possession"]:
            log.info(f"FBref → {stat}...")
            try:
                dfs[stat] = self.fbref.read_player_season_stats(stat_type=stat)
                time.sleep(3)   # respetar rate limit de FBref
            except Exception as e:
                log.warning(f"  Skipping {stat}: {e}")

        if not dfs:
            log.error("No se pudo descargar ninguna tabla de FBref.")
            return pd.DataFrame()

        # Merge progresivo sobre índice común
        base = list(dfs.values())[0]
        for key, df in list(dfs.items())[1:]:
            try:
                base = base.join(df, how="outer", rsuffix=f"_{key}")
            except Exception as e:
                log.warning(f"  No se pudo hacer join de {key}: {e}")

        return base

    def save(self, df: pd.DataFrame, name: str = "fbref_players"):
        if df.empty:
            log.warning("DataFrame vacío — nada que guardar.")
            return
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path)
        log.info(f"Guardado: {path}  ({len(df)} filas)")


# ---------------------------------------------------------------------------
# Transfermarkt — valores de mercado
# ---------------------------------------------------------------------------

class TransfermarktPipeline:
    """
    Descarga valores de mercado desde Transfermarkt via soccerdata.

    Columnas clave:
      - player_name, player_id
      - market_value_eur  (valor actual en €)
      - date              (fecha de la valoración)
      - age, position, nationality
      - club, league
    """

    def __init__(self, leagues: list[str], seasons: list[int]):
        self.leagues = leagues
        self.seasons = seasons
        try:
            self.tm = sd.Transfermarkt(leagues=leagues, seasons=seasons)
        except Exception as e:
            log.error(f"Error inicializando Transfermarkt: {e}")
            self.tm = None

    def fetch_player_valuations(self) -> pd.DataFrame:
        if self.tm is None:
            return pd.DataFrame()
        log.info("Transfermarkt → player valuations...")
        try:
            df = self.tm.read_player_market_values()
            return df
        except Exception as e:
            log.error(f"Error fetching valuations: {e}")
            return pd.DataFrame()

    def fetch_transfers(self) -> pd.DataFrame:
        if self.tm is None:
            return pd.DataFrame()
        log.info("Transfermarkt → transfer history...")
        try:
            df = self.tm.read_transfers()
            return df
        except Exception as e:
            log.error(f"Error fetching transfers: {e}")
            return pd.DataFrame()

    def save(self, df: pd.DataFrame, name: str = "tm_valuations"):
        if df.empty:
            log.warning("DataFrame vacío — nada que guardar.")
            return
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path)
        log.info(f"Guardado: {path}  ({len(df)} filas)")


# ---------------------------------------------------------------------------
# Procesador — unifica FBref + Transfermarkt
# ---------------------------------------------------------------------------

class DataProcessor:
    """
    Une los datos de FBref y Transfermarkt en un dataset maestro por jugador.

    Columnas del dataset final:
      player_name, season, club, league, position, age,
      market_value_eur, minutes_played, availability_rate,
      xg_per90, xag_per90, progressive_passes_per90,
      progressive_carries_per90, tackles_per90, interceptions_per90,
      performance_score (compuesto)
    """

    def __init__(self):
        self.fbref_path = RAW_DIR / "fbref_players.parquet"
        self.tm_path    = RAW_DIR / "tm_valuations.parquet"

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        fbref = pd.read_parquet(self.fbref_path) if self.fbref_path.exists() else pd.DataFrame()
        tm    = pd.read_parquet(self.tm_path)    if self.tm_path.exists()    else pd.DataFrame()
        log.info(f"Cargado FBref: {len(fbref)} filas | TM: {len(tm)} filas")
        return fbref, tm

    def clean_fbref(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza columnas y calcula métricas per90."""
        if df.empty:
            return df

        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Minutos jugados — columna puede variar por versión
        min_col = next((c for c in df.columns if "min" in c and "90" not in c), None)
        if min_col:
            df["minutes_played"] = pd.to_numeric(df[min_col], errors="coerce").fillna(0)
            df["availability_rate"] = (df["minutes_played"] / (90 * 38)).clip(0, 1)

        # xG y xAG per90
        for raw, per90 in [("xg", "xg_per90"), ("xag", "xag_per90"),
                            ("prg_p", "progressive_passes_per90"),
                            ("prg_c", "progressive_carries_per90"),
                            ("tkl", "tackles_per90"),
                            ("int", "interceptions_per90")]:
            col = next((c for c in df.columns if c.startswith(raw)), None)
            if col and "minutes_played" in df.columns:
                df[per90] = (pd.to_numeric(df[col], errors="coerce") /
                             (df["minutes_played"] / 90)).replace([np.inf, -np.inf], np.nan)

        return df

    def compute_performance_score(self, df: pd.DataFrame,
                                   position: str = "all") -> pd.DataFrame:
        """
        Score compuesto de rendimiento normalizado por posición.

        Pesos por posición:
          FW: xg 40%, xag 20%, progressive 25%, defensive 15%
          MF: xg 20%, xag 25%, progressive 35%, defensive 20%
          DF: xg  5%, xag 10%, progressive 20%, defensive 65%
          GK: métricas específicas (pendiente)
        """
        weights = {
            "FW": dict(xg=0.40, xag=0.20, prog=0.25, def_=0.15),
            "MF": dict(xg=0.20, xag=0.25, prog=0.35, def_=0.20),
            "DF": dict(xg=0.05, xag=0.10, prog=0.20, def_=0.65),
        }
        w = weights.get(position.upper(), weights["MF"])

        metrics = {
            "xg":   "xg_per90",
            "xag":  "xag_per90",
            "prog": "progressive_passes_per90",
            "def_": "tackles_per90",
        }

        score = pd.Series(0.0, index=df.index)
        for key, col in metrics.items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                normed = (s - s.mean()) / (s.std() + 1e-8)
                score += w[key] * normed

        df["performance_score"] = score
        return df

    def merge(self, fbref: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
        """Une ambas fuentes por nombre de jugador (fuzzy si es necesario)."""
        if fbref.empty or tm.empty:
            log.warning("Una o ambas fuentes están vacías — merge parcial.")
            return fbref if not fbref.empty else tm

        # Intentar merge directo por nombre normalizado
        fbref["_key"] = fbref.index.get_level_values("player").str.lower().str.strip()
        tm["_key"]    = tm.index.get_level_values("player").str.lower().str.strip() \
                        if "player" in tm.index.names else tm.get("player_name", pd.Series()).str.lower()

        merged = fbref.merge(tm[["_key", "market_value_eur"]].drop_duplicates("_key"),
                             on="_key", how="left")
        log.info(f"Merge completado: {len(merged)} jugadores | "
                 f"{merged['market_value_eur'].notna().sum()} con valor de mercado")
        return merged

    def save(self, df: pd.DataFrame):
        path = PROC_DIR / "master_players.parquet"
        df.to_parquet(path)
        log.info(f"Dataset maestro guardado: {path}  ({len(df)} jugadores)")
        # Preview
        print("\n── Preview dataset maestro ──────────────────────────")
        preview_cols = [c for c in ["minutes_played", "xg_per90", "xag_per90",
                                     "performance_score", "market_value_eur"]
                        if c in df.columns]
        print(df[preview_cols].describe().round(3).to_string())
        print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ValoreScout — data pipeline")
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES.keys()),
                        help="Códigos de liga: ESP1 ENG1 GER1 ITA1 FRA1")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024],
                        help="Temporadas (año de inicio): 2022 2023 2024")
    parser.add_argument("--skip-fbref", action="store_true")
    parser.add_argument("--skip-tm",    action="store_true")
    parser.add_argument("--process-only", action="store_true",
                        help="Solo procesar datos ya descargados")
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"ValoreScout pipeline | ligas={args.leagues} | temporadas={args.seasons}")

    if not args.process_only:
        # FBref
        if not args.skip_fbref:
            fbref_pipe = FBrefPipeline(args.leagues, args.seasons)
            df_fbref   = fbref_pipe.fetch_all()
            fbref_pipe.save(df_fbref)
        else:
            log.info("Skipping FBref download.")

        # Transfermarkt
        if not args.skip_tm:
            tm_pipe   = TransfermarktPipeline(args.leagues, args.seasons)
            df_vals   = tm_pipe.fetch_player_valuations()
            tm_pipe.save(df_vals, "tm_valuations")
            df_trans  = tm_pipe.fetch_transfers()
            tm_pipe.save(df_trans, "tm_transfers")
        else:
            log.info("Skipping Transfermarkt download.")

    # Procesado
    processor = DataProcessor()
    fbref, tm = processor.load()

    if not fbref.empty:
        fbref = processor.clean_fbref(fbref)
        fbref = processor.compute_performance_score(fbref)

    master = processor.merge(fbref, tm)
    processor.save(master)

    log.info("Pipeline completado ✓")


if __name__ == "__main__":
    main()