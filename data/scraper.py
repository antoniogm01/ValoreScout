"""
data/scraper.py
===============
Pipeline de datos para ValoreScout — versión 2 corregida.

Cambios vs v1:
  - stat_types correctos para soccerdata: standard, shooting, playing_time, misc
  - Aplanado de MultiIndex columns (FBref devuelve columnas jerárquicas)
  - Transfermarkt via requests+BS4 (soccerdata no tiene TM)
  - Usa 'Big 5 European Leagues Combined' (más eficiente)

Uso:
    python data/scraper.py --seasons 2024
    python data/scraper.py --process-only   # solo procesar datos ya descargados
"""

import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
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

LEAGUE_MODE  = "Big 5 European Leagues Combined"
VALID_STATS  = ["standard", "shooting", "playing_time", "misc"]


# ---------------------------------------------------------------------------
# Helper — aplanar MultiIndex
# ---------------------------------------------------------------------------

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FBref devuelve columnas MultiIndex tipo ('Standard', 'Gls').
    Las convertimos a strings planos: 'standard_gls'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(level).strip().lower().replace(" ", "_")
                     for level in col if level and str(level).strip())
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# FBref
# ---------------------------------------------------------------------------

class FBrefPipeline:

    def __init__(self, seasons: list[int]):
        self.seasons = seasons
        self.fbref   = sd.FBref(leagues=[LEAGUE_MODE], seasons=seasons)

    def fetch_all(self) -> pd.DataFrame:
        dfs = {}
        for stat in VALID_STATS:
            log.info(f"FBref → {stat}...")
            try:
                df = self.fbref.read_player_season_stats(stat_type=stat)
                df = flatten_columns(df.reset_index())
                dfs[stat] = df
                log.info(f"  ✓ {len(df)} filas, {len(df.columns)} cols")
                time.sleep(4)
            except Exception as e:
                log.warning(f"  Skipping {stat}: {e}")

        if not dfs:
            return pd.DataFrame()

        base = list(dfs.values())[0]
        key_cols = [c for c in ["league", "season", "team", "player"] if c in base.columns]

        for stat, df in list(dfs.items())[1:]:
            available_keys = [c for c in key_cols if c in df.columns]
            new_cols = available_keys + [c for c in df.columns if c not in base.columns]
            try:
                base = base.merge(df[new_cols], on=available_keys, how="left")
            except Exception as e:
                log.warning(f"  Merge {stat} fallido: {e}")

        log.info(f"FBref final: {len(base)} jugadores, {len(base.columns)} columnas")
        return base

    def save(self, df: pd.DataFrame):
        if df.empty:
            log.warning("FBref vacío.")
            return
        path = RAW_DIR / "fbref_players.parquet"
        df.to_parquet(path, index=False)
        log.info(f"Guardado: {path}")
        print("\nColumnas disponibles en FBref:")
        for c in sorted(df.columns):
            print(f"  {c}")


# ---------------------------------------------------------------------------
# Transfermarkt
# ---------------------------------------------------------------------------

class TransfermarktScraper:
    """Scraper ligero de valores de mercado por equipo."""

    BASE    = "https://www.transfermarkt.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    LEAGUES = {
        "La Liga":        "/primera-division/startseite/wettbewerb/ES1",
        "Premier League": "/premier-league/startseite/wettbewerb/GB1",
        "Bundesliga":     "/bundesliga/startseite/wettbewerb/L1",
        "Serie A":        "/serie-a/startseite/wettbewerb/IT1",
        "Ligue 1":        "/ligue-1/startseite/wettbewerb/FR1",
    }

    def fetch_league(self, league: str, season: int) -> pd.DataFrame:
        url = f"{self.BASE}{self.LEAGUES[league]}/plus/?saison_id={season}"
        log.info(f"Transfermarkt → {league} {season}...")
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=15)
            r.raise_for_status()
        except Exception as e:
            log.error(f"  Error: {e}")
            return pd.DataFrame()

        soup  = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"class": "items"})
        if table is None:
            log.warning(f"  Tabla no encontrada para {league} — TM puede estar bloqueando.")
            return pd.DataFrame()

        rows = []
        for row in table.find_all("tr", {"class": ["odd", "even"]}):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            rows.append({
                "team":            cells[0].get_text(strip=True),
                "league":          league,
                "season":          season,
                "total_value_str": cells[-1].get_text(strip=True),
            })

        df = pd.DataFrame(rows)
        log.info(f"  ✓ {len(df)} equipos")
        time.sleep(3)
        return df

    def fetch_all(self, season: int) -> pd.DataFrame:
        dfs = [self.fetch_league(lg, season) for lg in self.LEAGUES]
        return pd.concat([d for d in dfs if not d.empty], ignore_index=True)

    def save(self, df: pd.DataFrame):
        if df.empty:
            log.warning("TM vacío — continuando solo con FBref.")
            return
        path = RAW_DIR / "tm_team_values.parquet"
        df.to_parquet(path, index=False)
        log.info(f"Guardado: {path}  ({len(df)} equipos)")


# ---------------------------------------------------------------------------
# Procesador
# ---------------------------------------------------------------------------

class DataProcessor:

    def load(self) -> pd.DataFrame:
        path = RAW_DIR / "fbref_players.parquet"
        if not path.exists():
            log.error("No existe fbref_players.parquet — ejecuta primero sin --process-only")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        log.info(f"Cargado: {len(df)} jugadores, {len(df.columns)} columnas")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Minutos — columna '90s' × 90 o columna 'min'
        if "90s" in df.columns:
            df["minutes_played"] = pd.to_numeric(df["90s"], errors="coerce").fillna(0) * 90
        elif "playing_time_min" in df.columns:
            df["minutes_played"] = pd.to_numeric(df["playing_time_min"], errors="coerce").fillna(0)
        else:
            min_col = next((c for c in df.columns if c.endswith("_min") or c == "min"), None)
            df["minutes_played"] = pd.to_numeric(df[min_col], errors="coerce").fillna(0) if min_col else 0

        df["availability_rate"] = (df["minutes_played"] / (90 * 38)).clip(0, 1)

        # Métricas per90
        per90_map = {
            "xg_per90":                  ["expected_xg", "xg"],
            "xag_per90":                 ["expected_xag", "xag"],
            "goals_per90":               ["standard_gls", "gls"],
            "assists_per90":             ["standard_ast", "ast"],
            "progressive_passes_per90":  ["passing_prg", "prg_p"],
            "progressive_carries_per90": ["possession_prg_c", "prg_c"],
            "tackles_per90":             ["challenges_tkl", "tkl"],
        }

        mins = df["minutes_played"].replace(0, np.nan)
        for out_col, candidates in per90_map.items():
            src = next((c for c in candidates if c in df.columns), None)
            if src:
                raw = pd.to_numeric(df[src], errors="coerce")
                df[out_col] = (raw / (mins / 90)).replace([np.inf, -np.inf], np.nan)

        return df

    def performance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = {
            "xg_per90":                  0.25,
            "xag_per90":                 0.20,
            "goals_per90":               0.15,
            "progressive_passes_per90":  0.20,
            "progressive_carries_per90": 0.10,
            "tackles_per90":             0.10,
        }
        score = pd.Series(0.0, index=df.index)
        used  = []
        for col, w in weights.items():
            if col in df.columns:
                s      = pd.to_numeric(df[col], errors="coerce")
                normed = (s - s.mean()) / (s.std() + 1e-8)
                score += w * normed.fillna(0)
                used.append(col)
        df["performance_score"] = score
        log.info(f"Performance score calculado con: {used}")
        return df

    def save(self, df: pd.DataFrame):
        path = PROC_DIR / "master_players.parquet"
        df.to_parquet(path, index=False)
        log.info(f"Dataset maestro: {path}  ({len(df)} jugadores)")

        # Top 10 por performance
        id_cols   = [c for c in ["player", "team", "league", "minutes_played",
                                  "xg_per90", "goals_per90", "performance_score"]
                     if c in df.columns]
        top10 = (df[id_cols]
                 .dropna(subset=["performance_score"])
                 .sort_values("performance_score", ascending=False)
                 .head(10))

        print("\n── Top 10 jugadores por performance score ───────────")
        print(top10.to_string(index=False))
        num_cols = [c for c in ["minutes_played", "xg_per90", "performance_score"] if c in df.columns]
        print("\n── Estadísticas descriptivas ────────────────────────")
        print(df[num_cols].describe().round(3).to_string())
        print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ValoreScout — data pipeline v2")
    p.add_argument("--seasons",       nargs="+", type=int, default=[2024])
    p.add_argument("--skip-fbref",    action="store_true")
    p.add_argument("--skip-tm",       action="store_true")
    p.add_argument("--process-only",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"ValoreScout pipeline v2 | temporadas={args.seasons}")

    if not args.process_only:
        if not args.skip_fbref:
            pipe = FBrefPipeline(args.seasons)
            df   = pipe.fetch_all()
            pipe.save(df)

        if not args.skip_tm:
            tm = TransfermarktScraper()
            df = tm.fetch_all(season=args.seasons[0])
            tm.save(df)

    proc = DataProcessor()
    df   = proc.load()
    if not df.empty:
        df = proc.clean(df)
        df = proc.performance_score(df)
        proc.save(df)

    log.info("Pipeline completado ✓")


if __name__ == "__main__":
    main()