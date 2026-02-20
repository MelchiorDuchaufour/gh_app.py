import json
import os
from enum import Enum
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, field_validator

from influxdb_client import InfluxDBClient

# Optional EPW support (nice to have)
try:
    from pvlib.iotools import read_epw
    PVLIB_OK = True
except Exception:
    PVLIB_OK = False


# -----------------------------
# Models (what your simulator will consume)
# -----------------------------
class GreenhouseType(str, Enum):
    VENLO = "Venlo"
    TUNNEL = "Tunnel"
    GLASSHOUSE = "Glasshouse"
    POLYTUNNEL = "Polytunnel"
    OTHER = "Other"


class EnergySource(str, Enum):
    ELECTRICITY = "Electricity"
    NATURAL_GAS = "Natural gas"
    BIOMASS = "Biomass"
    DISTRICT_HEAT = "District heating"
    SOLAR_THERMAL = "Solar thermal"
    HEAT_PUMP = "Heat pump"


class HeatingSystemType(str, Enum):
    NONE = "None"
    HOT_WATER_PIPES = "Hot water pipes"
    HOT_AIR = "Hot air unit heaters"
    RADIANT = "Radiant / IR"
    BOILER_LOOP = "Boiler + hydronic loop"


class CoolingSystemType(str, Enum):
    NONE = "None"
    VENTILATION = "Natural/forced ventilation"
    PAD_FAN = "Pad & fan"
    FOGGING = "Fogging / misting"
    CHILLER = "Chiller / DX cooling"


class ZonePlacement(str, Enum):
    GROUND = "Ground"
    BENCH = "Bench"


class PlantZone(BaseModel):
    name: str = Field(..., min_length=1)
    placement: ZonePlacement
    x_m: float = Field(..., ge=0)
    y_m: float = Field(..., ge=0)
    w_m: float = Field(..., gt=0)
    l_m: float = Field(..., gt=0)
    bench_height_m: Optional[float] = Field(None, ge=0)

    @field_validator("bench_height_m")
    @classmethod
    def bench_height_required_if_bench(cls, v, info):
        placement = info.data.get("placement")
        if placement == ZonePlacement.BENCH and v is None:
            raise ValueError("bench_height_m required when placement is Bench")
        return v


class GreenhouseConfig(BaseModel):
    greenhouse_type: GreenhouseType
    width_m: float = Field(..., gt=0)
    length_m: float = Field(..., gt=0)
    energy_sources: List[EnergySource]
    heating_system: HeatingSystemType
    cooling_system: CoolingSystemType
    zones: List[PlantZone] = Field(default_factory=list)
    weather_kind: str  # "EPW" or "CSV"
    weather_columns: List[str] = Field(default_factory=list)
    weather_preview_rows: int = 0

    @field_validator("zones")
    @classmethod
    def zones_within_bounds(cls, zones, info):
        w = info.data.get("width_m")
        l = info.data.get("length_m")
        if w is None or l is None:
            return zones
        for z in zones:
            if z.x_m + z.w_m > w + 1e-9 or z.y_m + z.l_m > l + 1e-9:
                raise ValueError(f"Zone '{z.name}' exceeds greenhouse bounds.")
        return zones


# -----------------------------
# Helpers
# -----------------------------
def draw_floorplan(width_m: float, length_m: float, zones: List[PlantZone]):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=width_m, y1=length_m)

    for z in zones:
        fig.add_shape(type="rect", x0=z.x_m, y0=z.y_m, x1=z.x_m + z.w_m, y1=z.y_m + z.l_m)
        label = z.name if z.placement == ZonePlacement.GROUND else f"{z.name} (bench {z.bench_height_m}m)"
        fig.add_annotation(x=z.x_m + z.w_m / 2, y=z.y_m + z.l_m / 2, text=label, showarrow=False)

    fig.update_layout(
        title="Greenhouse floorplan (top view)",
        xaxis_title="Width (m)",
        yaxis_title="Length (m)",
        xaxis=dict(range=[-0.5, width_m + 0.5], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-0.5, length_m + 0.5]),
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def parse_weather(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".epw"):
        if not PVLIB_OK:
            st.warning("EPW uploaded, but pvlib isn't available. Install pvlib to parse EPW properly.")
            return "EPW", pd.DataFrame()
        data, meta = read_epw(uploaded_file)
        return "EPW", data.reset_index()
    df = pd.read_csv(uploaded_file)
    return "CSV", df


# -----------------------------
# InfluxDB helpers
# -----------------------------
def get_influx_settings():
    # session_state -> secrets -> env -> defaults
    return {
        "url": st.session_state.get("INFLUX_URL")
        or st.secrets.get("INFLUX_URL", os.getenv("INFLUX_URL", "http://127.0.0.1:8086")),
        "org": st.session_state.get("INFLUX_ORG")
        or st.secrets.get("INFLUX_ORG", os.getenv("INFLUX_ORG", "phd")),
        "bucket": st.session_state.get("INFLUX_BUCKET")
        or st.secrets.get("INFLUX_BUCKET", os.getenv("INFLUX_BUCKET", "greenhouse")),
        "token": st.session_state.get("INFLUX_TOKEN")
        or st.secrets.get("INFLUX_TOKEN", os.getenv("INFLUX_TOKEN", "")),
    }


def influx_client():
    cfg = get_influx_settings()
    if not cfg["token"]:
        raise RuntimeError("Influx token missing. Add it in the Connection panel (or Streamlit secrets).")
    return InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"])


def query_zone_forcing(bucket: str, zone:
