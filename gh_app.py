import json
from enum import Enum
from typing import List, Optional
from influxdb_client import InfluxDBClient
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, field_validator

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
    weather_columns: List[str] = Field(default_factory=list)  # what we detected/expect
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

    # Greenhouse rectangle
    fig.add_shape(type="rect", x0=0, y0=0, x1=width_m, y1=length_m)

    # Zones
    for z in zones:
        fig.add_shape(
            type="rect",
            x0=z.x_m, y0=z.y_m,
            x1=z.x_m + z.w_m, y1=z.y_m + z.l_m,
        )
        label = z.name if z.placement == ZonePlacement.GROUND else f"{z.name} (bench {z.bench_height_m}m)"
        fig.add_annotation(
            x=z.x_m + z.w_m / 2,
            y=z.y_m + z.l_m / 2,
            text=label,
            showarrow=False
        )

    fig.update_layout(
        title="Greenhouse floorplan (top view)",
        xaxis_title="Width (m)",
        yaxis_title="Length (m)",
        xaxis=dict(range=[-0.5, width_m + 0.5], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-0.5, length_m + 0.5]),
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
def influx_client():
    url = st.secrets.get("INFLUX_URL", os.getenv("INFLUX_URL"))
    token = st.secrets.get("INFLUX_TOKEN", os.getenv("INFLUX_TOKEN"))
    org = st.secrets.get("INFLUX_ORG", os.getenv("INFLUX_ORG"))
    if not url or not token or not org:
        raise RuntimeError("Missing INFLUX_URL / INFLUX_TOKEN / INFLUX_ORG in secrets or env vars.")
    return InfluxDBClient(url=url, token=token, org=org)

def query_zone_forcing(bucket: str, zone: str, hours: int = 24) -> pd.DataFrame:
    """
    Returns a dataframe indexed by time with columns:
    temperature_air, humidity_air, temperature_soil, moisture_soil (if present)
    """
    flux = f'''
from(bucket: "{bucket}")
  |> range(start: -{hours}h)
  |> filter(fn: (r) => r._measurement == "greenhouse")
  |> filter(fn: (r) => r.zone == "{zone}")
  |> filter(fn: (r) => r._field == "temperature" or r._field == "humidity" or r._field == "moisture")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","temperature","humidity","moisture","sensor"])
  |> sort(columns: ["_time"])
'''
    with influx_client() as client:
        q = client.query_api().query_data_frame(flux)
        if isinstance(q, list):
            df = pd.concat(q, ignore_index=True)
        else:
            df = q

    if df.empty:
        return df

    # Some pivots can produce extra cols; normalize
    if "_time" not in df.columns:
        return pd.DataFrame()

    df["_time"] = pd.to_datetime(df["_time"])
    df = df.sort_values("_time")

    # If you have a "sensor" tag (air/soil) you can split fields properly.
    # If not, you can still use temperature/humidity/moisture directly.
    df = df.set_index("_time")

    # Rename to model-friendly names
    out = pd.DataFrame(index=df.index)
    if "temperature" in df.columns:
        out["temperature"] = df["temperature"]
    if "humidity" in df.columns:
        out["humidity"] = df["humidity"]
    if "moisture" in df.columns:
        out["moisture"] = df["moisture"]

    return out


def parse_weather(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".epw"):
        if not PVLIB_OK:
            st.warning("EPW uploaded, but pvlib isn't available. Install pvlib to parse EPW properly.")
            return "EPW", pd.DataFrame()
        data, meta = read_epw(uploaded_file)
        return "EPW", data.reset_index()
    else:
        # CSV assumed
        df = pd.read_csv(uploaded_file)
        return "CSV", df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Greenhouse Config Demo", layout="wide")
st.title("Greenhouse Config Demo")
st.caption("Build a greenhouse setup + plant zones + weather input, export as JSON. Humans love forms.")

if "zones" not in st.session_state:
    st.session_state["zones"] = []

1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Greenhouse", "Systems", "Plant zones", "Weather", "Export", "Data (InfluxDB)"])

with tab1:
    st.subheader("Greenhouse geometry")
    col1, col2, col3 = st.columns(3)
    with col1:
        gh_type = st.selectbox("Type", list(GreenhouseType), index=0)
    with col2:
        width_m = st.number_input("Width (m)", min_value=0.1, value=9.6, step=0.1)
    with col3:
        length_m = st.number_input("Length (m)", min_value=0.1, value=30.0, step=0.1)

with tab2:
    st.subheader("Energy + HVAC-ish choices")
    col1, col2, col3 = st.columns(3)
    with col1:
        energy_sources = st.multiselect(
            "Energy sources",
            options=list(EnergySource),
            default=[EnergySource.ELECTRICITY],
        )
    with col2:
        heating = st.selectbox("Heating system", list(HeatingSystemType), index=1)
    with col3:
        cooling = st.selectbox("Cooling system", list(CoolingSystemType), index=1)

with tab3:
    st.subheader("Plant zones (rectangles placed in the greenhouse)")
    st.caption("Coordinates are in meters from the bottom-left corner (0,0). Zones must fit within width/length.")

    with st.expander("Add a zone", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            z_name = st.text_input("Zone name", value=f"Zone {len(st.session_state['zones'])+1}")
        with c2:
            z_place = st.selectbox("Placement", list(ZonePlacement), index=0)
        with c3:
            z_x = st.number_input("x (m)", min_value=0.0, value=0.0, step=0.1, key="zx")
        with c4:
            z_y = st.number_input("y (m)", min_value=0.0, value=0.0, step=0.1, key="zy")

        c5, c6, c7 = st.columns(3)
        with c5:
            z_w = st.number_input("Zone width w (m)", min_value=0.1, value=3.0, step=0.1, key="zw")
        with c6:
            z_l = st.number_input("Zone length l (m)", min_value=0.1, value=6.0, step=0.1, key="zl")
        with c7:
            z_bench_h = None
            if z_place == ZonePlacement.BENCH:
                z_bench_h = st.number_input("Bench height (m)", min_value=0.0, value=0.8, step=0.05, key="zbh")

        if st.button("Add zone"):
            try:
                zone = PlantZone(
                    name=z_name,
                    placement=z_place,
                    x_m=z_x, y_m=z_y,
                    w_m=z_w, l_m=z_l,
                    bench_height_m=z_bench_h
                )
                st.session_state["zones"].append(zone.model_dump())
                st.success(f"Added: {zone.name}")
            except ValidationError as e:
                st.error(e)

    if st.session_state["zones"]:
        st.markdown("#### Current zones")
        df_z = pd.DataFrame(st.session_state["zones"])
        st.dataframe(df_z, use_container_width=True)

        if st.button("Remove last zone"):
            st.session_state["zones"].pop()

        # Draw floorplan
        zones_models = [PlantZone(**z) for z in st.session_state["zones"]]
        draw_floorplan(width_m, length_m, zones_models)
    else:
        st.info("No zones yet. Add one to see the floorplan.")

with tab4:
    st.subheader("Weather input")
    st.caption("Upload EPW (EnergyPlus) or CSV. CSV is expected to contain time + basic met variables.")

    uploaded = st.file_uploader("Upload weather file", type=["epw", "csv"])
    weather_kind = "CSV"
    weather_cols = []
    weather_preview_rows = 0

    if uploaded is not None:
        kind, dfw = parse_weather(uploaded)
        weather_kind = kind
        if not dfw.empty:
            weather_cols = list(dfw.columns)
            weather_preview_rows = min(20, len(dfw))
            st.write(f"Detected format: **{kind}**")
            st.dataframe(dfw.head(weather_preview_rows), use_container_width=True)
        else:
            st.warning("Weather file could not be parsed into a table preview.")

    st.markdown("**Tip for CSV columns** (you can map these later in your simulator):")
    st.code("datetime, temp_air_C, rh_pct, wind_mps, ghi_Wm2, dni_Wm2, dhi_Wm2, pressure_Pa (optional), co2_ppm (optional)")

with 5:
    st.subheader("Export config")
    try:
        zones_models = [PlantZone(**z) for z in st.session_state["zones"]]
        cfg = GreenhouseConfig(
            greenhouse_type=gh_type,
            width_m=width_m,
            length_m=length_m,
            energy_sources=energy_sources,
            heating_system=heating,
            cooling_system=cooling,
            zones=zones_models,
            weather_kind=weather_kind,
            weather_columns=weather_cols,
            weather_preview_rows=weather_preview_rows,
        )
        cfg_json = json.dumps(cfg.model_dump(), indent=2)
        st.code(cfg_json, language="json")
        st.download_button("Download config.json", data=cfg_json, file_name="config.json", mime="application/json")
    except ValidationError as e:
        st.error("Config invalid. Fix inputs first.")
        st.code(str(e))
with tab6:
    st.subheader("Use real sensor data (InfluxDB) as forcing")

    bucket = st.secrets.get("INFLUX_BUCKET", os.getenv("INFLUX_BUCKET", "greenhouse"))
    hours = st.slider("Lookback (hours)", 1, 168, 24)

    # Use your zone names from the config builder
    zone_names = [z["name"] for z in st.session_state.get("zones", [])] or ["Zone 1"]
    zone = st.selectbox("Zone", zone_names)

    if st.button("Load data from InfluxDB"):
        try:
            df_forcing = query_zone_forcing(bucket=bucket, zone=zone, hours=hours)
            if df_forcing.empty:
                st.warning("No data returned. Check measurement/tags in InfluxDB.")
            else:
                st.success(f"Loaded {len(df_forcing)} rows")
                st.dataframe(df_forcing.tail(50), use_container_width=True)
                st.line_chart(df_forcing[["temperature","humidity"]] if "humidity" in df_forcing.columns else df_forcing)
                st.session_state["forcing_df"] = df_forcing  # store for model use
        except Exception as e:
            st.error(str(e))

    if "forcing_df" in st.session_state:
        st.markdown("### Ready for modelling")
        st.write("Forcing dataframe is stored in `st.session_state['forcing_df']`.")

