import os
from io import BytesIO
from typing import Literal

import boto3
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel

import logging

load_dotenv()

MODEL_OBJECT_KEY = "models/halfmarathon_gradient_boosting_v1.joblib"
LANGFUSE_ENABLED = all(
    os.getenv(name)
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    )
)

def mask_openai_content(*, params: MaskOtelSpansParams):
    """Remove user text and model output from OpenAI spans sent to Langfuse."""
    patches = {}
    for span_id, span in params.spans.items():
        if span.instrumentation_scope_name == "openai":
            content_attributes = tuple(
                name
                for name in span.attributes
                if (
                    name.startswith("gen_ai.prompt.")
                    or name.startswith("gen_ai.completion.")
                )
                and name.endswith(".content")
            )
            patches[span_id] = OtelSpanPatch(
                delete_attributes=content_attributes,
                set_attributes={"content_masked": True},
            )
    if not patches:
        return None
    return MaskOtelSpansResult(span_patches=patches)


langfuse_client = (
    Langfuse(mask_otel_spans=mask_openai_content)
    if LANGFUSE_ENABLED
    else None
)


class RunnerProfile(BaseModel):
    gender: Literal["K", "M"] | None
    age: int | None
    time_5k_seconds: int | None


@st.cache_resource(show_spinner=False)
def fetch_model():
    """Pobiera i buforuje wytrenowany model z DigitalOcean Spaces."""
    required = [
        "DO_SPACES_BUCKET",
        "DO_SPACES_REGION",
        "DO_SPACES_ENDPOINT",
        "DO_SPACES_KEY",
        "DO_SPACES_SECRET",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Brakuje konfiguracji DigitalOcean Spaces.")

    client = boto3.client(
        "s3",
        region_name=os.environ["DO_SPACES_REGION"],
        endpoint_url=os.environ["DO_SPACES_ENDPOINT"],
        aws_access_key_id=os.environ["DO_SPACES_KEY"],
        aws_secret_access_key=os.environ["DO_SPACES_SECRET"],
    )
    response = client.get_object(
        Bucket=os.environ["DO_SPACES_BUCKET"],
        Key=MODEL_OBJECT_KEY,
    )
    return joblib.load(BytesIO(response["Body"].read()))


def _request_runner_profile(description: str) -> RunnerProfile:
    client_class = LangfuseOpenAI if LANGFUSE_ENABLED else OpenAIClient
    client = client_class(api_key=os.environ["OPENAI_API_KEY"])
    result = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Odczytaj z tekstu dane biegacza i zwróć je w wymaganym formacie. "
                    "Płeć koduj jako K (kobieta) lub M (mężczyzna), wiek jako liczbę "
                    "ukończonych lat, a wynik na 5 km jako całkowitą liczbę sekund. "
                    "Jeśli którejś informacji nie ma, zwróć null."
                ),
            },
            {"role": "user", "content": description},
        ],
        response_format=RunnerProfile,
        **(
            {
                "name": "runner-profile-extraction",
                "metadata": {"langfuse_tags": ["structured-extraction"]},
            }
            if LANGFUSE_ENABLED
            else {}
        ),
    )
    profile = result.choices[0].message.parsed
    if profile is None:
        raise ValueError("Nie udało się odczytać danych z opisu.")
    return profile


def parse_runner_profile(description: str) -> dict:
    """Zwraca dane jako słownik i, jeśli skonfigurowano Langfuse, zapisuje metryki."""
    if langfuse_client is None:
        return _request_runner_profile(description).model_dump()

    with langfuse_client.start_as_current_observation(
        as_type="span",
        name="runner-profile-extraction",
        input={"description_length": len(description)},
    ) as span:
        profile = _request_runner_profile(description).model_dump()

        # ile pól w profilu jest niepustych
        present = sum(value is not None for value in profile.values())

        # ile pól w profilu przechodzi podstawowe kontrole zakresu
        valid = sum(
            (
                profile["gender"] in ("K", "M"),
                profile["age"] is not None and 16 <= profile["age"] <= 100,
                profile["time_5k_seconds"] is not None
                and profile["time_5k_seconds"] > 0,
            )
        )

        # wskźanik kompletności i poprawnoci
        completeness = present / 3
        basic_correctness = valid / 3

        # aktualizacja metryk w Langfuse
        span.update(
            output={
                "fields_present": present,
                "fields_passing_basic_checks": valid,
            }
        )
        span.score_trace(
            name="extraction_completeness",
            value=completeness,
            data_type="NUMERIC",
            comment="Odsetek niepustych pól.",
        )
        span.score_trace(
            name="extraction_basic_correctness",
            value=basic_correctness,
            data_type="NUMERIC",
            comment="Odsetek pól przechodzących kontrole zakresu.",
        )
        return profile


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


st.set_page_config(
    page_title="Półmaraton | prognoza",
    page_icon="🏃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #111827; color: #f3f4f6; }
    .block-container { max-width: 800px; padding-top: 2rem; }
    .hero, .hint-card, div[data-testid="stForm"] {
        background: #1f2937; color: #f3f4f6; border: 1px solid #374151;
        border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    }
    .hero h1, .hero p, .hint-card { color: #f3f4f6; }
    .hero-kicker, .result-label { color: #5eead4; }
    textarea { background: #111827 !important; color: #f9fafb !important; }
    div.stButton > button[kind="primaryFormSubmit"] {
        background: #0f766e; border: 0; border-radius: 8px;
    }
    .result-box { background: #1f2937; border-left: 4px solid #2dd4bf;
        border-radius: 10px; padding: 1rem; margin-top: 1rem; }
    .result-time { color: #f9fafb; font-size: 2rem; font-weight: 700; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Bieg · analiza · planowanie</div>
      <h1>Jaki może być Twój czas?</h1>
      <p>Podaj kilka informacji o swoim biegu na 5 km i poznaj szacowany wynik półmaratonu.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hint-card">
      <strong>Opisz się jednym zdaniem.</strong><br>
      Możesz napisać na przykład: „Jestem facetem, mam 42 lata, 5 km biegnę w 28 minut”.
      System odczyta z wypowiedzi potrzebne dane.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("runner_form"):
    description = st.text_area(
        "Informacje o biegaczu",
        placeholder="Wpisz płeć, wiek i typowy czas na 5 km…",
        height=125,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Wyznacz prognozę", type="primary", use_container_width=True)

if submitted:
    if not description.strip():
        st.warning("Dodaj krótki opis, aby można było przygotować prognozę.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("Nie skonfigurowano klucza API OpenAI.")
    else:
        try:
            with st.spinner("Odczytuję informacje z opisu…"):
                runner = parse_runner_profile(description)

            missing = []
            if runner["gender"] is None:
                missing.append("płeć")
            if runner["age"] is None:
                missing.append("wiek")
            if runner["time_5k_seconds"] is None:
                missing.append("czas na 5 km")

            if missing:
                st.warning("Nie udało się znaleźć: " + ", ".join(missing) + ". Uzupełnij opis i spróbuj ponownie.")
            elif not 16 <= runner["age"] <= 100:
                st.warning("Podaj wiek w zakresie od 16 do 100 lat.")
            elif runner["time_5k_seconds"] <= 0:
                st.warning("Czas na 5 km musi być większy od zera.")
            else:
                left, middle, right = st.columns(3)
                left.metric("Płeć", "Kobieta" if runner["gender"] == "K" else "Mężczyzna")
                middle.metric("Wiek", f"{runner['age']} lat")
                right.metric("5 km", format_duration(runner["time_5k_seconds"]))

                features = pd.DataFrame(
                    [{
                        "gender": runner["gender"],
                        "age": runner["age"],
                        "time_5k_seconds": runner["time_5k_seconds"],
                    }]
                )
                with st.spinner("Model oblicza przewidywany wynik…"):
                    prediction = fetch_model().predict(features)[0]

                st.markdown(
                    f"""
                    <div class="result-box">
                      <div class="result-label">Szacowany czas półmaratonu</div>
                      <div class="result-time">{format_duration(prediction)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("To przybliżenie oparte na wynikach historycznych. Średni błąd modelu testowego wynosi około 5 minut.")
        except Exception:
            logging.exception("Nie udało się przygotować prognozy")
            st.error("Nie udało się przygotować prognozy. Sprawdź logi aplikacji.")

with st.expander("Jak powstaje prognoza?"):
    st.write(
        "Opis jest przekształcany w trzy cechy: płeć, wiek oraz czas na 5 km. "
        "Model regresyjny wytrenowany na historycznych wynikach zawodów wykorzystuje je "
        "do oszacowania czasu ukończenia półmaratonu. Wynik ma charakter orientacyjny."
    )
