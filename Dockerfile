FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 appuser

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app /app/app
COPY published-models /app/published-models
COPY .streamlit /app/.streamlit

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=5)" || exit 1

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
