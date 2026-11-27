FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
COPY applywise ./applywise
RUN useradd --system --uid 10001 --create-home applywise
USER 10001
EXPOSE 8000
CMD ["python", "-m", "applywise.app"]
