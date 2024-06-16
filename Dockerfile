
ARG PYTHON_VERSION=3.11

FROM seven45/pdm-ci:$PYTHON_VERSION as pdm
WORKDIR /project
COPY pyproject.toml pdm.lock /project/
RUN python -m venv --copies .venv
RUN pdm install --prod --no-self --no-lock --no-editable

FROM python:$PYTHON_VERSION
WORKDIR /project
COPY --from=pdm /project/.venv /project/.venv
ENV PATH="/project/.venv/bin:$PATH"
COPY src /project/src
CMD ["python", "src/__main__.py"]