# syntax=docker/dockerfile:1.6
FROM python:3.12.0

ENV NUMBA_CACHE_DIR=/tmp

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# for kaleido
RUN mkdir -p /tmp /var/tmp && chmod 1777 /tmp /var/tmp
ENV TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp \
    XDG_CONFIG_HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates pkg-config gfortran \
    libgl1 libglib2.0-0 \
    libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libxext6 libxrender1 libsm6 \
    gdal-bin libgdal-dev libgeos-dev libproj-dev proj-data proj-bin \
    libhdf5-103-1 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY spoqc /app/spoqc
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml
# If you keep setup.py
COPY setup.py /app/setup.py

# Install into system site-packages (works from any Nextflow workdir)
RUN python -m pip install /app

# Now create and switch to non-root
RUN useradd -m -u 1000 appuser
USER appuser

ENV QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg

CMD ["python", "-c", "import sys; import spoqc; print('Python', sys.version); print('spoqc OK')"]