# Image Sol_Agents runs Python project code in. Sol builds it automatically the
# first time it is needed; to build it yourself:
#
#          docker build -t sol-agents-sandbox:latest sandbox
# Extra packages a project needs (the container has no network at test time):
#          docker build --build-arg EXTRA_PACKAGES="pillow numpy" -t sol-agents-sandbox:latest sandbox

FROM python:3.12-slim

# xvfb gives GUI code (tkinter, Qt) a virtual screen: components with
# "display": true run their checks under xvfb-run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

ARG EXTRA_PACKAGES=""
RUN pip install --no-cache-dir pytest==9.1.1 ${EXTRA_PACKAGES}

WORKDIR /project
