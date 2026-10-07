# Image Sol_Agents runs project tests in.
#
# Build:   docker build -t sol-agents-sandbox:latest sandbox
# Extra packages a project needs (the container has no network at test time):
#          docker build --build-arg EXTRA_PACKAGES="pillow numpy" -t sol-agents-sandbox:latest sandbox

FROM python:3.12-slim

ARG EXTRA_PACKAGES=""
RUN pip install --no-cache-dir pytest==9.1.1 ${EXTRA_PACKAGES}

WORKDIR /project
