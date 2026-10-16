# Image Sol_Agents runs C and C++ projects in: GCC, make, CMake and Ninja.
# Sol builds it automatically the first time a project needs it; to build it yourself:
#
#          docker build -t sol-agents-cpp:latest -f sandbox/cpp.Dockerfile sandbox

FROM gcc:14

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project
