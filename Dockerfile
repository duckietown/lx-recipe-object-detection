# syntax=docker/dockerfile:1.4
# parameters
ARG EXERCISE_NAME="lx-object-detection"
ARG DESCRIPTION="Build a deep learning based model to detect duckies"
ARG MAINTAINER="Liam P"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH
ARG DISTRO=ente
ARG DOCKER_REGISTRY=docker.io
ARG BASE_IMAGE=dt-machine-learning-base-environment
# JetPack major version of the NVIDIA Jetson the image is built on (set by `dts devel build` on a Duckiebot, empty
# elsewhere): selects the base image's Jetson variant `<distro>-jetpack<N>-<arch>`, which uses the robot's GPU
ARG JETPACK=
ARG BASE_TAG=${DISTRO}${JETPACK:+-jetpack${JETPACK}}-${ARCH}
ARG LAUNCHER=default

# define base image
FROM ${DOCKER_REGISTRY}/duckietown/${BASE_IMAGE}:${BASE_TAG} as base

# recall all arguments
ARG DISTRO
ARG EXERCISE_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG BASE_TAG
ARG BASE_IMAGE
ARG JETPACK
ARG LAUNCHER
# - buildkit
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# check build arguments
RUN dt-build-env-check "${EXERCISE_NAME}" "${MAINTAINER}" "${DESCRIPTION}"

# define/create repository path
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${EXERCISE_NAME}"
ARG LAUNCH_PATH="${LAUNCH_DIR}/${EXERCISE_NAME}"
RUN mkdir -p "${REPO_PATH}" "${LAUNCH_PATH}"
WORKDIR "${REPO_PATH}"

# keep some arguments as environment variables
ENV DT_MODULE_TYPE="exercise" \
    DT_MODULE_NAME="${EXERCISE_NAME}" \
    DT_MODULE_DESCRIPTION="${DESCRIPTION}" \
    DT_MAINTAINER="${MAINTAINER}" \
    DT_REPO_PATH="${REPO_PATH}" \
    DT_LAUNCH_PATH="${LAUNCH_PATH}" \
    DT_LAUNCHER="${LAUNCHER}"

# install apt dependencies
COPY --from=recipe ./dependencies-apt.txt ./dependencies-apt.noble.txt "${REPO_PATH}/"
RUN dt-apt-install ${REPO_PATH}/dependencies-apt.txt && \
    if [ "$(. /etc/os-release && echo "${VERSION_CODENAME}")" = "noble" ]; then \
        dt-apt-install ${REPO_PATH}/dependencies-apt.noble.txt; \
    fi

# install python3 dependencies
ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
COPY --from=recipe ./dependencies-py3.* "${REPO_PATH}/"
RUN dt-pip3-install "${REPO_PATH}/dependencies-py3.*"

# install ONNX Runtime, unless the base image provides it (dt-machine-learning-base-environment provides the GPU builds:
# TensorRT/CUDA on the Jetson, CUDA on an x86_64 laptop)
COPY --from=recipe ./dependencies-onnxruntime.txt "${REPO_PATH}/"
RUN python3 -c "import onnxruntime" 2>/dev/null || dt-pip3-install "${REPO_PATH}/dependencies-onnxruntime.txt"

# copy the assets (recipe)
COPY --from=recipe ./assets "${REPO_PATH}/assets"

# copy the source code (recipe)
COPY --from=recipe ./packages "${REPO_PATH}/packages"

# copy the assets (meat)
COPY --from=meat ./assets/. "${REPO_PATH}/assets/"

# copy the source code (meat)
COPY --from=meat ./packages/. "${REPO_PATH}/packages/"

# install launcher scripts
COPY --from=recipe ./launchers/. "${LAUNCH_PATH}/"
RUN dt-install-launchers "${LAUNCH_PATH}"

# define default command
CMD ["bash", "-c", "dt-launcher-${DT_LAUNCHER}"]

# store module metadata
LABEL org.duckietown.label.module.type="exercise" \
    org.duckietown.label.module.name="${EXERCISE_NAME}" \
    org.duckietown.label.module.description="${DESCRIPTION}" \
    org.duckietown.label.platform.os="${TARGETOS}" \
    org.duckietown.label.platform.architecture="${TARGETARCH}" \
    org.duckietown.label.platform.variant="${TARGETVARIANT}" \
    org.duckietown.label.code.location="${REPO_PATH}" \
    org.duckietown.label.code.version.distro="${DISTRO}" \
    org.duckietown.label.base.image="${BASE_IMAGE}" \
    org.duckietown.label.base.tag="${BASE_TAG}" \
    org.duckietown.label.maintainer="${MAINTAINER}"
# <== Do not change the code above this line
# <==================================================

ENV YOLOv5_AUTOINSTALL=false
# make `packages/` directory discoverable
ENV PYTHONPATH=${REPO_PATH}/packages
