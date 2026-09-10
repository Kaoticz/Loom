from enum import Enum


class SandboxStage(str, Enum):
    """
    Which step of preparing the sandbox failed.

    Every cause used to surface as one indistinguishable sentence, which is what
    made an unusable sandbox hard to diagnose. Naming the step lets the interface
    and the reader tell a missing container runtime from a Dockerfile that will
    not build.
    """

    runtime_connectivity = 'runtime_connectivity'
    path_discovery = 'path_discovery'
    image_build = 'image_build'
    image_pull = 'image_pull'
    container_start = 'container_start'
    readiness_probe = 'readiness_probe'
