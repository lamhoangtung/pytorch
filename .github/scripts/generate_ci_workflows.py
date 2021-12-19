#!/usr/bin/env python3

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Set, List

import jinja2
import json
import os
import sys
from typing_extensions import Literal, TypedDict

Arch = Literal["windows", "linux", "macos"]


class Config(TypedDict):
    num_shards: int
    runner: str


DOCKER_REGISTRY = "308535385114.dkr.ecr.us-east-1.amazonaws.com"
GITHUB_DIR = Path(__file__).resolve().parent.parent

WINDOWS_CPU_TEST_RUNNER = "windows.4xlarge"
# contains 1 gpu
WINDOWS_CUDA_TEST_RUNNER = "windows.8xlarge.nvidia.gpu"
WINDOWS_RUNNERS = {
    WINDOWS_CPU_TEST_RUNNER,
    WINDOWS_CUDA_TEST_RUNNER,
}

LINUX_CPU_TEST_RUNNER = "linux.2xlarge"
# contains 1 gpu
LINUX_CUDA_TEST_RUNNER = "linux.4xlarge.nvidia.gpu"
LINUX_RUNNERS = {
    LINUX_CPU_TEST_RUNNER,
    LINUX_CUDA_TEST_RUNNER,
}

MACOS_TEST_RUNNER_10_15 = "macos-10.15"
MACOS_TEST_RUNNER_11 = "macos-11"

MACOS_RUNNERS = {
    MACOS_TEST_RUNNER_10_15,
    MACOS_TEST_RUNNER_11,
}

CUDA_RUNNERS = {
    WINDOWS_CUDA_TEST_RUNNER,
    LINUX_CUDA_TEST_RUNNER,
}
CPU_RUNNERS = {
    WINDOWS_CPU_TEST_RUNNER,
    LINUX_CPU_TEST_RUNNER,
}

LABEL_CIFLOW_ALL = "ciflow/all"
LABEL_CIFLOW_BAZEL = "ciflow/bazel"
LABEL_CIFLOW_CPU = "ciflow/cpu"
LABEL_CIFLOW_CUDA = "ciflow/cuda"
LABEL_CIFLOW_DOCS = "ciflow/docs"
LABEL_CIFLOW_DEFAULT = "ciflow/default"
LABEL_CIFLOW_LIBTORCH = "ciflow/libtorch"
LABEL_CIFLOW_LINUX = "ciflow/linux"
LABEL_CIFLOW_MOBILE = "ciflow/mobile"
LABEL_CIFLOW_ANDROID = "ciflow/android"
LABEL_CIFLOW_SANITIZERS = "ciflow/sanitizers"
LABEL_CIFLOW_ONNX = "ciflow/onnx"
LABEL_CIFLOW_SCHEDULED = "ciflow/scheduled"
LABEL_CIFLOW_SLOW = "ciflow/slow"
LABEL_CIFLOW_WIN = "ciflow/win"
LABEL_CIFLOW_XLA = "ciflow/xla"
LABEL_CIFLOW_NOARCH = "ciflow/noarch"
LABEL_CIFLOW_VULKAN = "ciflow/vulkan"
LABEL_CIFLOW_PREFIX = "ciflow/"
LABEL_CIFLOW_SLOW_GRADCHECK = "ciflow/slow-gradcheck"
LABEL_CIFLOW_DOCKER = "ciflow/docker"
LABEL_CIFLOW_IOS = "ciflow/ios"
LABEL_CIFLOW_MACOS = "ciflow/macos"
LABEL_CIFLOW_TRUNK = "ciflow/trunk"


@dataclass
class CIFlowConfig:
    # For use to enable workflows to run on pytorch/pytorch-canary
    run_on_canary: bool = False
    labels: Set[str] = field(default_factory=set)
    trigger_action: str = "unassigned"
    trigger_actor: str = "pytorchbot"
    root_job_name: str = "ciflow_should_run"
    root_job_condition: str = ""
    label_conditions: str = ""

    def gen_root_job_condition(self) -> None:
        # CIFlow conditions:
        #  - Workflow should always run on push
        #  - CIFLOW_DEFAULT workflows should run on PRs even if no `ciflow/` labels on PR
        #  - Otherwise workflow should be scheduled on all qualifying events
        label_conditions = [
            f"contains(github.event.pull_request.labels.*.name, '{label}')"
            for label in sorted(self.labels)
        ]
        self.label_conditions = " || ".join(label_conditions)
        repo_condition = (
            "github.repository_owner == 'pytorch'"
            if self.run_on_canary
            else "github.repository == 'pytorch/pytorch'"
        )
        push_event = "github.event_name == 'push'"
        scheduled_event = "github.event_name == 'schedule'"
        pr_updated_event = f"github.event_name == 'pull_request' && github.event.action != '{self.trigger_action}'"
        if LABEL_CIFLOW_DEFAULT in self.labels:
            run_with_no_labels = (
                f"({pr_updated_event}) && "
                f"!contains(join(github.event.pull_request.labels.*.name), '{LABEL_CIFLOW_PREFIX}')"
            )
        else:
            run_with_no_labels = "false"
        self.root_job_condition = (
            f"${{{{ ({repo_condition}) && (\n"
            f"            ({push_event}) ||\n"
            f"            ({scheduled_event}) ||\n"
            f"            ({self.label_conditions}) ||\n"
            f"            ({run_with_no_labels}))\n"
            f"         }}}}"
        )

    def reset_root_job(self) -> None:
        self.root_job_name = ""
        self.root_job_condition = ""

    def __post_init__(self) -> None:
        self.labels.add(LABEL_CIFLOW_ALL)
        if LABEL_CIFLOW_SCHEDULED not in self.labels:
            self.labels.add(LABEL_CIFLOW_TRUNK)
        assert all(label.startswith(LABEL_CIFLOW_PREFIX) for label in self.labels)
        self.gen_root_job_condition()


@dataclass
class CIFlowRuleset:
    version = "v1"
    output_file = f"{GITHUB_DIR}/generated-ciflow-ruleset.json"
    label_rules: Dict[str, Set[str]] = field(default_factory=dict)

    def add_label_rule(self, labels: Set[str], workflow_name: str) -> None:
        for label in labels:
            if label in self.label_rules:
                self.label_rules[label].add(workflow_name)
            else:
                self.label_rules[label] = {workflow_name}

    def generate_json(self) -> None:
        GENERATED = "generated"  # Note that please keep the variable GENERATED otherwise phabricator will hide the whole file
        output = {
            "__comment": f"@{GENERATED} DO NOT EDIT MANUALLY, Generation script: .github/scripts/generate_ci_workflows.py",
            "version": self.version,
            "label_rules": {
                label: sorted(list(workflows))
                for label, workflows in self.label_rules.items()
            },
        }
        with open(self.output_file, "w") as outfile:
            json.dump(output, outfile, indent=2, sort_keys=True)
            outfile.write("\n")


@dataclass
class CIWorkflow:
    # Required fields
    arch: Arch
    build_environment: str
    template: str

    # Optional fields
    test_runner_type: str = ""
    ciflow_config: CIFlowConfig = field(default_factory=CIFlowConfig)
    cuda_version: str = ""
    docker_image_base: str = ""
    enable_doc_jobs: bool = False
    exclude_test: bool = False
    build_generates_artifacts: bool = True
    build_with_debug: bool = False
    is_scheduled: str = ""
    num_test_shards: int = 1
    only_run_smoke_tests_on_pull_request: bool = False
    num_test_shards_on_pull_request: int = -1
    timeout_after: int = 240
    xcode_version: str = ""

    # The following variables will be set as environment variables,
    # so it's easier for both shell and Python scripts to consume it if false is represented as the empty string.
    enable_jit_legacy_test: bool = False
    enable_distributed_test: bool = True
    enable_multigpu_test: bool = False
    enable_nogpu_no_avx_test: bool = False
    enable_nogpu_no_avx2_test: bool = False
    enable_slow_test: bool = False
    enable_docs_test: bool = False
    enable_backwards_compat_test: bool = False
    enable_xla_test: bool = False
    enable_noarch_test: bool = False
    enable_force_on_cpu_test: bool = False

    def __post_init__(self) -> None:
        if not self.build_generates_artifacts:
            self.exclude_test = True

        # If num_test_shards_on_pull_request is not user-defined, default to num_test_shards unless we are
        # only running smoke tests on the pull request.
        if self.num_test_shards_on_pull_request == -1:
            # Don't run the default if we are only running smoke tests
            if self.only_run_smoke_tests_on_pull_request:
                self.num_test_shards_on_pull_request = 0
            else:
                self.num_test_shards_on_pull_request = self.num_test_shards
        self.assert_valid()

    def assert_valid(self) -> None:
        err_message = (
            f"invalid test_runner_type for {self.arch}: {self.test_runner_type}"
        )
        if self.arch == "linux":
            assert self.test_runner_type in LINUX_RUNNERS, err_message
        if self.arch == "windows":
            assert self.test_runner_type in WINDOWS_RUNNERS, err_message

        assert LABEL_CIFLOW_ALL in self.ciflow_config.labels
        assert LABEL_CIFLOW_ALL in self.ciflow_config.label_conditions
        if self.arch == "linux":
            assert LABEL_CIFLOW_LINUX in self.ciflow_config.labels
        if self.arch == "windows":
            assert LABEL_CIFLOW_WIN in self.ciflow_config.labels
        if self.arch == "macos":
            assert LABEL_CIFLOW_MACOS in self.ciflow_config.labels
        # Make sure that jobs with tests have a test_runner_type
        if not self.exclude_test:
            assert self.test_runner_type != ""
        if self.test_runner_type in CUDA_RUNNERS:
            assert LABEL_CIFLOW_CUDA in self.ciflow_config.labels
        if self.test_runner_type in CPU_RUNNERS and not self.exclude_test:
            assert LABEL_CIFLOW_CPU in self.ciflow_config.labels
        if self.is_scheduled:
            assert LABEL_CIFLOW_DEFAULT not in self.ciflow_config.labels
            assert LABEL_CIFLOW_TRUNK not in self.ciflow_config.labels
            assert LABEL_CIFLOW_SCHEDULED in self.ciflow_config.labels
        if self.build_with_debug:
            assert self.build_environment.endswith("-debug")

    # def generate_workflow_file(self, workflow_template: jinja2.Template) -> None:
    #     output_file_path = GITHUB_DIR / f"workflows/generated-{self.build_environment}.yml"
    #     with open(output_file_path, "w") as output_file:
    #         GENERATED = "generated"  # Note that please keep the variable GENERATED otherwise phabricator will hide the whole file
    #         output_file.writelines([f"# @{GENERATED} DO NOT EDIT MANUALLY\n"])
    #         try:
    #             content = workflow_template.render(asdict(self))
    #         except Exception as e:
    #             print(f"Failed on template: {workflow_template}", file=sys.stderr)
    #             raise e
    #         output_file.write(content)
    #         if content[-1] != "\n":
    #             output_file.write("\n")
    #     print(output_file_path)

    def normalized_build_environment(self, suffix: str) -> str:
        return self.build_environment.replace(".", "_") + suffix

    def build_name(self):
        return self.normalized_build_environment("-build")

    def test_name(self):
        return self.normalized_build_environment("-test")

    def docs_name(self):
        return self.normalized_build_environment("-docs")

    def test_jobs(self):
        if self.arch == "linux":
            MULTIGPU_RUNNER_TYPE = "linux.16xlarge.nvidia.gpu"
            DISTRIBUTED_GPU_RUNNER_TYPE = "linux.8xlarge.nvidia.gpu"
            NOGPU_RUNNER_TYPE = "linux.2xlarge"
        elif self.arch == "windows":
            DISTRIBUTED_GPU_RUNNER_TYPE = self.test_runner_type
            NOGPU_RUNNER_TYPE = "windows.4xlarge"

        test_jobs = []

        # todo: no run_as_if_on_trunk implemented
        num_test_shards = self.num_test_shards_on_pull_request

        configs: Dict[str, Config] = {}
        if self.enable_jit_legacy_test:
            configs["jit_legacy"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_multigpu_test:
            configs["multigpu"] = {"num_shards": 1, "runner": MULTIGPU_RUNNER_TYPE}

        if self.enable_nogpu_no_avx_test:
            configs["nogpu_NO_AVX"] = {"num_shards": 1, "runner": NOGPU_RUNNER_TYPE}
        if self.enable_nogpu_no_avx2_test:
            configs["nogpu_NO_AVX2"] = {"num_shards": 1, "runner": NOGPU_RUNNER_TYPE}
        if self.enable_force_on_cpu_test:
            configs["force_on_cpu"] = {"num_shards": 1, "runner": NOGPU_RUNNER_TYPE}
        if self.enable_distributed_test:
            configs["distributed"] = {
                "num_shards": 1,
                "runner": DISTRIBUTED_GPU_RUNNER_TYPE
                if "cuda" in str(self.build_environment)
                else self.test_runner_type,
            }
        if self.enable_slow_test:
            configs["slow"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_docs_test:
            configs["docs_test"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_backwards_compat_test:
            configs["backwards_compat"] = {
                "num_shards": 1,
                "runner": self.test_runner_type,
            }
        if self.enable_xla_test:
            configs["xla"] = {"num_shards": 1, "runner": self.test_runner_type}
        if self.enable_noarch_test:
            configs["noarch"] = {"num_shards": 1, "runner": self.test_runner_type}

        run_smoke_tests = self.only_run_smoke_tests_on_pull_request
        if run_smoke_tests:
            configs["smoke_tests"] = {"num_shards": 1, "runner": self.test_runner_type}

        for name, config in configs.items():
            for shard in range(1, config["num_shards"] + 1):
                test_jobs.append(
                    {
                        "name": self.test_name()
                        + f"_{name}_{shard}_{config['num_shards']}",
                        "config": name,
                        "shard": shard,
                        "num_shards": config["num_shards"],
                        "runner": config["runner"],
                    }
                )

        for shard in range(1, num_test_shards + 1):
            test_jobs.append(
                {
                    "name": self.test_name() + f"_default_{shard}_{num_test_shards}",
                    "config": "default",
                    "shard": shard,
                    "num_shards": num_test_shards,
                    "runner": self.test_runner_type,
                }
            )
        return test_jobs


@dataclass
class DockerWorkflow:
    build_environment: str
    docker_images: List[str]

    # Optional fields
    ciflow_config: CIFlowConfig = field(default_factory=CIFlowConfig)
    cuda_version: str = ""
    is_scheduled: str = ""

    def generate_workflow_file(self, workflow_template: jinja2.Template) -> None:
        output_file_path = GITHUB_DIR / "workflows/generated-docker-builds.yml"
        with open(output_file_path, "w") as output_file:
            GENERATED = "generated"  # Note that please keep the variable GENERATED otherwise phabricator will hide the whole file
            output_file.writelines([f"# @{GENERATED} DO NOT EDIT MANUALLY\n"])
            try:
                content = workflow_template.render(asdict(self))
            except Exception as e:
                print(f"Failed on template: {workflow_template}", file=sys.stderr)
                raise e
            output_file.write(content)
            if content[-1] != "\n":
                output_file.write("\n")
        print(output_file_path)


PULL_JOBS = [
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cpu-py3",
        template="job/windows_job.yml.j2",
        cuda_version="cpu",
        test_runner_type=WINDOWS_CPU_TEST_RUNNER,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_CPU, LABEL_CIFLOW_WIN},
        ),
    ),
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cuda11.3-py3",
        template="job/windows_job.yml.j2",
        cuda_version="11.3",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        only_run_smoke_tests_on_pull_request=True,
        enable_force_on_cpu_test=True,
        ciflow_config=CIFlowConfig(
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_CUDA, LABEL_CIFLOW_WIN},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-gcc5.4",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        enable_jit_legacy_test=True,
        enable_backwards_compat_test=True,
        enable_docs_test=True,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-docs",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        enable_doc_jobs=True,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_DEFAULT,
                LABEL_CIFLOW_DOCS,
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_CPU,
            }
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc7",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3-clang5-mobile-build",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_MOBILE, LABEL_CIFLOW_DEFAULT},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3-clang5-mobile-custom-build-static",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_MOBILE, LABEL_CIFLOW_DEFAULT},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-clang7-asan",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-asan",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=3,
        enable_distributed_test=False,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_DEFAULT,
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_SANITIZERS,
                LABEL_CIFLOW_CPU,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-clang7-onnx",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-onnx",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=2,
        enable_distributed_test=False,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_DEFAULT,
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_ONNX,
                LABEL_CIFLOW_CPU,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-cuda11.3-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            labels=set([LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-py3.6-clang9",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=2,
        enable_distributed_test=False,
        enable_noarch_test=True,
        enable_xla_test=True,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_DEFAULT,
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_CPU,
                LABEL_CIFLOW_XLA,
                LABEL_CIFLOW_NOARCH,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-vulkan-bionic-py3.6-clang9",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        num_test_shards=1,
        enable_distributed_test=False,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_DEFAULT,
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_CPU,
                LABEL_CIFLOW_VULKAN,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single",
        template="job/android_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_CPU,
                LABEL_CIFLOW_ANDROID,
                LABEL_CIFLOW_DEFAULT,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single-full-jit",
        template="job/android_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_CPU,
                LABEL_CIFLOW_ANDROID,
                LABEL_CIFLOW_DEFAULT,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-cuda11.3-py3.6-gcc7-bazel-test",
        template="job/bazel_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_DEFAULT,
                LABEL_CIFLOW_BAZEL,
                LABEL_CIFLOW_CPU,
                LABEL_CIFLOW_LINUX,
            },
        ),
    ),
]

TRUNK_JOBS = [
    CIWorkflow(
        arch="linux",
        build_environment="parallelnative-linux-xenial-py3.6-gcc5.4",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    # Build PyTorch with BUILD_CAFFE2=ON
    CIWorkflow(
        arch="linux",
        build_environment="caffe2-linux-xenial-py3.6-gcc5.4",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-cuda10.2-py3.9-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.9-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        enable_jit_legacy_test=True,
        enable_multigpu_test=True,
        enable_nogpu_no_avx_test=True,
        enable_nogpu_no_avx2_test=True,
        enable_slow_test=True,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            run_on_canary=True,
            labels={LABEL_CIFLOW_SLOW, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="libtorch-linux-xenial-cuda10.2-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels=set([LABEL_CIFLOW_LIBTORCH, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-coreml",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-full-jit",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-custom-ops",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-arm64-metal",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-x86-64",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-x86-64-coreml",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="ios-12-5-1-x86-64-full-jit",
        template="job/ios_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_IOS, LABEL_CIFLOW_MACOS},
        ),
    ),
    # Distributed tests are still run on MacOS, but part of regular shards
    CIWorkflow(
        arch="macos",
        build_environment="macos-11-py3-x86-64",
        template="job/macos_job.yml.j2",
        xcode_version="12.4",
        test_runner_type=MACOS_TEST_RUNNER_11,
        num_test_shards=2,
        enable_distributed_test=False,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="macos-10-15-py3-lite-interpreter-x86-64",
        template="job/macos_job.yml.j2",
        xcode_version="12",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        build_generates_artifacts=False,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="macos",
        build_environment="macos-10-15-py3-arm64",
        template="job/macos_job.yml.j2",
        test_runner_type=MACOS_TEST_RUNNER_10_15,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_MACOS},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="libtorch-linux-xenial-cuda11.3-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels=set([LABEL_CIFLOW_LIBTORCH, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
]

PERIODIC_JOBS = [
    CIWorkflow(
        arch="windows",
        build_environment="periodic-win-vs2019-cuda11.5-py3",
        template="job/windows_job.yml.j2",
        cuda_version="11.5",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        enable_force_on_cpu_test=True,
        is_scheduled="45 4,10,16,22 * * *",
        ciflow_config=CIFlowConfig(
            run_on_canary=True,
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_CUDA, LABEL_CIFLOW_WIN},
        ),
    ),
    CIWorkflow(
        arch="windows",
        build_environment="periodic-win-vs2019-cuda11.1-py3",
        template="job/windows_job.yml.j2",
        cuda_version="11.1",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 0,4,8,12,16,20 * * *",
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_WIN, LABEL_CIFLOW_CUDA}
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-bionic-cuda11.5-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda11.5-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 4,10,16,22 * * *",
        ciflow_config=CIFlowConfig(
            labels=set([LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-libtorch-linux-bionic-cuda11.5-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda11.5-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        is_scheduled="45 4,10,16,22 * * *",
        exclude_test=True,
        ciflow_config=CIFlowConfig(
            labels=set(
                [
                    LABEL_CIFLOW_SCHEDULED,
                    LABEL_CIFLOW_LIBTORCH,
                    LABEL_CIFLOW_LINUX,
                    LABEL_CIFLOW_CUDA,
                ]
            ),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-xenial-cuda11.1-py3.6-gcc7-debug",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        build_with_debug=True,
        is_scheduled="45 0,4,8,12,16,20 * * *",
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA}
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-libtorch-linux-xenial-cuda11.1-py3.6-gcc7",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        build_generates_artifacts=False,
        exclude_test=True,
        is_scheduled="45 0,4,8,12,16,20 * * *",
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_SCHEDULED,
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_LIBTORCH,
                LABEL_CIFLOW_CUDA,
            },
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        enable_distributed_test=False,
        timeout_after=360,
        # Only run this on master 4 times per day since it does take a while
        is_scheduled="0 */4 * * *",
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_LINUX,
                LABEL_CIFLOW_CUDA,
                LABEL_CIFLOW_SLOW_GRADCHECK,
                LABEL_CIFLOW_SLOW,
                LABEL_CIFLOW_SCHEDULED,
            },
        ),
    ),
]

NIGHTLY_JOBS = [
    CIWorkflow(
        arch="linux",
        build_environment="linux-docs-push",
        template="job/linux_job.yml.j2",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        enable_doc_jobs=True,
        exclude_test=True,
        is_scheduled="0 0 * * *",  # run pushes only on a nightly schedule
        # NOTE: This is purposefully left without LABEL_CIFLOW_DOCS so that you can run
        #       docs builds on your PR without the fear of anything pushing
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU}
        ),
    ),
]

DOCKER_IMAGES = {
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.6-clang9",  # for pytorch/xla
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm4.1-py3.6",  # for rocm
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm4.2-py3.6",  # for rocm
    f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm4.3.1-py3.6",  # for rocm
}

DOCKER_IMAGES.update(
    {
        workflow.docker_image_base
        for workflow in [*PULL_JOBS, *PERIODIC_JOBS, *TRUNK_JOBS]
        if workflow.docker_image_base
    }
)

DOCKER_JOBS = [
    DockerWorkflow(
        build_environment="docker-builds",
        docker_images=sorted(DOCKER_IMAGES),
        # Run weekly to ensure they can build
        is_scheduled="1 * */7 * *",
    ),
]


def main() -> None:
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
        undefined=jinja2.StrictUndefined,
    )

    def generate_workflow(jobs, template, f):
        f = GITHUB_DIR / "workflows" / f
        template = jinja_env.get_template(template)
        content = template.render(jobs=jobs)

        with open(f, "w") as f:
            f.write(content)

    generate_workflow(PULL_JOBS, "pull.yml.j2", "pull.yml")
    generate_workflow(PERIODIC_JOBS, "periodic.yml.j2", "periodic.yml")
    generate_workflow(TRUNK_JOBS, "trunk.yml.j2", "trunk.yml")
    # generate_workflow(DOCKER_JOBS, "ci_docker_build.yml.j2", "ci_docker_build.yml")


    # template_and_workflows = [
    #     (jinja_env.get_template("linux_ci_workflow.yml.j2"), LINUX_WORKFLOWS),
    # (jinja_env.get_template("windows_ci_workflow.yml.j2"), WINDOWS_WORKFLOWS),
    # (jinja_env.get_template("bazel_ci_workflow.yml.j2"), BAZEL_WORKFLOWS),
    # (jinja_env.get_template("ios_ci_workflow.yml.j2"), IOS_WORKFLOWS),
    # (jinja_env.get_template("macos_ci_workflow.yml.j2"), MACOS_WORKFLOWS),
    # (jinja_env.get_template("docker_builds_ci_workflow.yml.j2"), DOCKER_WORKFLOWS),
    # (jinja_env.get_template("android_ci_workflow.yml.j2"), ANDROID_WORKFLOWS),
    # ]
    # Delete the existing generated files first, this should align with .gitattributes file description.
    existing_workflows = GITHUB_DIR.glob("workflows/generated-*")
    for w in existing_workflows:
        try:
            os.remove(w)
        except Exception as e:
            print(f"Error occurred when deleting file {w}: {e}")



if __name__ == "__main__":
    main()
