import json
import os
from typing import Any, Dict

from generate_ci_workflows import LINUX_WORKFLOWS
from typing_extensions import TypedDict


class Config(TypedDict):
    num_shards: int
    runner: str

def should_run_as_if_on_trunk(build_environment) -> bool:
    ON_PULL_REQUEST = os.getenv('GITHUB_HEAD_REF')
    if not ON_PULL_REQUEST:
        return True

    from pathlib import Path
    GITHUB_DIR = Path(__file__).resolve().parent.parent

    with open(f'{GITHUB_DIR}/generated-ciflow-ruleset.json') as f:
        labels_to_workflows = json.load(f)['label_rules']

    pr_labels = json.loads(os.getenv('PR_LABELS', '[]'))
    current_workflow_triggered_by_label = False
    for label in pr_labels:
        if label != 'ciflow/default' and label in labels_to_workflows:
            workflows_triggered_by_label = labels_to_workflows[label]
            if any([build_environment in workflow for workflow in workflows_triggered_by_label]):
                current_workflow_triggered_by_label = True
                break

    return current_workflow_triggered_by_label

MULTIGPU_RUNNER_TYPE = "linux.16xlarge.nvidia.gpu"
DISTRIBUTED_GPU_RUNNER_TYPE = "linux.8xlarge.nvidia.gpu"
NOGPU_RUNNER_TYPE = "linux.2xlarge"

matrix: Any = {"include": []}
for workflow in LINUX_WORKFLOWS:
    test_runner_type = workflow.test_runner_type
    assert test_runner_type is not None

    run_as_if_on_trunk = should_run_as_if_on_trunk(workflow.build_environment)


    num_test_shards_on_pull_request = workflow.num_test_shards_on_pull_request
    num_test_shards = workflow.num_test_shards
    if not run_as_if_on_trunk:
        num_test_shards = num_test_shards_on_pull_request

    configs: Dict[str, Config] = {}
    if workflow.enable_jit_legacy_test:
        configs['jit_legacy'] = {'num_shards': 1, 'runner': test_runner_type}
    if workflow.enable_multigpu_test:
        configs['multigpu'] = {'num_shards': 1, 'runner': MULTIGPU_RUNNER_TYPE}

    if workflow.enable_nogpu_no_avx_test:
        configs['nogpu_NO_AVX'] = {'num_shards': 1, 'runner': NOGPU_RUNNER_TYPE}
    if workflow.enable_nogpu_no_avx2_test:
        configs['nogpu_NO_AVX2'] = {'num_shards': 1, 'runner': NOGPU_RUNNER_TYPE}
    if workflow.enable_force_on_cpu_test:
        configs['force_on_cpu'] = {'num_shards': 1, 'runner': NOGPU_RUNNER_TYPE}
    if workflow.enable_distributed_test:
        configs['distributed'] = {
            'num_shards': 1,
            'runner': DISTRIBUTED_GPU_RUNNER_TYPE if "cuda" in str(workflow.build_environment) else test_runner_type
        }
    if workflow.enable_slow_test:
        configs['slow'] = {'num_shards': 1, 'runner': test_runner_type}
    if workflow.enable_docs_test:
        configs['docs_test'] = {'num_shards': 1, 'runner': test_runner_type}
    if workflow.enable_backwards_compat_test:
        configs['backwards_compat'] = {'num_shards': 1, 'runner': test_runner_type}
    if workflow.enable_xla_test:
        configs['xla'] = {'num_shards': 1, 'runner': test_runner_type}
    if workflow.enable_noarch_test:
        configs['noarch'] = {'num_shards': 1, 'runner': test_runner_type}

    run_smoke_tests = workflow.only_run_smoke_tests_on_pull_request and not run_as_if_on_trunk(workflow.build_environment)
    if run_smoke_tests:
        configs['smoke_tests'] = {'num_shards': 1, 'runner': test_runner_type}

    for name, config in configs.items():
        for shard in range(1, config["num_shards"] + 1):
            matrix["include"].append(
                {
                    "needs": f"{workflow.build_environment}-build",
                    'config': name,
                    'shard': shard,
                    'num_shards': config['num_shards'],
                    'runner': config['runner'],
                    "build_environment": workflow.build_environment,
                    "docker_image_base": workflow.docker_image_base,
                    "build_with_debug": workflow.build_with_debug,
                    "timeout_after": workflow.timeout_after,
                }
            )

    for shard in range(1, num_test_shards + 1):
        matrix["include"].append(
            {
                "needs": f"{workflow.build_environment}-build",
                'config': 'default',
                'shard': shard,
                'num_shards': num_test_shards,
                'runner': test_runner_type,
                "build_environment": workflow.build_environment,
                "docker_image_base": workflow.docker_image_base,
                "build_with_debug": workflow.build_with_debug,
                "timeout_after": workflow.timeout_after,
            }
        )

print(json.dumps(matrix, indent=2))
print(f'::set-output name=matrix::{json.dumps(matrix)}')
