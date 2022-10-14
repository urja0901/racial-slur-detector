import argparse
import logging

from servicefoundry import Build, Job, PythonBuild, Resources

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()

# servicefoundry uses this specification to automatically create a Dockerfile and build an image,
python_build = PythonBuild(
    python_version="3.9",
    command="python train.py",
)
env = {
    # These will automatically map the secret value to the environment variable.
    "MLF_HOST": "https://app.develop.truefoundry.tech",
    "MLF_API_KEY": "tfy-secret://urja-jobanputra:test01:key1",
}
job = Job(
    name="twitter-sentiment",
    image=Build(build_spec=python_build),
    env=env,
    resources=Resources(
        cpu_request=1, cpu_limit=1.5, memory_request=1000, memory_limit=1500
    ),
)
job.deploy(workspace_fqn=args.workspace_fqn)