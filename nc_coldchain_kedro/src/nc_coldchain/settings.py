"""Project settings for the NC Cold-Chain Kedro project."""
from nc_coldchain.hooks import CredentialsToEnvHook, LineageHook, PipelineTimerHook

# Registered hooks. CredentialsToEnvHook exposes the weather key to the engine;
# LineageHook records dataset lineage + graph version on save; PipelineTimerHook
# emits the readable [STAGE] ... logs the operator sees in Docker.
HOOKS = (CredentialsToEnvHook(), PipelineTimerHook(), LineageHook())

# Configuration source folder and environments.
CONF_SOURCE = "conf"

# Config loader picks up parameters/catalog from conf/base then conf/<env>.
from kedro.config import OmegaConfigLoader  # noqa: E402

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "catalog": ["catalog*", "catalog*/**", "**/catalog*"],
        "credentials": ["credentials*", "credentials*/**", "**/credentials*"],
    },
}
