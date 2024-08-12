"""
Configuration loader can load its config data from either a plain text yaml file
or from an encrypted .zip file.

If it finds a zip file in the ./conf folder it will use it by default

zip file is created with normal compression protocol
use sasrtsng password to encrypt and decrypt

NOTE a bootstrap load is required for basic environment definition
TO DO introduce a boot configuration rtsng_loader.yaml for the test system with

- environment
- config file and version to be used
- uncrypted config switch

"""

import sys
import zipfile
from pathlib import Path

import yaml

from utils import logger

ROOT_DIR = Path(".")
CONF_DIR = ROOT_DIR / "conf"

path_to_file = CONF_DIR / "dev_config.zip"
path = Path(path_to_file)

encrypted = False
if path.is_file():
    encrypted = True


class _Config:
    def __init__(self):

        self.logger = logger.get_logger(__name__)
        self.logger.info("Loading config")

        try:

            if encrypted:

                with zipfile.ZipFile(CONF_DIR / "dev_config.zip", mode="r") as arc:
                    arc.setpassword(b"sasrtsng")
                    yaml_config_file = arc.read("dev_config.yaml")

                    self.config = yaml.safe_load(yaml_config_file)

            else:

                with open(CONF_DIR / "dev_config.yaml", "r") as yaml_config_file:
                    self.config = yaml.safe_load(yaml_config_file)

        except FileNotFoundError:

            print("Missing config file...")
            sys.exit()

    def __getattr__(self, name):

        try:
            return self.config[name]

        except KeyError:
            return getattr(name)


config = _Config().config
