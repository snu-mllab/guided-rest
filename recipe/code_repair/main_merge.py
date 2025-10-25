# Copyright 2025 SNU MLLAB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pprint import pprint

import hydra
from omegaconf import DictConfig, OmegaConf

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


@hydra.main(config_path="config", config_name="merge", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # create model merger
    merger_config = ModelMergerConfig(**config.merger_config)
    merger = FSDPModelMerger(merger_config)

    # merge and save model
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
