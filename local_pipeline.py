
"""
Author: Rival Haikal Hafizh
Date: 26/04/2023
This is the local_pipeline.py module.
Usage:
- Create TFX Pipeline
"""

import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "rivalhaikalhafizh-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/bmi_transform.py"
TRAINER_MODULE_FILE = "modules/bmi_trainer.py"
TUNER_MODULE_FILE = "modules/bmi_tuner.py"
# requirement_file = os.path.join(root, "requirements.txt")

# pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_roots = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_roots, "metadata.sqlite")


def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """Initiate tfx pipeline
    Args:
        pipeline_root (Text): a path to th pipeline directory
        pipeline_name (str): pipeline name
        metadata_path (str): a path to the metadata directory
        components (dict): tfx components
    Returns:
        pipeline.Pipeline: pipeline orchestration
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components
    components_args = {
        "data_dir": DATA_ROOT,
        "training_module": TRAINER_MODULE_FILE,
        "tuner_module": TUNER_MODULE_FILE,
        "transform_module": TRANSFORM_MODULE_FILE,
        "training_steps": 300,
        "eval_steps": 200,
        "serving_model_dir": serving_model_dir,
    }

    componentss = init_components(components_args)

    pipeline = init_local_pipeline(componentss, pipeline_roots)
    BeamDagRunner().run(pipeline=pipeline)
