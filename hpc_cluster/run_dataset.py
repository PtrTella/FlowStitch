import sys
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.dataset_generation import generate_dataset

PROMPTS = [
    "a red cube",
    "a blue sphere",
    "a yellow pyramid",
    "a red cube and a blue sphere",
    "a blue cube and a red sphere",
    "a red cube on the left, a blue sphere on the right",
    "a blue sphere on the left, a red cube on the right",
    "a majestic mountain",
    "a crystal clear lake",
    "a majestic mountain reflecting in a crystal clear lake"
]

if __name__ == "__main__":
    config = FlowStitchConfig()
    generate_dataset(config, PROMPTS)
