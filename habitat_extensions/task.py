import gzip
import json
import os
from typing import List, Optional

import attr
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import ALL_SCENES_MASK
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    r"""
    instruction_index_string: optional identifier of instruction.
    """
    instruction_index_string: Optional[str] = attr.ib(default=None)
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNExtendedEpisode) -> str:
        r"""Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return a sorted list of scenes
        """
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        scenes = {cls._scene_from_episode(episode) for episode in dataset.episodes}

        return sorted(list(scenes))

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)
