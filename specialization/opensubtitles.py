import json
import os
import random
import datasets

# _CITATION = """
# """

# _DESCRIPTION = """
# """

class OpensubtitlesConfig(datasets.BuilderConfig):
    """BuilderConfig for Opensubtitles."""

    def __init__(
        self,
        language,
        data_dir,
        **kwargs,
    ):
        """BuilderConfig for Opensubtitles.
        Args:
          language: `string`, which language in use
          data_dir: `string`, directory to load the file from
          **kwargs: keyword arguments forwarded to super.
        """
        super(OpensubtitlesConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.language = language
        self.data_dir = data_dir

class Opensubtitles(datasets.GeneratorBasedBuilder):
    """Opensubtitles Dataset."""
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        OpensubtitlesConfig(
            name="cn",
            description="chinese",
            language="cn",
            data_dir="../LangOpenSubtitles/rs-x/rs_dialogue_300K.en-zh.json",
        ),
        OpensubtitlesConfig(
            name="de",
            description="german",
            language="de",
            data_dir="../LangOpenSubtitles/rs-x/rs_dialogue_300K.en-de.json",
        ),
        OpensubtitlesConfig(
            name="ar",
            description="arabic",
            language="ar",
            data_dir="../LangOpenSubtitles/rs-x/rs_dialogue_300K.en-ar.json",
        ),
        OpensubtitlesConfig(
            name="ru",
            description="russian",
            language="ru",
            data_dir="../LangOpenSubtitles/rs-x/rs_dialogue_300K.en-ru.json",
        ),
        OpensubtitlesConfig(
            name="cn-cn",
            description="chinese-chinese",
            language="cn-cn",
            data_dir="../LangOpenSubtitles/rs-mono/rs_mono_dialogue_300K.zh-zh.json",
        ),
        OpensubtitlesConfig(
            name="de-de",
            description="german-german",
            language="de-de",
            data_dir="../LangOpenSubtitles/rs-mono/rs_mono_dialogue_300K.de-de.json",
        ),
        OpensubtitlesConfig(
            name="ar-ar",
            description="arabic-arabic",
            language="ar-ar",
            data_dir="../LangOpenSubtitles/rs-mono/rs_mono_dialogue_300K.ar-ar.json",
        ),
        OpensubtitlesConfig(
            name="ru-ru",
            description="russian-russian",
            language="ru-ru",
            data_dir="../LangOpenSubtitles/rs-mono/rs_mono_dialogue_300K.ru-ru.json",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "response": datasets.Value("string"),
                    "label": datasets.Value("int8"),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )
    
    def _split_generators(self, dl_manager):
        data_file = dl_manager.download_and_extract(self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_file,
                },),]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath,  "r") as f:
            json_reader = json.load(f)
            
        for id_, dial in enumerate(json_reader):
            context = dial['context']
            response = dial['response']
            label = dial['label']
            yield id_, {"context": context, "response": response, "label": label}