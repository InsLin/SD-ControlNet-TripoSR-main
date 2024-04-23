from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    ZoeDetector,
)

PREPROCCES_DICT = {
    # "Hed": HEDdetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "Midas": MidasDetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "MLSD": MLSDdetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "Openpose": OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "PidiNet": PidiNetDetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "NormalBae": NormalBaeDetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    "Lineart": LineartDetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "LineartAnime": LineartAnimeDetector.from_pretrained(
    #     "lllyasviel/Annotators", cache_dir="./model/"
    # ),
    # "Zoe": ZoeDetector.from_pretrained("lllyasviel/Annotators", cache_dir="./model/"),
    # "Canny": CannyDetector(),
    # "ContentShuffle": ContentShuffleDetector(),
    # "MediapipeFace": MediapipeFaceDetector(),
}
