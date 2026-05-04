import os

from Interface import generateKey, preparePRC, applyTreeRing

BASE_DIR =  r"D:\W\Data"
MODEL_ID = "sd-research/stable-diffusion-2-1-base"

if __name__ == "__main__":

    print(generateKey(BASE_DIR))

    preparePRC(os.path.join(BASE_DIR, "models"), MODEL_ID, allow_download=True)

    print(applyTreeRing(BASE_DIR, MODEL_ID, ["a photo of a cat"])[1])