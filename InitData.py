import os

from Interface import generateKey, preparePRC

BASE_DIR =  r"D:\W\Data"

if __name__ == "__main__":

    print(generateKey(BASE_DIR))

    preparePRC(os.path.join(BASE_DIR, "models"), "sd-research/stable-diffusion-2-1-base", allow_download=True)