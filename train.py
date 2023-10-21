import hydra
from omegaconf import DictConfig, OmegaConf
from data.nyu.nyu_defocus import nyudefocus
import cv2
import numpy as np

def get_showable_image(img):
    if not type(img)==np.ndarray:
        img=img.cpu().detach().numpy()
    if len(img.shape)<3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    else:
        if img.shape[2]>3:
            img=np.swapaxes(img,0,-1)
    return img

def show_image(img):
    img=get_showable_image(img)
    cv2.imshow('Image', img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

def show_data(d):
    image=get_showable_image(d['image'])
    depth=get_showable_image(d['depth'])
    blur=get_showable_image(d['blur'])

    stack=depth
    if depth.shape==blur.shape:
        stack=np.concatenate((stack,blur),axis=1)
    if depth.shape==image.shape:
        stack=np.concatenate((stack,image),axis=1)

    cv2.imshow('hori', stack) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    dl=nyudefocus(conf,'train')
    d=next(iter(dl))
    # show_image(d['image'])
    show_data(d)
    print('here')

if __name__ == "__main__":
    train()









