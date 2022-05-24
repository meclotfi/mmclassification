# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.custom import CustomDataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


class VisualWakeWords(COCO):
    def __init__(self, *args):
        super(VisualWakeWords, self).__init__(*args)

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for ann in anns:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            if ann['category_id'] == 1:
                [x, y, width, height] = ann['bbox']
                rect = patches.Rectangle((x, y), width, height, edgecolor=c, facecolor=c, linewidth=2, alpha=0.4)
                ax.add_patch(rect)

    def download(self, *args):
        raise AttributeError("Cannot download Visual Wake Words Dataset. "
                             "See instructions on github.com/Mxbonn/visualwakewords to create"
                             "the Visual Wake Words Dataset.")

    def loadRes(self, resFile):
        raise AttributeError("Method not implemented for the Visual Wake Words Dataset.")

    def annToRLE(self, ann):
        raise AttributeError("Method not implemented for the Visual Wake Words Dataset.")

    def annToMask(self, ann):
        raise AttributeError("Method not implemented for the Visual Wake Words Dataset.")

@DATASETS.register_module()
class Vww(CustomDataset):
    

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = ["no person","person"]

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = ["no person","person"],
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 subset_size=None,
                 file_client_args: Optional[dict] = None):
        self.subset_size=subset_size
        self.vww = VisualWakeWords(ann_file)
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode,
            file_client_args=file_client_args)
        
        
    
    
    
    
    def getimg_info(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the index of the target class.
        """
        vww = self.vww
        img_id = self.ids[index]
        ann_ids = vww.getAnnIds(imgIds=img_id)
        if ann_ids:
            target = vww.loadAnns(ann_ids)[0]['category_id']
        else:
            target = 0

        path = vww.loadImgs(img_id)[0]['file_name']
        return path,target

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if self.ann_file is None:
            samples = self._find_samples()
        elif isinstance(self.ann_file, str):
            self.ids= list(sorted(self.vww.imgs.keys()))
        else:
            raise TypeError('ann_file must be a str or None')

        data_infos = []
        size=len(self.ids) if self.subset_size is None else self.subset_size
        for id in range(size):
            filename,gt_label=self.getimg_info(id)
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array([gt_label], dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
