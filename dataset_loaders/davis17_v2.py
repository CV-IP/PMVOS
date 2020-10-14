import glob
import os
import json
from collections import OrderedDict
from PIL import Image
import torch
from dataset_loaders import dataset_utils


def get_sample_all():
    return lambda lst: lst


def get_anno_ids(anno_path, pic_to_tensor_function, threshold):
    pic = Image.open(anno_path)
    tensor = pic_to_tensor_function(pic)
    values = (tensor.view(-1).bincount() > threshold).nonzero().view(-1).tolist()
    if 0 in values:
        values.remove(0)
    if 255 in values:
        values.remove(255)
    return values


class DAVIS17V2(torch.utils.data.Dataset):
    def __init__(self, root_path, version, image_set, image_read=None, anno_read=None,
                 samplelen=4, obj_selection=get_sample_all(), min_num_obj=1, start_frame='random'):
        self._min_num_objects = min_num_obj
        self._root_path = root_path
        self._version = version
        self._image_set = image_set
        self._image_read = image_read
        self._anno_read = anno_read
        self._seqlen = samplelen
        self._obj_selection = obj_selection
        self._start_frame = start_frame
        self._init_data()
        
    def _init_data(self):
        framework_path = os.path.join(os.path.dirname(__file__), '..')
        cache_path = os.path.join(framework_path, 'cache', 'davis17_v2_visible_objects_100px_threshold.json')

        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._visible_objects = json.load(f)
                self._visible_objects = {seqname: OrderedDict((int(idx), objlst) for idx, objlst in val.items())
                                         for seqname, val in self._visible_objects.items()}
        else:
            seqnames = os.listdir(os.path.join(self._root_path, 'JPEGImages', '480p'))
            self._visible_objects = {}
            for seqname in seqnames:
                anno_paths = sorted(glob.glob(self._full_anno_path(seqname, '*.png')))
                self._visible_objects[seqname] = OrderedDict(
                    (self._frame_name_to_idx(os.path.basename(path)),
                     get_anno_ids(path, dataset_utils.LabelToLongTensor(), 100))
                    for path in anno_paths)

            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            with open(cache_path, 'w') as f:
                json.dump(self._visible_objects, f)
            print("Datafile {} was not found, creating it with {} sequences.".format(cache_path, len(self._visible_objects)))

        with open(os.path.join(self._root_path, 'ImageSets', self._version, self._image_set + '.txt'), 'r') as f:
            self._all_seqs = f.read().splitlines()

        self._nonempty_frame_ids = {seq: [frame_idx for frame_idx, obj_ids in lst.items() if len(obj_ids) >=
                                          self._min_num_objects]
                                    for seq, lst in self._visible_objects.items()}

        self._viable_seqs = [seq for seq in self._all_seqs if
                             len(self._nonempty_frame_ids[seq]) > 0
                             and len(self.get_image_frame_ids(seq)[min(self._nonempty_frame_ids[seq]):
                                                                   max(self._visible_objects[seq].keys()) + 1])
                             >= self._seqlen]

    def __len__(self):
        return len(self._viable_seqs)

    def _frame_idx_to_image_fname(self, idx):
        return "{:05d}.jpg".format(idx)

    def _frame_idx_to_anno_fname(self, idx):
        return "{:05d}.png".format(idx)

    def _frame_name_to_idx(self, fname):
        return int(os.path.splitext(fname)[0])

    def get_all_seqnames(self):
        return self._all_seqs

    def get_anno_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, "Annotations", "480p", seqname))

    def get_anno_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_anno_frame_names(seqname)])

    def get_image_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, "JPEGImages", "480p", seqname))

    def get_image_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def get_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def _full_image_path(self, seqname, image):
        if isinstance(image, int):
            image = self._frame_idx_to_image_fname(image)
        return os.path.join(self._root_path, 'JPEGImages', "480p", seqname, image)

    def _full_anno_path(self, seqname, anno):
        if isinstance(anno, int):
            anno = self._frame_idx_to_anno_fname(anno)
        return os.path.join(self._root_path, 'Annotations', "480p", seqname, anno)

    def _get_snippet(self, seqname, frame_ids):
        images = torch.stack([self._image_read(self._full_image_path(seqname, idx))
                              for idx in frame_ids]).unsqueeze(0)
        segannos = torch.stack([self._anno_read(self._full_anno_path(seqname, idx))
                              for idx in frame_ids]).squeeze().unsqueeze(0)
        if self._version == '2016':
            segannos = (segannos != 0).long()
        given_segannos = [self._anno_read(self._full_anno_path(seqname, idx)).unsqueeze(0)
                        if idx == self.get_anno_frame_ids(seqname)[0] else None for idx in frame_ids]
        for i in range(len(given_segannos)):
            if given_segannos[i] is not None:
                given_segannos[i][given_segannos[i] == 255] = 0
                if self._version == '2016':
                    given_segannos[i] = (given_segannos[i] != 0).long()

        fnames = [self._frame_idx_to_anno_fname(idx) for idx in frame_ids]
        return {'images': images, 'given_segannos': given_segannos, 'segannos': segannos, 'fnames': fnames}
        
    def _get_video(self, seqname):
        seq_frame_ids = self.get_frame_ids(seqname)
        partitioned_frame_ids = [seq_frame_ids[start_idx: start_idx + self._seqlen]
                                 for start_idx in range(0, len(seq_frame_ids), self._seqlen)]
        for frame_ids in partitioned_frame_ids:
            yield self._get_snippet(seqname, frame_ids)

    def get_video_generator(self):
        for seqname in self.get_all_seqnames():
            yield (seqname, self._get_video(seqname))

