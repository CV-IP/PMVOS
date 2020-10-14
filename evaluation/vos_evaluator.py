import torch
import threading
import queue
import time
import os
import png
import numpy
import math

BASE_PALETTE_4BIT = [[0,   0,   0],
                     [236,  94, 102],
                     [249, 144,  87],
                     [250, 199,  98],
                     [153, 199, 148],
                     [97, 179, 177],
                     [102, 153, 204],
                     [196, 148, 196],
                     [171, 120, 102],
                     [255, 255, 255],
                     [101, 115, 125],
                     [10,  10,  10],
                     [12,  12,  12],
                     [13,  13,  13],
                     [13,  13,  13],
                     [14,  14,  14]]

DAVIS_PALETTE_4BIT = [[0,   0,   0],
                      [128,   0,   0],
                      [0, 128,   0],
                      [128, 128,   0],
                      [0,   0, 128],
                      [128,   0, 128],
                      [0, 128, 128],
                      [128, 128, 128],
                      [64,   0,   0],
                      [191,   0,   0],
                      [64, 128,   0],
                      [191, 128,   0],
                      [64,   0, 128],
                      [191,   0, 128],
                      [64, 128, 128],
                      [191, 128, 128]]


class ReadSaveImage(object):
    def __init__(self):
        super(ReadSaveImage, self).__init__()

    def check_path(self, fullpath):
        path, filename = os.path.split(fullpath)
        if not os.path.exists(path):
            os.makedirs(path)


class ReadSaveDAVISChallengeLabels(ReadSaveImage):
    def __init__(self, bpalette=DAVIS_PALETTE_4BIT, palette=None):
        super(ReadSaveDAVISChallengeLabels, self).__init__()
        self._palette = palette
        self._bpalette = bpalette
        self._width = 0
        self._height = 0

    @property
    def palette(self):
        return self._palette

    def save(self, image, path):
        self.check_path(path)

        if self._palette is None:
            palette = self._bpalette
        else:
            palette = self._palette

        bitdepth = int(math.log(len(palette))/math.log(2))

        height, width = image.shape
        file = open(path, 'wb')
        writer = png.Writer(width, height, palette=palette, bitdepth=bitdepth)
        writer.write(file, image)

    def read(self, path):
        try:
            reader = png.Reader(path)
            width, height, data, meta = reader.read()
            if self._palette is None:
                self._palette = meta['palette']
            image = numpy.vstack(data)
            self._height, self._width = image.shape
        except png.FormatError:
            image = numpy.zeros((self._height, self._width))
            self.save(image, path)

        return image


class ImageSaveHelper(threading.Thread):
    def __init__(self, queueSize=100000):
        super(ImageSaveHelper, self).__init__()
        self._alive = True
        self._queue = queue.Queue(queueSize)
        self.start()

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, alive):
        self._alive = alive

    @property
    def queue(self):
        return self._queue

    def kill(self):
        self._alive = False

    def enqueue(self, datatuple):
        ret = True
        try:
            self._queue.put(datatuple, block=False)
        except queue.Full:
            print("ImageSaveHelper - enqueue full")
            ret = False
        return ret

    def run(self):
        while True:
            while not self._queue.empty():
                args, method = self._queue.get(block=False, timeout=2)
                method.save(*args)

                self._queue.task_done()

            if not self._alive and self._queue.empty():
                break

            time.sleep(0.001)


class VOSEvaluator(object):
    def __init__(self, dataset, device='cuda', save=False):
        self._dataset = dataset
        self._device = device
        self._save = save
        self._imsavehlp = ImageSaveHelper()
        if dataset.__class__.__name__ == 'DAVIS17V2':
            self._sdm = ReadSaveDAVISChallengeLabels()

    def read_video_part(self, video_part):
        images = video_part['images'].to(self._device)
        given_segannos = [seganno.to(self._device) if seganno is not None else None
                          for seganno in video_part['given_segannos']]
        segannos = video_part['segannos'].to(self._device) if video_part.get('segannos') is not None else None
        fnames = video_part['fnames']
        return images, given_segannos, segannos, fnames

    def evaluate_video(self, model, seqname, video_parts, output_path, save, video_seq):
        for video_part in video_parts:
            images, given_segannos, segannos, fnames = self.read_video_part(video_part)

            if video_seq == 0:
                _, _ = model(images, given_segannos, None)

            t0 = time.time()
            tracker_out, _ = model(images, given_segannos, None)
            t1 = time.time()

            if save is True:
                for idx in range(len(fnames)):
                    fpath = os.path.join(output_path, seqname, fnames[idx])
                    data = ((tracker_out['segs'][0, idx, 0, :, :].cpu().byte().numpy(), fpath), self._sdm)
                    self._imsavehlp.enqueue(data)
        return t1-t0, len(fnames)

    def evaluate(self, model, output_path):
        model.to(self._device)
        model.eval()
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with torch.no_grad():
            tot_time, tot_frames, video_seq = 0.0, 0.0, 0
            for seqname, video_parts in self._dataset.get_video_generator():
                savepath = os.path.join(output_path, seqname)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                time_elapsed, frames = self.evaluate_video(model, seqname, video_parts, output_path, self._save,
                                                           video_seq)
                tot_time += time_elapsed
                tot_frames += frames
                video_seq += 1

                if self._save is False:
                    print(seqname, 'fps:{}, frames:{}, time:{}'.format(frames / time_elapsed, frames, time_elapsed))
                else:
                    print(seqname, 'saved')

            if self._save is False:
                print('\nTotal fps:{}\n\n'.format(tot_frames/tot_time))
            else:
                print('\nTotal seq saved\n\n')

        self._imsavehlp.kill()

