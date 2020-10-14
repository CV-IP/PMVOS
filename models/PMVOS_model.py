from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import models


def softmax_aggregate(predicted_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in predicted_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logits = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
              for n, seg in [(-1, bg_seg)] + list(predicted_seg.items())}
    logits_sum = torch.cat(list(logits.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logits[n] / logits_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    final_seg_wrongids = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    assert final_seg_wrongids.dtype == torch.int64
    final_seg = torch.zeros_like(final_seg_wrongids)
    for idx, obj_idx in enumerate(object_ids):
        final_seg[final_seg_wrongids == (idx + 1)] = obj_idx
    return final_seg, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in
                       enumerate(object_ids)}


def get_required_padding(height, width, div):
    height_pad = (div - height % div) % div
    width_pad = (div - width % div) % div
    padding = [(width_pad + 1) // 2, width_pad // 2, (height_pad + 1) // 2, height_pad // 2]
    return padding


def apply_padding(x, y, padding):
    B, L, C, H, W = x.size()
    x = x.view(B * L, C, H, W)
    x = F.pad(x, padding, mode='reflect')
    _, _, height, width = x.size()
    x = x.view(B, L, C, height, width)
    y = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in y]
    return x, y


def unpad(tensor, padding):
    if isinstance(tensor, (dict, OrderedDict)):
        return {key: unpad(val, padding) for key, val in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return [unpad(elem, padding) for elem in tensor]
    else:
        _, _, _, height, width = tensor.size()
        tensor = tensor[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]
        return tensor


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PixelLevelMatching(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_in = Conv(in_channel, out_channel, 1, 1, 0)

    def generate_state(self, feats, given_seg):
        conv_feats_norm = self.get_conv_feats_norm(feats)
        pm_feats = conv_feats_norm.reshape(conv_feats_norm.size(0), conv_feats_norm.size(1), conv_feats_norm.size(2) * conv_feats_norm.size(3))
        pm_feats = pm_feats.permute([0, 2, 1])
        given_seg_bg = given_seg[:, 0, :, :]
        given_seg_bg = given_seg_bg.reshape(given_seg_bg.size(0), given_seg_bg.size(1) * given_seg_bg.size(2), 1)
        given_seg_fg = given_seg[:, 1, :, :]
        given_seg_fg = given_seg_fg.reshape(given_seg_fg.size(0), given_seg_fg.size(1) * given_seg_fg.size(2), 1)
        pm_feats_bg = pm_feats * given_seg_bg
        pm_feats_fg = pm_feats * given_seg_fg
        return [pm_feats_bg, pm_feats_fg]

    def generate_one_pixel_state(self, pm_feats_bg, pm_feats_fg):
        one_pixel_bg = pm_feats_bg.sum(1, keepdim=True).clamp(min=1e-7)
        one_pixel_fg = pm_feats_fg.sum(1, keepdim=True).clamp(min=1e-7)
        one_pixel_bg = one_pixel_bg / one_pixel_bg.norm(dim=-1, keepdim=True)
        one_pixel_fg = one_pixel_fg / one_pixel_fg.norm(dim=-1, keepdim=True)
        return one_pixel_bg, one_pixel_fg

    def get_conv_feats_norm(self, feats):
        conv_feats = self.conv_in(feats)
        conv_feats_norm = conv_feats / conv_feats.norm(dim=1, keepdim=True)
        return conv_feats_norm

    def update_state(self, conv_feats_norm, pred_seg, state):
        pm_feats = conv_feats_norm.reshape(conv_feats_norm.size(0), conv_feats_norm.size(1), conv_feats_norm.size(2) * conv_feats_norm.size(3))
        pm_feats = pm_feats.permute([0, 2, 1])
        pred_seg_bg = pred_seg[:, 0, :, :]
        pred_seg_bg = pred_seg_bg.reshape(pred_seg_bg.size(0), pred_seg_bg.size(1) * pred_seg_bg.size(2), 1)
        pred_seg_fg = pred_seg[:, 1, :, :]
        pred_seg_fg = pred_seg_fg.reshape(pred_seg_fg.size(0), pred_seg_fg.size(1) * pred_seg_fg.size(2), 1)
        pm_feats_bg = pm_feats * pred_seg_bg
        pm_feats_fg = pm_feats * pred_seg_fg
        state['pm_prev'] = [pm_feats_bg, pm_feats_fg]

        one_pixel_bg = pm_feats_bg.sum(1, keepdim=True).clamp(min=1e-7)
        one_pixel_fg = pm_feats_fg.sum(1, keepdim=True).clamp(min=1e-7)
        one_pixel_bg = one_pixel_bg / one_pixel_bg.norm(dim=-1, keepdim=True)
        one_pixel_fg = one_pixel_fg / one_pixel_fg.norm(dim=-1, keepdim=True)
        pm_feats_bg, pm_feats_fg = state['pm_int']
        pm_feats_bg = 0.9 * pm_feats_bg + 0.1 * one_pixel_bg
        pm_feats_fg = 0.9 * pm_feats_fg + 0.1 * one_pixel_fg
        state['pm_int'] = [pm_feats_bg, pm_feats_fg]
        return state

    def forward(self, feats, pm_state):
        B = feats.size(0)
        conv_feats_norm = self.get_conv_feats_norm(feats)

        init_pm_feats_bg, init_pm_feats_fg = pm_state['pm_init']
        init_bg_fg_corr_lst = []
        for i in range(B):
            temp_bg = F.conv2d(conv_feats_norm[i:i+1], init_pm_feats_bg[i].unsqueeze(2).unsqueeze(3))
            temp_bg, _ = temp_bg.max(1, keepdim=True)
            temp_fg = F.conv2d(conv_feats_norm[i:i+1], init_pm_feats_fg[i].unsqueeze(2).unsqueeze(3))
            temp_fg, _ = temp_fg.max(1, keepdim=True)
            init_bg_fg_corr_lst.append(torch.cat([temp_bg, temp_fg], 1))
        init_bg_fg_scores = torch.cat(init_bg_fg_corr_lst, 0)

        prev_pm_feats_bg, prev_pm_feats_fg = pm_state['pm_prev']
        prev_bg_fg_corr_lst = []
        for i in range(B):
            temp_bg = F.conv2d(conv_feats_norm[i:i+1], prev_pm_feats_bg[i].unsqueeze(2).unsqueeze(3))
            temp_bg, _ = temp_bg.max(1, keepdim=True)
            temp_fg = F.conv2d(conv_feats_norm[i:i+1], prev_pm_feats_fg[i].unsqueeze(2).unsqueeze(3))
            temp_fg, _ = temp_fg.max(1, keepdim=True)
            prev_bg_fg_corr_lst.append(torch.cat([temp_bg, temp_fg], 1))
        prev_bg_fg_scores = torch.cat(prev_bg_fg_corr_lst, 0)

        int_pm_feats_bg, int_pm_feats_fg = pm_state['pm_int']
        int_bg_fg_corr_lst = []
        for i in range(B):
            temp_bg = F.conv2d(conv_feats_norm[i:i+1], int_pm_feats_bg[i].unsqueeze(2).unsqueeze(3))
            temp_fg = F.conv2d(conv_feats_norm[i:i+1], int_pm_feats_fg[i].unsqueeze(2).unsqueeze(3))
            int_bg_fg_corr_lst.append(torch.cat([temp_bg, temp_fg], 1))
        int_bg_fg_scores = torch.cat(int_bg_fg_corr_lst, 0)
        return init_bg_fg_scores, prev_bg_fg_scores, int_bg_fg_scores, conv_feats_norm


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(SpatialAttentionModule, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv(in_dim, in_dim, 1, 1, 0)
        self.key_conv = Conv(in_dim, in_dim, 1, 1, 0)
        self.value_conv = Conv(in_dim, in_dim, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma*out + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(B, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)
        out = self.gamma*out + x
        return out


class SelfAttentionModule(nn.Module):
    def __init__(self, map_count):
        super().__init__()
        self.sa = SpatialAttentionModule(map_count)
        self.ca = ChannelAttentionModule()

    def forward(self, cor_bg, cor_fg):
        sa_cor_bg = self.sa(cor_bg)
        ca_cor_bg = self.ca(cor_bg)
        sa_cor_fg = self.sa(cor_fg)
        ca_cor_fg = self.ca(cor_fg)
        return sa_cor_bg, ca_cor_bg, sa_cor_fg, ca_cor_fg


class REFINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s16 = ConvRelu(2048, 256, 1, 1, 0)
        self.blend_s16 = ConvRelu(256 + 12 + 2, 128, 3, 1, 1)
        self.conv_s8 = ConvRelu(512, 128, 1, 1, 0)
        self.blend_s8 = ConvRelu(130, 128, 3, 1, 1)
        self.conv_s4 = ConvRelu(256, 128, 1, 1, 0)
        self.blend_s4 = ConvRelu(130, 128, 3, 1, 1)
        self.deconv1_1 = nn.ConvTranspose2d(128, 2, 4, 2, 1)
        self.deconv1_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 2, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 2, 3, 1, 1)
        self.predictor = nn.ConvTranspose2d(6, 2, 6, 4, 1)

    def forward(self, feats, sa_cor_bg, ca_cor_bg, sa_cor_fg, ca_cor_fg, state):
        u = torch.cat([self.conv_s16(feats['s16']), sa_cor_bg, ca_cor_bg, sa_cor_fg, ca_cor_fg, state['prev_seg']], dim=-3)
        u = self.blend_s16(u)
        out_16 = self.deconv1_1(u)
        u = torch.cat([self.conv_s8(feats['s8']), out_16], dim=-3)
        u = self.blend_s8(u)
        out_8 = self.deconv2(u)
        u = torch.cat([self.conv_s4(feats['s4']), out_8], dim=-3)
        u = self.blend_s4(u)
        out_4 = self.deconv3(u)
        segscore = self.predictor(torch.cat([self.deconv1_2(out_16), out_8, out_4], dim=1))
        return segscore


class VOS(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()
        self.backbone = getattr(models.backbones, backbone_cfg[0])(*backbone_cfg[1])
        self.pm = PixelLevelMatching(2048, 512)
        self.att = SelfAttentionModule(3)
        self.refine = REFINE()

    def get_init_state(self, feats, given_seg):
        state = {}
        state['pm_init'] = self.pm.generate_state(feats['s16'], given_seg)
        state['pm_prev'] = state['pm_init'].copy()
        state['pm_int_temp'] = state['pm_init'].copy()
        state['pm_int'] = self.pm.generate_one_pixel_state(state['pm_int_temp'][0], state['pm_int_temp'][1])
        state['prev_seg'] = given_seg
        return state

    def update(self, pred_seg, state):
        state = self.pm.update_state(state['conv_feats_norm'], pred_seg, state)
        state['prev_seg'] = pred_seg
        return state

    def extract_feats(self, img):
        feats = self.backbone.get_features(img)
        return feats

    def forward(self, feats, full_state, object_ids):
        segscore = {}
        for id in object_ids:
            init_pm_score, prev_pm_score, int_pm_score, full_state[id]['conv_feats_norm'] = self.pm(feats['s16'], full_state[id])
            cor_bg = torch.cat([init_pm_score[:, :1], prev_pm_score[:, :1], int_pm_score[:, :1]], dim=1)
            cor_fg = torch.cat([init_pm_score[:, 1:], prev_pm_score[:, 1:], int_pm_score[:, 1:]], dim=1)
            sa_cor_bg, ca_cor_bg, sa_cor_fg, ca_cor_fg = self.att(cor_bg, cor_fg)
            segscore[id] = self.refine(feats, sa_cor_bg, ca_cor_bg, sa_cor_fg, ca_cor_fg, full_state[id])
        return full_state, segscore


class PMVOS(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.vos = VOS(backbone)

    def forward(self, x, given_labels=None, state=None):
        batchsize, nframes, nchannels, prepad_height, prepad_width = x.size()

        required_padding = get_required_padding(prepad_height, prepad_width, 16)
        if tuple(required_padding) != (0, 0, 0, 0):
            x, given_labels = apply_padding(x, given_labels, required_padding)
        _, _, _, height, width = x.size()
        video_frames = [elem.view(batchsize, nchannels, height, width) for elem in x.split(1, dim=1)]
        feats = self.vos.extract_feats(video_frames[0])
        init_label = given_labels[0]
        object_ids = init_label.unique().tolist()
        if 0 in object_ids:
            object_ids.remove(0)

        state = {}
        for obj_idx in object_ids:
            given_seg = F.avg_pool2d(torch.cat([init_label != obj_idx, init_label == obj_idx], dim=-3).float(), 16)
            state[obj_idx] = self.vos.get_init_state(feats, given_seg)

        seg_lst = []
        seg_lst.append(given_labels[0])
        frames_to_process = range(1, nframes)

        for i in frames_to_process:
            feats = self.vos.extract_feats(video_frames[i])
            state, segscore = self.vos(feats, state, object_ids)
            predicted_seg = {k: F.softmax(segscore[k], dim=-3) for k in object_ids}
            output_seg, aggregated_seg = softmax_aggregate(predicted_seg, object_ids)
            update_seg = {n: F.avg_pool2d(aggregated_seg[n], 16) for n in object_ids}
            for k in object_ids:
                state[k] = self.vos.update(update_seg[k], state[k])
            seg_lst.append(output_seg)

        output = {}
        output['segs'] = torch.stack(seg_lst, dim=1)
        output['segs'] = unpad(output['segs'], required_padding)
        return output, state