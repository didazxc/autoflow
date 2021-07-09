import torch
from torch import nn


class UserModel(nn.Module):

    def __init__(self, filter_sizes, userprofile_size, applist_size, output_size=32, char_embed_size=32,
                 chars_size=6000, out_channel_size=3):
        super().__init__()
        # embedding of lines and apps
        self.embed = nn.Embedding(chars_size, char_embed_size)
        # charCNN
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(char_embed_size, out_channel_size, filter_size))
            for filter_size in filter_sizes
        ])
        self.conv_out_size = out_channel_size * len(filter_sizes)
        self.lines_output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.conv_out_size, output_size),
            nn.ReLU()
        )
        # combine
        self.output_layer = nn.Sequential(
            nn.Linear(output_size+applist_size+userprofile_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU()
        )

    def forward(self, lines, applist, userprofile):
        # lines
        lines_embed_out = self.embed(lines).permute(0, 2, 1)
        conv_outs = [conv(lines_embed_out) for conv in self.convs]
        pool_outs = [nn.functional.max_pool1d(out, out.shape[-1]) for out in conv_outs]
        convs_out = torch.cat(pool_outs, dim=1).view(-1, self.conv_out_size)
        lines_output = self.lines_output_layer(convs_out)
        # combine
        output = torch.cat([lines_output, applist, userprofile], dim=-1)
        output = self.output_layer(output)
        return output


class DS(nn.Module):
    def __init__(self, userprofile_size=1384, applist_size=1000, embed_size=32,
                 vid_table_size=22261, aid_table_size=13727):
        super().__init__()
        self.user_embed = UserModel([3, 5, 10], userprofile_size, applist_size, output_size=embed_size)
        self.vid_embed = nn.Embedding(vid_table_size, embed_size)
        self.aid_embed = nn.Embedding(aid_table_size, embed_size)
        self.video_output_layer = nn.Sequential(
            nn.Linear(2*embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU()
        )

    def forward(self, vid, aid, lines, applist, userprofile):
        user = self.user_embed(lines, applist, userprofile)
        video = self.video_output_layer(torch.cat([self.vid_embed(vid), self.aid_embed(aid)], dim=-1))
        cosine = torch.cosine_similarity(user, video, dim=-1)
        return cosine

    @staticmethod
    def get_cate(x):
        return 1 if x >= 0.5 else 0

