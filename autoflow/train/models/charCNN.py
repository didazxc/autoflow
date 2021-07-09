import torch
from torch import nn


class CharCNN(nn.Module):

    def __init__(self, filter_sizes, embed_size=100, chars_size=5000, output_size=2, out_channel_size=3):
        super().__init__()
        self.embed = nn.Embedding(chars_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embed_size, out_channel_size, filter_size))
            for filter_size in filter_sizes
        ])
        self.conv_out_size = out_channel_size * len(filter_sizes)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.conv_out_size, output_size)
        )

    def forward(self, data):
        embed_out = self.embed(data).permute(0, 2, 1)
        conv_outs = [conv(embed_out) for conv in self.convs]
        pool_outs = [nn.functional.max_pool1d(out, out.shape[-1]) for out in conv_outs]
        convs_out = torch.cat(pool_outs, dim=1).view(-1, self.conv_out_size)
        output = self.output_layer(convs_out)
        return output

    @staticmethod
    def category_from_output(output):
        return output.topk(1).indices[0].item()

    @classmethod
    def category_from_outputs(cls, outputs):
        return [cls.category_from_output(output) for output in outputs]
