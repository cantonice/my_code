import unittest
import torch
from torch.nn import Identity
from patch_embed import PatchEmbed

class TestPatchEmbed(unittest.TestCase):
    def setUp(self):
        self.patch_embed = PatchEmbed(img_size=224, in_c=3, embed_dim=768, patch_size=16, norm_layer=None)

    def test_forward_happy_path(self):
        # 测试正常情况
        x = torch.randn(1, 3, 224, 224)
        output = self.patch_embed(x)
        self.assertEqual(output.shape, (1, 196, 768))

    def test_forward_wrong_image_size(self):
        # 测试输入图片尺寸不符合要求的情况
        x = torch.randn(1, 3, 225, 224)
        with self.assertRaises(AssertionError):
            self.patch_embed(x)

    def test_forward_with_norm_layer(self):
        # 测试带有层归一化的情况
        patch_embed_with_norm = PatchEmbed(img_size=224, in_c=3, embed_dim=768, patch_size=16, norm_layer=Identity)
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed_with_norm(x)
        self.assertEqual(output.shape, (1, 196, 768))

    def test_forward_batch_size(self):
        # 测试不同批量大小的输入
        x = torch.randn(2, 3, 224, 224)
        output = self.patch_embed(x)
        self.assertEqual(output.shape, (2, 196, 768))

    def test_forward_different_channels(self):
        # 测试不同通道数的输入
        patch_embed_diff_channels = PatchEmbed(img_size=224, in_c=1, embed_dim=256, patch_size=16, norm_layer=None)
        x = torch.randn(1, 1, 224, 224)
        output = patch_embed_diff_channels(x)
        self.assertEqual(output.shape, (1, 196, 256))

if __name__ == '__main__':
    unittest.main()
