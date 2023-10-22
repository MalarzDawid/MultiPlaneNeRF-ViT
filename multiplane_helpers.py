import torch
import torchvision
from vit import VisionTransformer

INPUT_CONV_CH = 5
PADDING = 10
CROP_SIZE = 5
CROP_STEP = CROP_SIZE // 2

class Args:
    def __init__(self) -> None:
        self.n_channels = 3
        self.embed_dim = 700
        self.patch_size = CROP_SIZE
        self.img_size = None
        self.n_attention_heads = 4
        self.forward_mul = 2
        self.n_classes = None
        self.n_layers = 6

# class ConvStage(torch.nn.Module):
#     def __init__(self, input_size: int = 5, output_size: int = 500):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.block = torch.nn.Sequential(
#             torch.nn.Conv2d(self.input_size, 8, (3, 3)),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d((2, 2)),
#             torch.nn.Conv2d(8, 16, (3, 3)),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d((2, 2)),
#             torch.nn.Conv2d(16, 32, (3, 3)),
#             torch.nn.ReLU(),
#         )
#         self.l1 = torch.nn.Linear(32 * 9 * 9, 1024)  # 5184 -> 1024 -> 512 -> 500
#         self.l2 = torch.nn.Linear(1024, 512)
#         self.l3 = torch.nn.Linear(512, self.output_size)

#     def forward(self, x):
#         x = self.block(x)
#         x = x.view(x.size(0), -1)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         return x


class RenderNetwork(torch.nn.Module):
    def __init__(self, input_size, dir_count):
        super().__init__()
        self.input_size = 3 * input_size + input_size * 2
        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )

        self.layers_main_2 = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )

        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(
                256 + self.input_size, 128
            ),  # dodane wejscie tutaj moze cos pomoze
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size + dir_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code, dirs):
        x = self.layers_main(triplane_code)
        x1 = torch.concat([x, triplane_code], dim=1)

        x = self.layers_main_2(x1)
        xs = torch.concat([x, triplane_code], dim=1)

        sigma = self.layers_sigma(xs)
        x = torch.concat([x, triplane_code, dirs], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class RenderNetworkEmbedded(torch.nn.Module):
    def __init__(
        self,
        input_size=100 * 3,
    ):
        input_size = input_size + 200 + 32
        super().__init__()
        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
        )
        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code):
        x = self.layers_main(triplane_code)
        sigma = self.layers_sigma(x)
        x = torch.concat([x, triplane_code], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class ImagePlanes(torch.nn.Module):
    def __init__(self, focal, poses, images, count, device="cuda"):
        super(ImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []
        # self.conv_stage = ConvStage(INPUT_CONV_CH, INPUT_CONV_CH * count)
        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = M @ torch.Tensor(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [
                    [self.focal.item(), 0, 0.5 * image.shape[0]],
                    [0, self.focal.item(), 0.5 * image.shape[0]],
                    [0, 0, 1],
                ]
            )

            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)

    def forward(self, points=None, transformer=None):
        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat(
            [points, torch.ones(points.shape[0], 1).to(points.device)], 1
        ).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T  # (x, y, z) -> (x, y, w)
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]  # remove w
        pixels = pixels / self.size
        pixels = torch.clamp(pixels, 0, 1)
        coord_norm = pixels * 2.0 - 1.0
        pixels = pixels * self.size
        pixels = pixels.permute(0, 2, 1)
        coord_norm = coord_norm.permute(0, 2, 1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            image_plane = self.image_plane[img]
            image_plane_border = torch.nn.functional.pad(
                image_plane,
                pad=(PADDING, PADDING, PADDING, PADDING),
                mode="constant",
                value=255,
            )
            coord = pixels[img]
            coord_norm_img = coord_norm[img]
            x = coord[:, 0] + PADDING - CROP_STEP
            y = coord[:, 1] + PADDING - CROP_STEP
            patches = []
            for r in range(CROP_SIZE):
                x_i = x + r
                patches.append(
                    torch.stack(
                        [
                            image_plane_border[:, x_i.long(), (y + c).long()].T
                            for c in range(CROP_SIZE)
                        ],
                        dim=-1,
                    )
                )
            
            
            patches = torch.stack(patches, dim=-2)
            # torch_ones = torch.ones((coord.size(0), 1, CROP_SIZE, CROP_SIZE))
            # for i in range(2):
            #     patches = torch.cat((patches, torch_ones), dim=1)
            #     patches[:, 3 + i, :, :] = patches[:, 3 + i, :, :] * coord_norm_img[
            #         :, i
            #     ].view(-1, 1, 1)
            conv_out = transformer(patches, coord_norm_img)
            feats.append(conv_out)
        # img: (100, 32k, 5, 5, 3) # coord: (100, 32k, 2)
        # img: (100, 32k, 96) coord: (100, 32k, 96)
        # [8, 32768, 5, 5, 5]
        feats = torch.stack(feats).squeeze(1)
        # coord_norm_flat = coord_norm.reshape(coord_norm.size(0)*coord_norm.size(1), 2)
        # feats = feats.reshape(feats.size(0)*feats.size(1), CROP_SIZE, CROP_SIZE, 3)
        # feats = feats.permute(0, 3, 1, 2)
        # feats = feats.permute(1, 0, 2, 3, 4)
        # # TODO -> 10
        # feats = feats.reshape(feats.size(0), 10, 10, 5, CROP_SIZE, CROP_SIZE)
        # feats = feats.permute(0, 3, 1, 4, 2, 5)
        # feats = feats.flatten(4)
        # feats = feats.permute(0, 1, 4, 2, 3)
        # feats = feats.flatten(3)
        # 32k, 3, 5, 5
        # for
        # print(feats)

        return conv_out


class LLFFImagePlanes(torch.nn.Module):
    def __init__(self, hwf, poses, images, count, device="cuda"):
        super(LLFFImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.H, self.W, self.focal = hwf

        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = torch.cat([M, torch.Tensor([[0, 0, 0, 1]]).to(M.device)], dim=0)

            M = M @ torch.Tensor(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [
                    [self.focal, 0, 0.5 * self.W],
                    [0, self.focal, 0.5 * self.H],
                    [0, 0, 1],
                ]
            )

            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)

    def forward(self, points=None):
        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat(
            [points, torch.ones(points.shape[0], 1).to(points.device)], 1
        ).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        pixels[:, 0] = torch.div(pixels[:, 0], self.W)
        pixels[:, 1] = torch.div(pixels[:, 1], self.H)
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            feats.append(feat)
        feats = torch.stack(feats).squeeze(1)
        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)
        feats = feats.permute(2, 3, 0, 1)
        feats = feats.flatten(2)

        feats = torch.cat((feats[0], pixels), 1)
        return feats


class ImageEmbedder(torch.nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((4, 4)),
            torch.nn.Conv2d(
                in_channels=3, out_channels=1, kernel_size=(3, 3), padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(625, 32),
        )

    def forward(self, input_image):
        input_image = torch.from_numpy(input_image).to("cuda")
        input_image = input_image.permute(2, 0, 1)
        return self.model(input_image)


class MultiImageNeRF(torch.nn.Module):
    def __init__(self, image_plane, count, dir_count):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetwork(count, dir_count)
        self.args = Args()
        self.args.n_classes = 3 * count + count * 2
        self.args.N_samples = 32
        self.args.N_rand = 128
        self.transformer =  VisionTransformer(self.args)
        self.input_ch_views = dir_count

    def parameters(self):
        return self.render_network.parameters()

    def forward(self, x):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts, self.transformer)
        return self.render_network(x, input_views)


class EmbeddedMultiImageNeRF(torch.nn.Module):
    def __init__(self, image_plane, count):
        super(EmbeddedMultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetworkEmbedded(count * 3)

    def parameters(self):
        return self.render_network.parameters()

    def set_embedding(self, emb):
        self.embedding = emb

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        x = self.image_plane(input_pts)
        e = self.embedding.repeat(x.shape[0], 1)
        x = torch.cat([x, e], -1)
        return self.render_network(x, input_views)
