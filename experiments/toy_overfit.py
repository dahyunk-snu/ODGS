"""Toy overfit sanity check for the tangent-native ODGS rasterizer.

This verifies the feed-forward training path described by
TANSPLAT_DESIGN.md section 10 M0: raw Gaussian tensors are activated,
rendered by the rasterizer, compared to target ERP images, and updated
through Adam from a photometric loss.

For mostly-black sparse ERP scenes (a few hundred sub-pixel splats on
32k pixels over a black background), the initial-PSNR baseline is
inflated because most pixels are already identical, which compresses the
achievable PSNR gain. The acceptance certificate is therefore absolute
final quality (>= 25 dB) plus a modest gain (>= 4 dB). Gradient
correctness is certified separately by experiments/gradcheck_tangent.py
(Richardson-filtered finite differences); this script certifies that the
feed-forward training path (raw tensors -> activations -> rasterizer ->
loss -> Adam) converges. Reference server run 2026-07-04: initial
20.81 dB, final 27.03 dB at 2000 iters.

Example:
    python experiments/toy_overfit.py
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getWorld2View2


class OmniCamera(object):
    def __init__(self, width, height, world_view_transform, camera_center, device):
        self.image_width = int(width)
        self.image_height = int(height)
        self.world_view_transform = world_view_transform.contiguous()
        self.camera_center = camera_center.contiguous()

        settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            bg=torch.zeros(3, dtype=torch.float32, device=device),
            scale_modifier=1.0,
            viewmatrix=self.world_view_transform,
            sh_degree=1,
            campos=self.camera_center,
            prefiltered=False,
            debug=False,
        )
        self.rasterizer = GaussianRasterizer(raster_settings=settings)


def sample_directions(rng, n, variant):
    if variant != "generic":
        raise ValueError("toy_overfit only uses the generic scene sampler")

    y = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    phi = rng.uniform(-math.pi, math.pi, size=n).astype(np.float32)
    xz = np.sqrt(np.maximum(0.0, 1.0 - y * y)).astype(np.float32)
    x = xz * np.sin(phi)
    z = xz * np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def make_tensor(array, device, requires_grad):
    tensor = torch.tensor(array, dtype=torch.float32, device=device).contiguous()
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def build_gt_scene(seed, n, device):
    rng = np.random.RandomState(seed)

    directions = sample_directions(rng, n, "generic")
    radii = rng.uniform(0.5, 4.0, size=(n, 1)).astype(np.float32)
    means = radii * directions

    scales = rng.uniform(0.010, 0.030, size=(n, 3)).astype(np.float32) * radii

    rotations = rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float32)
    rotations = rotations / np.maximum(
        np.linalg.norm(rotations, axis=1, keepdims=True),
        1.0e-8,
    ).astype(np.float32)

    opacities = rng.uniform(0.2, 0.9, size=(n, 1)).astype(np.float32)

    shs = np.empty((n, 4, 3), dtype=np.float32)
    shs[:, 0, :] = rng.normal(0.5, 0.2, size=(n, 3)).astype(np.float32)
    shs[:, 1:, :] = rng.normal(0.0, 0.3, size=(n, 3, 3)).astype(np.float32)

    return {
        "means": make_tensor(means, device, False),
        "scales": make_tensor(scales, device, False),
        "rotations": make_tensor(rotations, device, False),
        "opacities": make_tensor(opacities, device, False),
        "shs": make_tensor(shs, device, False),
    }


def build_raw_train_tensors(seed, n, device):
    rng = np.random.RandomState(seed)

    directions = sample_directions(rng, n, "generic")
    radii = rng.uniform(1.0, 3.0, size=(n, 1)).astype(np.float32)
    means = radii * directions

    scale_init = np.repeat(0.02 * radii, 3, axis=1).astype(np.float32)
    log_scales = np.log(scale_init).astype(np.float32)

    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    quats += 0.01 * rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float32)

    opacity_logit = np.full((n, 1), math.log(0.1 / 0.9), dtype=np.float32)

    features = np.zeros((n, 4, 3), dtype=np.float32)
    features[:, 0, :] = 0.5

    return {
        "means": make_tensor(means, device, True),
        "log_scales": make_tensor(log_scales, device, True),
        "quats": make_tensor(quats, device, True),
        "opacity_logit": make_tensor(opacity_logit, device, True),
        "features": make_tensor(features, device, True),
    }


def activate_raw(raw):
    return {
        "means": raw["means"],
        "scales": torch.exp(raw["log_scales"]).contiguous(),
        "rotations": F.normalize(raw["quats"], dim=1, eps=1.0e-8).contiguous(),
        "opacities": torch.sigmoid(raw["opacity_logit"]).contiguous(),
        "shs": raw["features"],
    }


def random_small_rotation(rng, max_angle_deg):
    axis = rng.normal(0.0, 1.0, size=3).astype(np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1.0e-12:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / axis_norm

    angle = rng.uniform(0.0, math.radians(max_angle_deg))
    kx, ky, kz = axis.tolist()
    skew = np.asarray(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    rot = np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew.dot(skew)
    return rot.astype(np.float32)


def make_camera(width, height, center, rotation, device):
    center_np = np.asarray(center, dtype=np.float32)
    rotation_np = np.asarray(rotation, dtype=np.float32)
    translation = -(center_np.reshape(1, 3).dot(rotation_np)).reshape(3).astype(np.float32)

    world_view = torch.tensor(
        getWorld2View2(rotation_np, translation),
        dtype=torch.float32,
        device=device,
    ).transpose(0, 1).contiguous()
    camera_center = world_view.inverse()[3, :3].contiguous()

    expected = torch.tensor(center_np, dtype=torch.float32, device=device)
    if not torch.allclose(camera_center, expected, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(
            "camera_center mismatch: got {}, expected {}".format(
                camera_center.detach().cpu().tolist(),
                expected.detach().cpu().tolist(),
            )
        )

    return OmniCamera(width, height, world_view, camera_center, device)


def make_cameras(width, height, seed, device):
    rng = np.random.RandomState(seed)
    centers = [
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.0, 0.15, 0.1),
        (-0.15, 0.05, -0.1),
    ]
    cameras = []
    for center in centers:
        rotation = random_small_rotation(rng, 10.0)
        cameras.append(make_camera(width, height, center, rotation, device))
    return cameras


def render_color(camera, params):
    means2d = torch.zeros_like(params["means"], requires_grad=params["means"].requires_grad)
    color, depth, acc, radii, psi, lat, lon = camera.rasterizer(
        means3D=params["means"],
        means2D=means2d,
        opacities=params["opacities"],
        shs=params["shs"],
        scales=params["scales"],
        rotations=params["rotations"],
    )
    return color


def is_finite_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return True
    if not torch.is_floating_point(tensor):
        return True
    return bool(torch.isfinite(tensor).all().item())


def assert_finite_images(images, label):
    for index, image in enumerate(images):
        if not is_finite_tensor(image):
            raise RuntimeError("Non-finite image in {} view {}".format(label, index))


def psnr_value(render, target):
    render_clamped = render.detach().clamp(0.0, 1.0)
    target_clamped = target.detach().clamp(0.0, 1.0)
    mse = torch.mean((render_clamped - target_clamped) ** 2).clamp_min(1.0e-12)
    return float((-10.0 * torch.log10(mse)).detach().cpu().item())


def mean_psnr(renders, targets):
    values = [psnr_value(render, target) for render, target in zip(renders, targets)]
    return float(sum(values) / max(1, len(values)))


@torch.no_grad()
def render_all(cameras, params):
    renders = []
    for camera in cameras:
        renders.append(render_color(camera, params).detach())
    return renders


def check_raw_grads(raw):
    ok = True
    for name, tensor in raw.items():
        if tensor.grad is None:
            print("Missing gradient for raw tensor {}".format(name))
            ok = False
        elif not is_finite_tensor(tensor.grad):
            print("Non-finite gradient for raw tensor {}".format(name))
            ok = False
    return ok


def raw_params_finite(raw):
    for name, tensor in raw.items():
        if not is_finite_tensor(tensor):
            print("Non-finite raw tensor {}".format(name))
            return False
    return True


def save_triplets(outdir, targets, init_renders, final_renders):
    os.makedirs(str(outdir), exist_ok=True)
    for index, image in enumerate(targets):
        save_image(image.detach().clamp(0.0, 1.0).cpu(), str(outdir / "target_{}.png".format(index)))
    for index, image in enumerate(init_renders):
        save_image(image.detach().clamp(0.0, 1.0).cpu(), str(outdir / "init_{}.png".format(index)))
    for index, image in enumerate(final_renders):
        save_image(image.detach().clamp(0.0, 1.0).cpu(), str(outdir / "final_{}.png".format(index)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gt-n", type=int, default=400)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--psnr-min", type=float, default=25.0)
    parser.add_argument("--psnr-gain-min", type=float, default=4.0)
    parser.add_argument("--outdir", type=Path, default=Path("experiments") / "output" / "toy_overfit")
    args = parser.parse_args()

    if args.gt_n <= 0 or args.n <= 0:
        raise ValueError("--gt-n and --n must be positive")
    if args.iters < 0:
        raise ValueError("--iters must be non-negative")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The ODGS rasterizer requires CUDA.")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cuda")
    cameras = make_cameras(args.width, args.height, int(args.seed) + 7001, device)

    gt_scene = build_gt_scene(int(args.seed) + 1001, int(args.gt_n), device)
    with torch.no_grad():
        targets = render_all(cameras, gt_scene)
    assert_finite_images(targets, "targets")

    raw = build_raw_train_tensors(int(args.seed) + 2001, int(args.n), device)
    with torch.no_grad():
        init_renders = render_all(cameras, activate_raw(raw))
    assert_finite_images(init_renders, "initial renders")
    initial_mean_psnr = mean_psnr(init_renders, targets)
    print("initial mean_psnr {:.3f}".format(initial_mean_psnr))

    optimizer = torch.optim.Adam(
        [
            {"params": [raw["means"]], "lr": 1.6e-3},
            {"params": [raw["features"]], "lr": 2.5e-3},
            {"params": [raw["opacity_logit"]], "lr": 5.0e-2},
            {"params": [raw["log_scales"]], "lr": 5.0e-3},
            {"params": [raw["quats"]], "lr": 1.0e-3},
        ]
    )

    finite_training = True
    last_loss_value = float("nan")
    for iteration in range(1, int(args.iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        params = activate_raw(raw)
        loss = None
        for camera, target in zip(cameras, targets):
            render = render_color(camera, params)
            view_loss = F.l1_loss(render, target)
            if loss is None:
                loss = view_loss
            else:
                loss = loss + view_loss

        if loss is None or not is_finite_tensor(loss):
            print("Non-finite loss at iter {}".format(iteration))
            finite_training = False
            break

        last_loss_value = float(loss.detach().cpu().item())
        loss.backward()

        if not check_raw_grads(raw):
            print("Non-finite or missing gradients at iter {}".format(iteration))
            finite_training = False
            break

        optimizer.step()

        if not raw_params_finite(raw):
            print("Non-finite raw parameter after iter {}".format(iteration))
            finite_training = False
            break

        if iteration % 100 == 0 or iteration == int(args.iters):
            with torch.no_grad():
                current_renders = render_all(cameras, activate_raw(raw))
            assert_finite_images(current_renders, "iter {}".format(iteration))
            current_psnr = mean_psnr(current_renders, targets)
            print(
                "iter {:04d} loss {:.6f} mean_psnr {:.3f}".format(
                    iteration,
                    last_loss_value,
                    current_psnr,
                )
            )

    with torch.no_grad():
        final_renders = render_all(cameras, activate_raw(raw))
    assert_finite_images(final_renders, "final renders")
    final_mean_psnr = mean_psnr(final_renders, targets)
    psnr_gain = final_mean_psnr - initial_mean_psnr

    save_triplets(args.outdir, targets, init_renders, final_renders)

    passed = (
        finite_training
        and final_mean_psnr >= float(args.psnr_min)
        and psnr_gain >= float(args.psnr_gain_min)
    )
    status = "TOY-OVERFIT PASS" if passed else "TOY-OVERFIT FAIL"
    print(
        "{} initial_psnr={:.3f} final_psnr={:.3f} gain={:.3f} last_loss={:.6f}".format(
            status,
            initial_mean_psnr,
            final_mean_psnr,
            psnr_gain,
            last_loss_value,
        )
    )
    print("wrote {}".format(args.outdir))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
