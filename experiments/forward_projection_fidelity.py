import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


@dataclass
class OmniCamera:
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    camera_center: torch.Tensor


def make_camera(width, height, device):
    view = torch.eye(4, dtype=torch.float32, device=device)
    return OmniCamera(
        image_width=width,
        image_height=height,
        world_view_transform=view,
        camera_center=torch.zeros(3, dtype=torch.float32, device=device),
    )


def spherical_mean(radius, theta_deg, phi_deg, device):
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return torch.tensor(
        [
            radius * math.cos(theta) * math.sin(phi),
            -radius * math.sin(theta),
            radius * math.cos(theta) * math.cos(phi),
        ],
        dtype=torch.float32,
        device=device,
    )


def render_single_gaussian(camera, mean, sigma, opacity, device):
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    xyz = mean.reshape(1, 3).contiguous()
    means2d = torch.zeros_like(xyz, requires_grad=False)
    colors = torch.ones((1, 3), dtype=torch.float32, device=device)
    opacities = torch.full((1, 1), opacity, dtype=torch.float32, device=device)
    scales = torch.full((1, 3), sigma, dtype=torch.float32, device=device)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    color, depth, acc, radii, psi, lat, lon = rasterizer(
        means3D=xyz,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )
    return {
        "image": color.mean(dim=0),
        "acc": acc,
        "depth": depth,
        "radii": radii,
        "psi": psi,
        "lat": lat,
        "lon": lon,
    }


@torch.no_grad()
def raytrace_single_gaussian(mean, sigma, opacity, width, height, footprint_radius, device):
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    lon = (xs * 2.0 / width - 1.0) * math.pi
    lat = (0.5 - ys / height) * math.pi
    dirs = torch.stack(
        [torch.cos(lat) * torch.sin(lon), -torch.sin(lat), torch.cos(lat) * torch.cos(lon)],
        dim=-1,
    )
    t = (dirs * mean).sum(dim=-1)
    dist2 = ((mean * mean).sum() - t.square()).clamp_min(0.0)
    line_integral = 0.5 * torch.exp(-0.5 * dist2 / (sigma * sigma)) * torch.erfc(
        -t / (math.sqrt(2.0) * sigma)
    )
    alpha = torch.clamp(opacity * line_integral, max=0.99)

    dist_xz = torch.sqrt(mean[0] * mean[0] + mean[2] * mean[2])
    center_x = ((torch.atan2(mean[0], mean[2]) / math.pi + 1.0) * width * 0.5).item()
    center_y = ((0.5 - torch.atan2(-mean[1], dist_xz) / math.pi) * height).item()
    radius = footprint_radius.item()
    footprint = (xs - center_x).abs().le(radius) & (ys - center_y).abs().le(radius)
    return torch.where(footprint & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))


def center_intensity(image, center_x, center_y):
    h, w = image.shape
    x = int(round(center_x))
    y = int(round(center_y))
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    return image[y, x]


def crop_centered(image, center_x, center_y, crop_size):
    h, w = image.shape
    half = crop_size // 2
    out = np.zeros((crop_size, crop_size), dtype=image.dtype)

    x0 = int(round(center_x)) - half
    y0 = int(round(center_y)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def save_visualization(cases, sigmas, thetas, out_path):
    import matplotlib.pyplot as plt

    nrows = len(sigmas)
    ncols = len(thetas) * 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.1 * ncols, 2.25 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    case_lookup = {(sigma_key(c["sigma"]), theta_key(c["theta"])): c for c in cases}
    for row, sigma in enumerate(sigmas):
        for theta_idx, theta in enumerate(thetas):
            case = case_lookup[(sigma_key(sigma), theta_key(theta))]
            panels = [
                ("renderer", case["renderer_crop"], "magma", 0.0, 1.0),
                ("ray trace", case["raytrace_crop"], "magma", 0.0, 1.0),
                ("diff", case["diff_crop"], "coolwarm", -1.0, 1.0),
            ]
            for panel_idx, (name, image, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[row][theta_idx * 3 + panel_idx]
                ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"theta={theta:g} {name}", fontsize=9)
                if theta_idx == 0 and panel_idx == 0:
                    ax.set_ylabel(f"sigma={sigma:g}", fontsize=10)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_l1_plot(rows, sigmas, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sigma in sigmas:
        sigma_rows = [r for r in rows if sigma_key(r["sigma"]) == sigma_key(sigma)]
        sigma_rows.sort(key=lambda r: r["theta_deg"])
        ax.plot(
            [r["theta_deg"] for r in sigma_rows],
            [r["l1"] for r in sigma_rows],
            marker="o",
            markersize=2.4,
            linewidth=1.5,
            label=f"sigma={sigma:g}",
        )

    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("L1 loss")
    ax.set_ylim(0, 0.042)
    ax.set_title("Renderer vs Ray-traced L1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def theta_key(theta):
    return round(float(theta), 8)


def sigma_key(sigma):
    return round(float(sigma), 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--opacity", type=float, default=0.95)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--plot-sigmas", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--thetas", type=float, nargs="+", default=[0.0, 40, 55.0, 70.0, 85.0])
    parser.add_argument("--loss-theta-min", type=float, default=0.0)
    parser.add_argument("--loss-theta-max", type=float, default=89.0)
    parser.add_argument("--loss-theta-step", type=float, default=1.0)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./experiments") / "output" / "forward_projection_fidelity",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The existing ODGS rasterizer requires CUDA.")
    if args.loss_theta_step <= 0:
        raise ValueError("--loss-theta-step must be positive.")

    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    camera = make_camera(args.width, args.height, device)
    cases = []
    metrics = []
    loss_rows = []

    loss_thetas = np.arange(
        args.loss_theta_min,
        args.loss_theta_max + 0.5 * args.loss_theta_step,
        args.loss_theta_step,
        dtype=np.float32,
    ).tolist()
    visual_sigma_keys = {sigma_key(sigma) for sigma in args.sigmas}
    plot_sigma_keys = {sigma_key(sigma) for sigma in args.plot_sigmas}
    visual_theta_keys = {theta_key(theta) for theta in args.thetas}
    loss_theta_keys = {theta_key(theta) for theta in loss_thetas}
    all_sigmas = sorted(
        {sigma_key(sigma): float(sigma) for sigma in [*args.sigmas, *args.plot_sigmas]}.values()
    )
    all_thetas = sorted(
        {theta_key(theta): float(theta) for theta in [*args.thetas, *loss_thetas]}.values()
    )

    with torch.no_grad():
        for sigma in all_sigmas:
            for theta in all_thetas:
                mean = spherical_mean(args.radius, theta, args.phi, device)
                rendered = render_single_gaussian(camera, mean, sigma, args.opacity, device)
                raytrace = raytrace_single_gaussian(
                    mean,
                    sigma,
                    args.opacity,
                    args.width,
                    args.height,
                    rendered["radii"][0],
                    device,
                )

                theta_rad = math.radians(theta)
                phi_rad = math.radians(args.phi)
                center_x = (phi_rad / math.pi + 1.0) * args.width * 0.5
                center_y = (0.5 - theta_rad / math.pi) * args.height
                renderer_center = center_intensity(rendered["image"], center_x, center_y)
                raytrace_center = center_intensity(raytrace, center_x, center_y)

                renderer_image = rendered["image"].detach().cpu().numpy()
                raytrace_image = raytrace.detach().cpu().numpy()
                diff = renderer_image - raytrace_image
                l1 = float(np.mean(np.abs(diff)))

                if sigma_key(sigma) in visual_sigma_keys and theta_key(theta) in visual_theta_keys:
                    case = {
                        "sigma": sigma,
                        "theta": theta,
                        "renderer": renderer_image,
                        "raytrace": raytrace_image,
                        "diff": diff,
                        "renderer_crop": crop_centered(renderer_image, center_x, center_y, args.crop_size),
                        "raytrace_crop": crop_centered(raytrace_image, center_x, center_y, args.crop_size),
                        "diff_crop": crop_centered(diff, center_x, center_y, args.crop_size),
                    }
                    cases.append(case)

                if sigma_key(sigma) in plot_sigma_keys and theta_key(theta) in loss_theta_keys:
                    loss_rows.append(
                        {
                            "sigma": sigma,
                            "theta_deg": theta,
                            "l1": l1,
                        }
                    )

                metrics.append(
                    {
                        "sigma": sigma,
                        "theta_deg": theta,
                        "mse": float(np.mean(diff * diff)),
                        "mae": l1,
                        "renderer_peak": float(rendered["image"].max().detach().cpu()),
                        "raytrace_peak": float(raytrace.max().detach().cpu()),
                        "renderer_center_intensity": float(renderer_center.detach().cpu()),
                        "raytrace_center_intensity": float(raytrace_center.detach().cpu()),
                        "renderer_radius_px": int(rendered["radii"][0].detach().cpu()),
                        "renderer_lat_rad": float(rendered["lat"][0].detach().cpu()),
                        "renderer_lon_rad": float(rendered["lon"][0].detach().cpu()),
                    }
                )

    save_visualization(
        cases,
        args.sigmas,
        args.thetas,
        args.out_dir / "comparison_patches.png",
    )
    save_l1_plot(loss_rows, args.plot_sigmas, args.out_dir / "l1_loss_vs_theta.png")

    np.savez_compressed(
        args.out_dir / "comparison_maps.npz",
        renderer=np.stack([c["renderer"] for c in cases]),
        raytrace=np.stack([c["raytrace"] for c in cases]),
        diff=np.stack([c["diff"] for c in cases]),
        sigmas=np.asarray([c["sigma"] for c in cases], dtype=np.float32),
        thetas=np.asarray([c["theta"] for c in cases], dtype=np.float32),
    )

    with (args.out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    with (args.out_dir / "l1_loss_vs_theta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loss_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loss_rows)

    print(f"Wrote {args.out_dir / 'comparison_patches.png'}")
    print(f"Wrote {args.out_dir / 'l1_loss_vs_theta.png'}")
    print(f"Wrote {args.out_dir / 'l1_loss_vs_theta.csv'}")
    print(f"Wrote {args.out_dir / 'metrics.csv'}")
    print(f"Wrote {args.out_dir / 'comparison_maps.npz'}")


if __name__ == "__main__":
    main()
