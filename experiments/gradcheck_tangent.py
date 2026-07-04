"""Finite-difference checks for the tangent-native ODGS rasterizer.

This implements TANGENT_MATH.md section 12 items 2 and 4:
finite-difference validation of the analytic backward on small random
scenes, plus finite-output/finite-gradient guards for generic, pole,
and seam-heavy ERP scenes.

CUDA atomics can make the backward slightly nondeterministic. The
analytic gradients reported here come from one backward call per scene.

Example:
    python experiments/gradcheck_tangent.py
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


TENSOR_NAMES = ("means3D", "scales", "rotations", "opacities", "shs")
LOSS_TERM_NAMES = ("color", "depth", "acc")
EPS_BY_TENSOR = {
    "means3D": 1.0e-3,
    "scales": 1.0e-4,
    "rotations": 1.0e-3,
    "opacities": 1.0e-3,
    "shs": 1.0e-3,
}


def parse_name_subset(text, valid_names, option_name):
    names = []
    valid = set(valid_names)
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in valid:
            raise ValueError("Unknown {} entries: {}".format(option_name, item))
        if item not in names:
            names.append(item)
    if not names:
        raise ValueError("--{} must name at least one entry".format(option_name))
    return tuple(names)


def parse_variants(text):
    variants = []
    for item in text.split(","):
        item = item.strip()
        if item:
            variants.append(item)
    valid = set(["generic", "pole", "seam"])
    unknown = [v for v in variants if v not in valid]
    if unknown:
        raise ValueError("Unknown variants: {}".format(",".join(unknown)))
    if not variants:
        raise ValueError("--variants must name at least one variant")
    return variants


def sample_directions(rng, n, variant):
    if variant == "generic":
        y = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        phi = rng.uniform(-math.pi, math.pi, size=n).astype(np.float32)
        xz = np.sqrt(np.maximum(0.0, 1.0 - y * y)).astype(np.float32)
        x = xz * np.sin(phi)
        z = xz * np.cos(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    if variant == "pole":
        max_delta = math.radians(8.0)
        cos_delta = rng.uniform(math.cos(max_delta), 1.0, size=n).astype(np.float32)
        delta = np.arccos(cos_delta)
        phi = rng.uniform(-math.pi, math.pi, size=n).astype(np.float32)
        y_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
        radial = np.sin(delta).astype(np.float32)
        x = radial * np.sin(phi)
        y = y_sign * cos_delta
        z = radial * np.cos(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    if variant == "seam":
        seam_width = math.radians(5.0)
        side = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
        offset = rng.uniform(0.0, seam_width, size=n).astype(np.float32)
        phi = np.where(side > 0.0, math.pi - offset, -math.pi + offset).astype(np.float32)
        y = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        xz = np.sqrt(np.maximum(0.0, 1.0 - y * y)).astype(np.float32)
        x = xz * np.sin(phi)
        z = xz * np.cos(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    raise ValueError("Unhandled variant {}".format(variant))


def make_tensor(array, device, requires_grad):
    tensor = torch.tensor(array, dtype=torch.float32, device=device).contiguous()
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def build_scene(seed, n, variant, device, requires_grad, scale_lo, scale_hi):
    rng = np.random.RandomState(seed)

    directions = sample_directions(rng, n, variant)
    radii = rng.uniform(0.5, 4.0, size=(n, 1)).astype(np.float32)
    means = radii * directions

    scales = rng.uniform(float(scale_lo), float(scale_hi), size=(n, 3)).astype(np.float32) * radii

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
        "means3D": make_tensor(means, device, requires_grad),
        "scales": make_tensor(scales, device, requires_grad),
        "rotations": make_tensor(rotations, device, requires_grad),
        "opacities": make_tensor(opacities, device, requires_grad),
        "shs": make_tensor(shs, device, requires_grad),
    }


def make_settings(height, width, device):
    return GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=device),
        sh_degree=1,
        campos=torch.zeros(3, dtype=torch.float32, device=device),
        prefiltered=False,
        debug=False,
    )


def render_scene(scene, settings, track_means2d):
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2d = torch.zeros_like(scene["means3D"], requires_grad=track_means2d)
    outputs = rasterizer(
        means3D=scene["means3D"],
        means2D=means2d,
        opacities=scene["opacities"],
        shs=scene["shs"],
        scales=scene["scales"],
        rotations=scene["rotations"],
    )
    color, depth, acc = outputs[:3]
    return (color.double(), depth.double(), acc.double()) + tuple(outputs[3:])


def make_loss_weights(outputs, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    color, depth, acc = outputs[:3]
    return {
        "color": torch.randn(color.shape, dtype=torch.float64, device=device, generator=gen),
        "depth": torch.randn(depth.shape, dtype=torch.float64, device=device, generator=gen),
        "acc": torch.randn(acc.shape, dtype=torch.float64, device=device, generator=gen),
    }


def scalar_loss(outputs, weights, loss_terms):
    color, depth, acc = outputs[:3]
    loss = color.new_zeros(())
    if "color" in loss_terms:
        loss = loss + (weights["color"] * color).sum()
    if "depth" in loss_terms:
        loss = loss + 0.1 * (weights["depth"] * depth).sum()
    if "acc" in loss_terms:
        loss = loss + 0.1 * (weights["acc"] * acc).sum()
    return loss


def is_finite_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return True
    if not torch.is_floating_point(tensor):
        return True
    return bool(torch.isfinite(tensor).all().item())


def check_outputs_finite(outputs, label):
    ok = True
    names = ("color", "depth", "acc", "radii", "psi", "lat", "lon")
    for name, tensor in zip(names, outputs):
        if not is_finite_tensor(tensor):
            print("Non-finite output in {}: {}".format(label, name))
            ok = False
    return ok


def check_grads_finite(scene, label, tensor_names):
    ok = True
    for name in tensor_names:
        grad = scene[name].grad
        if grad is None:
            print("Missing gradient in {}: {}".format(label, name))
            ok = False
        elif not is_finite_tensor(grad):
            print("Non-finite gradient in {}: {}".format(label, name))
            ok = False
    return ok


def visible_flat_indices(tensor, visible):
    visible_np = visible.detach().cpu().numpy().astype(np.bool_)
    visible_ids = np.nonzero(visible_np)[0].astype(np.int64)
    if visible_ids.size == 0:
        return np.asarray([], dtype=np.int64)

    per_gaussian = int(np.prod(tensor.shape[1:]))
    offsets = np.arange(per_gaussian, dtype=np.int64)
    return (visible_ids[:, None] * per_gaussian + offsets[None, :]).reshape(-1)


@torch.no_grad()
def forward_loss_value(scene, settings, weights, loss_terms):
    outputs = render_scene(scene, settings, track_means2d=False)
    if not check_outputs_finite(outputs, "finite-difference forward"):
        return None
    loss = scalar_loss(outputs, weights, loss_terms)
    if not is_finite_tensor(loss):
        print("Non-finite finite-difference loss")
        return None
    return float(loss.detach().cpu().item())


def finite_difference_one(scene, settings, weights, loss_terms, tensor_name, flat_index, eps):
    tensor = scene[tensor_name]
    flat = tensor.view(-1)
    original = float(flat[flat_index].detach().cpu().item())

    if tensor_name == "opacities":
        plus_value = min(max(original + eps, 0.01), 0.99)
        minus_value = min(max(original - eps, 0.01), 0.99)
    else:
        plus_value = original + eps
        minus_value = original - eps

    denom = plus_value - minus_value
    if denom == 0.0:
        return None

    try:
        with torch.no_grad():
            flat[flat_index] = plus_value
        loss_plus = forward_loss_value(scene, settings, weights, loss_terms)
        if loss_plus is None:
            return None

        with torch.no_grad():
            flat[flat_index] = minus_value
        loss_minus = forward_loss_value(scene, settings, weights, loss_terms)
        if loss_minus is None:
            return None

        return (loss_plus - loss_minus) / denom
    finally:
        with torch.no_grad():
            flat[flat_index] = original


def summarize_errors(errors):
    if not errors:
        return float("inf"), float("inf"), float("inf")
    arr = np.asarray(errors, dtype=np.float64)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 90.0)),
        float(np.max(arr)),
    )


def format_metric(value):
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0.0 else "inf"
    return "{:.3e}".format(value)


def ratio_fd_over_analytic(fd, analytic):
    if analytic == 0.0:
        if fd == 0.0:
            return 0.0
        return float("inf") if fd > 0.0 else float("-inf")
    return fd / analytic


def print_probe_rows(variant, tensor_name, probe_rows, count):
    if count <= 0:
        return

    print("")
    print("Probe diagnostics variant={} tensor={}".format(variant, tensor_name))
    if not probe_rows:
        print("  no finite probes")
        return

    header = "  {:<5} {:>11} {:>14} {:>14} {:>22} {:>12}".format(
        "rank",
        "flat_index",
        "analytic",
        "fd",
        "ratio_fd_over_analytic",
        "rel_err",
    )
    worst = sorted(probe_rows, key=lambda row: row["rel_err"], reverse=True)[:count]
    best = sorted(probe_rows, key=lambda row: row["rel_err"])[:count]

    for label, rows in (("worst", worst), ("best", best)):
        print("  {}".format(label))
        print(header)
        for rank, row in enumerate(rows, 1):
            print(
                "  {:<5d} {:>11d} {:>14} {:>14} {:>22} {:>12}".format(
                    rank,
                    row["flat_index"],
                    format_metric(row["analytic"]),
                    format_metric(row["fd"]),
                    format_metric(row["ratio"]),
                    format_metric(row["rel_err"]),
                )
            )


def print_fit_scale(variant, tensor_name, probe_rows):
    floor = 1.0e-12
    usable = [row for row in probe_rows if abs(row["analytic"]) > floor]
    if not usable:
        print(
            "Fit scale variant={} tensor={}: n=0 k_med=nan k_lsq=nan".format(
                variant,
                tensor_name,
            )
        )
        return

    analytic = np.asarray([row["analytic"] for row in usable], dtype=np.float64)
    fd = np.asarray([row["fd"] for row in usable], dtype=np.float64)
    ratios = fd / analytic
    denom = float(np.sum(analytic * analytic))
    k_lsq = float(np.sum(fd * analytic) / denom) if denom > 0.0 else float("nan")
    print(
        "Fit scale variant={} tensor={}: n={} k_med={} k_lsq={}".format(
            variant,
            tensor_name,
            len(usable),
            format_metric(float(np.median(ratios))),
            format_metric(k_lsq),
        )
    )


def print_table(rows):
    print("")
    print("Finite-difference relative errors")
    header = "{:<8} {:<10} {:>7} {:>9} {:>12} {:>12} {:>12} {:>6}".format(
        "variant",
        "tensor",
        "n_used",
        "skipped",
        "median",
        "p90",
        "max",
        "pass",
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "{:<8} {:<10} {:>7d} {:>9d} {:>12} {:>12} {:>12} {:>6}".format(
                row["variant"],
                row["tensor"],
                row["n_used"],
                row["n_skipped"],
                format_metric(row["median"]),
                format_metric(row["p90"]),
                format_metric(row["max"]),
                "yes" if row["passed"] else "no",
            )
        )


def run_variant(args, variant, variant_index, device):
    scene_seed = int(args.seed) + 1009 * variant_index
    fd_seed = int(args.seed) + 424242 + 1009 * variant_index
    weight_seed = int(args.seed) + 13579 + 1009 * variant_index

    scene = build_scene(
        scene_seed,
        int(args.n),
        variant,
        device,
        requires_grad=True,
        scale_lo=float(args.scale_lo),
        scale_hi=float(args.scale_hi),
    )
    settings = make_settings(args.height, args.width, device)

    outputs = render_scene(scene, settings, track_means2d=True)
    weights = make_loss_weights(outputs, weight_seed, device)
    finite_ok = check_outputs_finite(outputs, "{} analytic forward".format(variant))

    loss = scalar_loss(outputs, weights, args.loss_terms)
    if not is_finite_tensor(loss):
        print("Non-finite analytic loss in variant {}".format(variant))
        finite_ok = False

    if finite_ok:
        loss.backward()
        finite_ok = check_grads_finite(scene, "{} analytic backward".format(variant), args.tensors)

    rows = []
    if not finite_ok:
        for name in args.tensors:
            rows.append(
                {
                    "variant": variant,
                    "tensor": name,
                    "n_used": 0,
                    "n_skipped": 0,
                    "median": float("inf"),
                    "p90": float("inf"),
                    "max": float("inf"),
                    "passed": False,
                }
            )
        return rows, False

    visible = outputs[3].detach() > 0
    visible_count = int(visible.sum().detach().cpu().item())
    print(
        "Variant {}: {} / {} Gaussians visible in baseline forward".format(
            variant,
            visible_count,
            args.n,
        )
    )

    fd_rng = np.random.RandomState(fd_seed)
    variant_ok = True
    for name in args.tensors:
        candidates = visible_flat_indices(scene[name], visible)
        if candidates.size < max(1, int(args.probes) // 2):
            print(
                "Variant {}, tensor {}: only {} visible candidate elements; using available probes".format(
                    variant,
                    name,
                    candidates.size,
                )
            )

        probe_count = min(int(args.probes), int(candidates.size))
        if probe_count > 0:
            selected = fd_rng.choice(candidates, size=probe_count, replace=False)
        else:
            selected = np.asarray([], dtype=np.int64)

        errors = []
        probe_rows = []
        skipped = 0
        eps = EPS_BY_TENSOR[name] * float(args.eps_mult)
        analytic_flat = scene[name].grad.detach().view(-1)
        for flat_index in selected:
            flat_index = int(flat_index)
            fd = finite_difference_one(scene, settings, weights, args.loss_terms, name, flat_index, eps)
            if fd is None or not math.isfinite(fd):
                skipped += 1
                continue

            analytic = float(analytic_flat[flat_index].detach().cpu().item())
            scale = max(abs(analytic), abs(fd))
            if scale < 1.0e-6:
                skipped += 1
                continue

            rel_err = abs(analytic - fd) / scale
            errors.append(rel_err)
            probe_rows.append(
                {
                    "flat_index": flat_index,
                    "analytic": analytic,
                    "fd": float(fd),
                    "ratio": ratio_fd_over_analytic(float(fd), analytic),
                    "rel_err": rel_err,
                }
            )

        median, p90, max_err = summarize_errors(errors)
        passed = bool(errors) and median <= 1.0e-2 and p90 <= 5.0e-2
        variant_ok = variant_ok and passed
        rows.append(
            {
                "variant": variant,
                "tensor": name,
                "n_used": len(errors),
                "n_skipped": skipped,
                "median": median,
                "p90": p90,
                "max": max_err,
                "passed": passed,
            }
        )
        if args.fit_scale:
            print_fit_scale(variant, name, probe_rows)
        print_probe_rows(variant, name, probe_rows, int(args.dump_worst))

    return rows, variant_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--variants", type=str, default="generic,pole,seam")
    parser.add_argument("--loss-terms", type=str, default="color,depth,acc")
    parser.add_argument("--tensors", type=str, default=",".join(TENSOR_NAMES))
    parser.add_argument("--eps-mult", type=float, default=1.0)
    parser.add_argument("--scale-lo", type=float, default=0.010)
    parser.add_argument("--scale-hi", type=float, default=0.030)
    parser.add_argument("--dump-worst", type=int, default=6)
    parser.add_argument("--fit-scale", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if args.probes < 0:
        raise ValueError("--probes must be non-negative")
    if args.eps_mult <= 0.0:
        raise ValueError("--eps-mult must be positive")
    if args.scale_lo <= 0.0 or args.scale_hi <= 0.0:
        raise ValueError("--scale-lo and --scale-hi must be positive")
    if args.scale_lo > args.scale_hi:
        raise ValueError("--scale-lo must be <= --scale-hi")
    if args.dump_worst < 0:
        raise ValueError("--dump-worst must be non-negative")

    variants = parse_variants(args.variants)
    args.loss_terms = parse_name_subset(args.loss_terms, LOSS_TERM_NAMES, "loss-terms")
    args.tensors = parse_name_subset(args.tensors, TENSOR_NAMES, "tensors")
    if not torch.cuda.is_available():
        raise RuntimeError("The ODGS rasterizer requires CUDA.")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print("Active loss terms: {}".format(",".join(args.loss_terms)))
    print("Active tensors: {}".format(",".join(args.tensors)))
    print(
        "Scale range: [{:.6f}, {:.6f}], eps_mult: {:.6f}".format(
            float(args.scale_lo),
            float(args.scale_hi),
            float(args.eps_mult),
        )
    )

    device = torch.device("cuda")
    all_rows = []
    overall_ok = True

    for variant_index, variant in enumerate(variants):
        rows, ok = run_variant(args, variant, variant_index, device)
        all_rows.extend(rows)
        overall_ok = overall_ok and ok

    print_table(all_rows)

    if overall_ok:
        print("GRADCHECK PASS")
        sys.exit(0)

    print("GRADCHECK FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
