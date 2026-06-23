/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

__forceinline__ __device__ float clampf(float v, float lo, float hi)
{
	return fminf(hi, fmaxf(lo, v));
}

__forceinline__ __device__ bool isFinite(float v)
{
	return isfinite(v);
}

__forceinline__ __device__ bool isFinite(const float2 v)
{
	return isFinite(v.x) && isFinite(v.y);
}

__forceinline__ __device__ bool isFinite(const float3 v)
{
	return isFinite(v.x) && isFinite(v.y) && isFinite(v.z);
}

__forceinline__ __device__ float dot3(const float3 a, const float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 add3(const float3 a, const float3 b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__forceinline__ __device__ float3 sub3(const float3 a, const float3 b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__forceinline__ __device__ float3 mul3(const float3 a, const float s)
{
	return { a.x * s, a.y * s, a.z * s };
}

__forceinline__ __device__ float3 cross3(const float3 a, const float3 b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// ---------------------------------------------------------------------------
// Tangent-native spherical log-map.
//
// The Gaussian is evaluated directly in the tangent plane of its mean ray
// u0 on the unit sphere, in "equator pixel" units: one unit equals one ERP
// pixel of geodesic distance at the equator. The pixel delta of a pixel ray
// u is
//
//     d = alpha(q) * ( dot(u, ex), dot(u, ey) ),
//     q = dot(u, u0) = cos(gamma),  alpha = gamma / sin(gamma),
//
// where ex = (W/2pi) e_phi and ey = -(H/pi) e_theta are the pixel-scaled
// tangent basis vectors at u0. This uses dot(log_{u0}(u), e) =
// alpha * dot(u, e) for any e orthogonal to u0, so no residual vector or
// explicit (lambda, theta) coordinates are ever formed. |d| is the exact
// geodesic distance (scaled); there is no ERP distortion, no seam, and no
// pole singularity, hence no fallback path.
// ---------------------------------------------------------------------------

// Per-Gaussian tangent frame, built once in preprocess (without any
// trigonometric call) and consumed as three coalesced float4 loads by the
// render kernels.
struct OmniTangentFrame
{
	float4 u0_st;  // xyz: u0 (unit mean ray, camera space), w: sin(theta0)
	float4 ex_ct;  // xyz: ex = (W/2pi) * e_phi,             w: cos(theta0)
	float4 ey_d;   // xyz: ey = -(H/pi) * e_theta,           w: depth |t|
};

// Geometry of the tangent frame at camera-space position t. Shared between
// the forward preprocessing and the covariance backward so both passes see
// bit-identical values.
struct OmniTangentGeom
{
	float r;          // |t| (epsilon-guarded)
	float r_xz;       // sqrt(tx^2 + tz^2) (epsilon-guarded)
	float sin_theta0; // -ty / r
	float cos_theta0; // r_xz_raw / r  (>= 0)
	float3 u0;        // t / r
	float3 e_phi;     // unit azimuth tangent:   (cos(phi0), 0, -sin(phi0))
	float3 e_theta;   // unit elevation tangent: e_phi x u0
};

__forceinline__ __device__ OmniTangentGeom makeOmniTangentGeom(const float3 t)
{
	const float e = 0.0000001f;
	OmniTangentGeom g;
	const float r_xz_raw = sqrtf(t.x * t.x + t.z * t.z);
	g.r = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z) + e;
	g.r_xz = r_xz_raw + e;
	const float inv_r = 1.0f / g.r;
	g.u0 = { t.x * inv_r, t.y * inv_r, t.z * inv_r };
	g.sin_theta0 = -t.y * inv_r;
	g.cos_theta0 = r_xz_raw * inv_r;
	if (r_xz_raw > 1.0e-6f)
	{
		const float inv_r_xz = 1.0f / r_xz_raw;
		g.e_phi = { t.z * inv_r_xz, 0.0f, -t.x * inv_r_xz };
	}
	else
	{
		// Exactly at a pole the azimuth is arbitrary; any tangent direction
		// gives a valid frame as long as the covariance is expressed in the
		// same frame (it is, see computeTangentCov2D).
		g.e_phi = { 1.0f, 0.0f, 0.0f };
	}
	g.e_theta = cross3(g.e_phi, g.u0);
	return g;
}

__forceinline__ __device__ OmniTangentFrame makeOmniTangentFrame(
	const OmniTangentGeom& g,
	const int W,
	const int H,
	const float depth)
{
	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float kx = (float)W / two_pi;
	const float ky = (float)H / pi;
	OmniTangentFrame f;
	f.u0_st = { g.u0.x, g.u0.y, g.u0.z, g.sin_theta0 };
	f.ex_ct = { kx * g.e_phi.x, kx * g.e_phi.y, kx * g.e_phi.z, g.cos_theta0 };
	f.ey_d = { -ky * g.e_theta.x, -ky * g.e_theta.y, -ky * g.e_theta.z, depth };
	return f;
}

// Unit ray direction of an ERP pixel. The only per-pixel trigonometry of the
// whole rasterizer; each render thread calls this once per kernel launch.
__forceinline__ __device__ float3 pixelRayDirection(const float2 pix, const int W, const int H)
{
	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float lambda = (pix.x / (float)W - 0.5f) * two_pi;
	const float theta = (0.5f - pix.y / (float)H) * pi;
	float sin_lambda, cos_lambda, sin_theta, cos_theta;
	sincosf(lambda, &sin_lambda, &cos_lambda);
	sincosf(theta, &sin_theta, &cos_theta);
	return { cos_theta * sin_lambda, -sin_theta, cos_theta * cos_lambda };
}

// Treatment of q = dot(u, u0) before evaluating the log-map scale:
//  - contributions with q <= OMNI_Q_CUTOFF (gamma > ~177.4 deg) are skipped
//    by the render kernels: the log-map is ill-defined towards the antipode
//    (at q = -1 the tangent projection vanishes, which would turn into a
//    spurious full-strength splat). Testing with !(q > OMNI_Q_CUTOFF) also
//    filters NaNs.
//  - the upper clamp OMNI_Q_MAX keeps (1 - q)(1 + q) away from zero; the
//    alpha error it introduces is below 1e-7 (gamma < 4.5e-4 rad there).
#define OMNI_Q_CUTOFF (-0.999f)
#define OMNI_Q_MAX (0.9999999f)

// alpha^2(q) = gamma^2 / sin^2(gamma). The factored form (1 - q)(1 + q) for
// sin^2(gamma) is cancellation-free near q = 1, unlike 1 - q*q.
__forceinline__ __device__ float omniLogMapAlpha2(const float q)
{
	const float g = acosf(q);
	const float s2 = (1.0f - q) * (1.0f + q);
	return g * g / s2;
}

// alpha(q) and d(alpha)/dq for the backward pass.
// d(alpha)/dq = (gamma*q - sin(gamma)) / sin^3(gamma); the numerator loses
// all significant digits for small gamma, so switch to its Taylor expansion
// -1/3 - 2*gamma^2/15 there.
__forceinline__ __device__ void omniLogMapAlphaPair(const float q, float& alpha, float& dalpha_dq)
{
	const float g = acosf(q);
	const float s2 = (1.0f - q) * (1.0f + q);
	const float inv_s = rsqrtf(s2);
	alpha = g * inv_s;
	if (g < 0.03f)
	{
		dalpha_dq = -(1.0f / 3.0f) - (2.0f / 15.0f) * g * g;
	}
	else
	{
		const float sin_g = s2 * inv_s;
		dalpha_dq = (g * q - sin_g) * inv_s * inv_s * inv_s;
	}
}

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

struct OmniTileBounds
{
	uint2 rect_min0;
	uint2 rect_max0;
	uint2 rect_min1;
	uint2 rect_max1;
	int rect_count;
};

__forceinline__ __device__ void clearOmniTileBounds(OmniTileBounds& bounds)
{
	bounds.rect_min0 = { 0, 0 };
	bounds.rect_max0 = { 0, 0 };
	bounds.rect_min1 = { 0, 0 };
	bounds.rect_max1 = { 0, 0 };
	bounds.rect_count = 0;
}

__forceinline__ __device__ void getRectFromPixelBounds(
	const float x_min,
	const float x_max,
	const float y_min,
	const float y_max,
	uint2& rect_min,
	uint2& rect_max,
	dim3 grid)
{
	rect_min = { 0, 0 };
	rect_max = { 0, 0 };
	if (!isFinite(x_min) || !isFinite(x_max) || !isFinite(y_min) || !isFinite(y_max) ||
		x_max < x_min || y_max < y_min)
		return;

	const int min_px_x = max(0, (int)floorf(x_min));
	const int min_px_y = max(0, (int)floorf(y_min));
	const int max_px_x = max(0, (int)ceilf(x_max));
	const int max_px_y = max(0, (int)ceilf(y_max));

	rect_min = {
		min(grid.x, (uint32_t)(min_px_x / BLOCK_X)),
		min(grid.y, (uint32_t)(min_px_y / BLOCK_Y))
	};
	rect_max = {
		min(grid.x, (uint32_t)(max_px_x / BLOCK_X + 1)),
		min(grid.y, (uint32_t)(max_px_y / BLOCK_Y + 1))
	};
}

// Tile candidates of a Gaussian: the spherical cap of geodesic radius
// gamma_max around the mean direction, mapped to its ERP bounding box.
// Splits into two rectangles when the box crosses the azimuth seam; expands
// to the full row band when the cap touches a pole.
__forceinline__ __device__ OmniTileBounds getOmniCapTileBounds(
	const float center_px_x,
	const float theta0,
	const float cos_theta0,
	const float gamma_max,
	const int W,
	const int H,
	dim3 grid)
{
	OmniTileBounds bounds;
	clearOmniTileBounds(bounds);
	if (W <= 0 || H <= 0 || !isFinite(center_px_x) || !isFinite(theta0) ||
		!isFinite(gamma_max) || gamma_max <= 0.0f)
		return bounds;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;

	float center_x = fmodf(center_px_x, Wf);
	if (center_x < 0.0f)
		center_x += Wf;

	// Elevation band of the cap, mapped to pixel rows.
	const float theta_min = clampf(theta0 - gamma_max, -0.5f * pi, 0.5f * pi);
	const float theta_max = clampf(theta0 + gamma_max, -0.5f * pi, 0.5f * pi);
	const float y_min = (0.5f - theta_max / pi) * Hf;
	const float y_max = (0.5f - theta_min / pi) * Hf;

	const bool reaches_pole = fabsf(theta0) + gamma_max >= 0.5f * pi;
	float lon_delta = pi;
	if (!reaches_pole)
	{
		const float denom = fmaxf(fabsf(cos_theta0), 1.0e-6f);
		lon_delta = asinf(clampf(sinf(gamma_max) / denom, -1.0f, 1.0f));
	}

	const float x_delta = lon_delta * Wf / two_pi;
	if (!isFinite(lon_delta) || reaches_pole || x_delta >= 0.5f * Wf)
	{
		getRectFromPixelBounds(0.0f, Wf - 1.0f, y_min, y_max, bounds.rect_min0, bounds.rect_max0, grid);
		bounds.rect_count = ((bounds.rect_max0.x - bounds.rect_min0.x) * (bounds.rect_max0.y - bounds.rect_min0.y)) > 0 ? 1 : 0;
		return bounds;
	}

	const float x_min = center_x - x_delta;
	const float x_max = center_x + x_delta;
	if (x_min < 0.0f)
	{
		getRectFromPixelBounds(x_min + Wf, Wf - 1.0f, y_min, y_max, bounds.rect_min0, bounds.rect_max0, grid);
		getRectFromPixelBounds(0.0f, x_max, y_min, y_max, bounds.rect_min1, bounds.rect_max1, grid);
		bounds.rect_count = 2;
	}
	else if (x_max >= Wf)
	{
		getRectFromPixelBounds(x_min, Wf - 1.0f, y_min, y_max, bounds.rect_min0, bounds.rect_max0, grid);
		getRectFromPixelBounds(0.0f, x_max - Wf, y_min, y_max, bounds.rect_min1, bounds.rect_max1, grid);
		bounds.rect_count = 2;
	}
	else
	{
		getRectFromPixelBounds(x_min, x_max, y_min, y_max, bounds.rect_min0, bounds.rect_max0, grid);
		bounds.rect_count = ((bounds.rect_max0.x - bounds.rect_min0.x) * (bounds.rect_max0.y - bounds.rect_min0.y)) > 0 ? 1 : 0;
	}

	return bounds;
}

__forceinline__ __device__ uint32_t getOmniTileBoundsTileCount(const OmniTileBounds bounds)
{
	uint32_t count = 0;
	if (bounds.rect_count > 0)
		count += (bounds.rect_max0.y - bounds.rect_min0.y) * (bounds.rect_max0.x - bounds.rect_min0.x);
	if (bounds.rect_count > 1)
		count += (bounds.rect_max1.y - bounds.rect_min1.y) * (bounds.rect_max1.x - bounds.rect_min1.x);
	return count;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_sphere(int idx,
	const float* orig_points,
	const float* viewmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	// Bring points to screen space
	float dist = sqrt(p_view.x*p_view.x + p_view.y*p_view.y + p_view.z*p_view.z)+0.0000001f;

	if (dist <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
