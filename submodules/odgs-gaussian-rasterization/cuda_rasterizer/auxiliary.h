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

struct OmniLogMapDeltaResult
{
	float2 d;
	float ddx_dmx;
	float ddx_dmy;
	float ddy_dmx;
	float ddy_dmy;
	bool fallback;
};

struct OmniLogMapBaseResult
{
	float2 d;
	float lambda0;
	float theta0;
	float sin_theta0;
	float cos_theta0;
	float sin_lambda0;
	float cos_lambda0;
	float q;
	float gamma;
	float sin_gamma;
	float alpha;
	float3 u;
	float3 u0;
	float3 w;
	float3 v;
	float3 e_lambda;
	float3 e_theta;
	bool fallback;
};

struct OmniLogMapPixelContext
{
	float2 pix;
	float3 u;
	float sin_lambda;
	float cos_lambda;
	float sin_theta;
	float cos_theta;
	bool valid;
};

struct OmniLogMapMeanContext
{
	float2 mean;
	float lambda0;
	float theta0;
	float sin_theta0;
	float cos_theta0;
	float sin_lambda0;
	float cos_lambda0;
	float3 u0;
	float3 e_lambda;
	float3 e_theta;
	bool valid;
};

// Per-(tile, Gaussian) log-map data. The log-map is evaluated once per tile at
// the tile anchor pixel; each pixel then reconstructs its own delta with a
// first-order expansion, instead of paying a full log-map per pixel.
struct OmniLogMapTileResult
{
	float2 d;        // log-map delta at the tile anchor pixel
	float ddx_dpx;   // d(delta) / d(pixel): in-tile first-order reconstruction
	float ddx_dpy;
	float ddy_dpx;
	float ddy_dpy;
	float ddx_dmx;   // d(delta) / d(mean pixel): consumed by the backward pass
	float ddx_dmy;
	float ddy_dmx;
	float ddy_dmy;
	bool fallback;
};

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

__forceinline__ __device__ float omniPeriodicPixelDeltaX(const float pix_x, const float mean_x, const int W)
{
	float dx = pix_x - mean_x;
	if (W > 0 && isFinite(dx))
	{
		const float Wf = (float)W;
		dx = fmodf(dx + 0.5f * Wf, Wf);
		if (dx < 0.0f)
			dx += Wf;
		dx -= 0.5f * Wf;
	}
	return dx;
}

__forceinline__ __device__ float2 omniPixelDeltaFallback(const float2 pix, const float2 mean, const int W)
{
	return { omniPeriodicPixelDeltaX(pix.x, mean.x, W), pix.y - mean.y };
}

__forceinline__ __device__ OmniLogMapBaseResult makeOmniLogMapBaseFallback(
	const float2 pix,
	const float2 mean,
	const int W)
{
	OmniLogMapBaseResult result;
	result.d = omniPixelDeltaFallback(pix, mean, W);
	result.lambda0 = 0.0f;
	result.theta0 = 0.0f;
	result.sin_theta0 = 0.0f;
	result.cos_theta0 = 0.0f;
	result.sin_lambda0 = 0.0f;
	result.cos_lambda0 = 0.0f;
	result.q = 0.0f;
	result.gamma = 0.0f;
	result.sin_gamma = 0.0f;
	result.alpha = 1.0f;
	result.u = { 0.0f, 0.0f, 1.0f };
	result.u0 = { 0.0f, 0.0f, 1.0f };
	result.w = { 0.0f, 0.0f, 0.0f };
	result.v = { 0.0f, 0.0f, 0.0f };
	result.e_lambda = { 1.0f, 0.0f, 0.0f };
	result.e_theta = { 0.0f, -1.0f, 0.0f };
	result.fallback = true;
	return result;
}

__forceinline__ __device__ OmniLogMapPixelContext makeOmniLogMapPixelContext(
	const float2 pix,
	const int W,
	const int H)
{
	OmniLogMapPixelContext ctx;
	ctx.pix = pix;
	ctx.u = { 0.0f, 0.0f, 1.0f };
	ctx.sin_lambda = 0.0f;
	ctx.cos_lambda = 0.0f;
	ctx.sin_theta = 0.0f;
	ctx.cos_theta = 0.0f;
	ctx.valid = false;

	if (W <= 0 || H <= 0 || !isFinite(pix))
		return ctx;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;
	const float lambda = (pix.x / Wf - 0.5f) * two_pi;
	const float theta = (0.5f - pix.y / Hf) * pi;
	float sin_lambda, cos_lambda, sin_theta, cos_theta;
	sincosf(lambda, &sin_lambda, &cos_lambda);
	sincosf(theta, &sin_theta, &cos_theta);

	if (!isFinite(lambda) || !isFinite(theta) ||
		!isFinite(sin_lambda) || !isFinite(cos_lambda) ||
		!isFinite(sin_theta) || !isFinite(cos_theta))
		return ctx;

	ctx.u = {
		cos_theta * sin_lambda,
		-sin_theta,
		cos_theta * cos_lambda
	};
	ctx.sin_lambda = sin_lambda;
	ctx.cos_lambda = cos_lambda;
	ctx.sin_theta = sin_theta;
	ctx.cos_theta = cos_theta;
	ctx.valid = true;
	return ctx;
}

__forceinline__ __device__ OmniLogMapMeanContext makeOmniLogMapMeanContext(
	const float2 mean,
	const int W,
	const int H)
{
	OmniLogMapMeanContext ctx;
	ctx.mean = mean;
	ctx.lambda0 = 0.0f;
	ctx.theta0 = 0.0f;
	ctx.sin_theta0 = 0.0f;
	ctx.cos_theta0 = 0.0f;
	ctx.sin_lambda0 = 0.0f;
	ctx.cos_lambda0 = 0.0f;
	ctx.u0 = { 0.0f, 0.0f, 0.0f };
	ctx.e_lambda = { 0.0f, 0.0f, 0.0f };
	ctx.e_theta = { 0.0f, 0.0f, 0.0f };
	ctx.valid = false;

	if (W <= 0 || H <= 0 || !isFinite(mean))
		return ctx;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;
	const float lambda0 = (mean.x / Wf - 0.5f) * two_pi;
	const float theta0 = (0.5f - mean.y / Hf) * pi;
	float sin_lambda0, cos_lambda0, sin_theta0, cos_theta0;
	sincosf(lambda0, &sin_lambda0, &cos_lambda0);
	sincosf(theta0, &sin_theta0, &cos_theta0);

	if (!isFinite(lambda0) || !isFinite(theta0) ||
		!isFinite(sin_lambda0) || !isFinite(cos_lambda0) ||
		!isFinite(sin_theta0) || !isFinite(cos_theta0) ||
		fabsf(cos_theta0) < 1.0e-4f)
		return ctx;

	ctx.lambda0 = lambda0;
	ctx.theta0 = theta0;
	ctx.sin_theta0 = sin_theta0;
	ctx.cos_theta0 = cos_theta0;
	ctx.sin_lambda0 = sin_lambda0;
	ctx.cos_lambda0 = cos_lambda0;
	ctx.u0 = {
		cos_theta0 * sin_lambda0,
		-sin_theta0,
		cos_theta0 * cos_lambda0
	};
	ctx.e_lambda = {
		cos_lambda0,
		0.0f,
		-sin_lambda0
	};
	ctx.e_theta = {
		-sin_theta0 * sin_lambda0,
		-cos_theta0,
		-sin_theta0 * cos_lambda0
	};
	ctx.valid = true;
	return ctx;
}

__forceinline__ __device__ float2 logMapOmniDeltaPixel(
	const OmniLogMapPixelContext& pix_ctx,
	const OmniLogMapMeanContext& mean_ctx,
	const int W,
	const int H)
{
	const float2 fallback = omniPixelDeltaFallback(pix_ctx.pix, mean_ctx.mean, W);
	if (W <= 0 || H <= 0 || !pix_ctx.valid || !mean_ctx.valid)
		return fallback;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;

	const float3 u0 = mean_ctx.u0;
	const float3 e_lambda = mean_ctx.e_lambda;
	const float3 e_theta = mean_ctx.e_theta;

	const float q = dot3(pix_ctx.u, u0);
	if (!isFinite(q))
		return fallback;

	const float q_clamped = clampf(q, -1.0f, 1.0f);
	const float gamma = acosf(q_clamped);
	const float sin_gamma = sinf(gamma);
	if (!isFinite(gamma) || !isFinite(sin_gamma) ||
		gamma < 1.0e-6f || fabsf(sin_gamma) < 1.0e-6f)
		return fallback;

	const float gamma2 = gamma * gamma;
	const float alpha = (gamma < 1.0e-4f) ? (1.0f + gamma2 / 6.0f) : (gamma / sin_gamma);
	const float3 w = sub3(pix_ctx.u, mul3(u0, q_clamped));
	const float3 v = mul3(w, alpha);
	const float xi = dot3(v, e_lambda);
	const float eta = dot3(v, e_theta);
	const float2 d = {
		(xi / mean_ctx.cos_theta0) * Wf / two_pi,
		-eta * Hf / pi
	};

	if (!isFinite(alpha) || !isFinite(w) || !isFinite(v) || !isFinite(d))
		return fallback;
	return d;
}

__forceinline__ __device__ OmniLogMapBaseResult computeOmniLogMapBase(
	const OmniLogMapPixelContext& pix_ctx,
	const OmniLogMapMeanContext& mean_ctx,
	const int W,
	const int H)
{
	OmniLogMapBaseResult result = makeOmniLogMapBaseFallback(pix_ctx.pix, mean_ctx.mean, W);
	if (W <= 0 || H <= 0 || !pix_ctx.valid || !mean_ctx.valid)
		return result;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;

	const float3 u0 = mean_ctx.u0;
	const float3 e_lambda = mean_ctx.e_lambda;
	const float3 e_theta = mean_ctx.e_theta;

	const float q = dot3(pix_ctx.u, u0);
	if (!isFinite(q))
		return result;

	const float q_clamped = clampf(q, -1.0f, 1.0f);
	const float gamma = acosf(q_clamped);
	const float sin_gamma = sinf(gamma);

	if (!isFinite(gamma) || !isFinite(sin_gamma) ||
		gamma < 1.0e-6f || fabsf(sin_gamma) < 1.0e-6f)
		return result;

	const float gamma2 = gamma * gamma;
	const float alpha = (gamma < 1.0e-4f) ? (1.0f + gamma2 / 6.0f) : (gamma / sin_gamma);
	const float3 w = sub3(pix_ctx.u, mul3(u0, q_clamped));
	const float3 v = mul3(w, alpha);
	const float xi = dot3(v, e_lambda);
	const float eta = dot3(v, e_theta);
	const float2 d = {
		(xi / mean_ctx.cos_theta0) * Wf / two_pi,
		-eta * Hf / pi
	};

	if (!isFinite(alpha) || !isFinite(w) || !isFinite(v) || !isFinite(d))
		return result;

	result.d = d;
	result.lambda0 = mean_ctx.lambda0;
	result.theta0 = mean_ctx.theta0;
	result.sin_theta0 = mean_ctx.sin_theta0;
	result.cos_theta0 = mean_ctx.cos_theta0;
	result.sin_lambda0 = mean_ctx.sin_lambda0;
	result.cos_lambda0 = mean_ctx.cos_lambda0;
	result.q = q_clamped;
	result.gamma = gamma;
	result.sin_gamma = sin_gamma;
	result.alpha = alpha;
	result.u = pix_ctx.u;
	result.u0 = u0;
	result.w = w;
	result.v = v;
	result.e_lambda = e_lambda;
	result.e_theta = e_theta;
	result.fallback = false;
	return result;
}

__forceinline__ __device__ OmniLogMapBaseResult computeOmniLogMapBase(
	const float2 pix,
	const float2 mean,
	const int W,
	const int H)
{
	const OmniLogMapPixelContext pix_ctx = makeOmniLogMapPixelContext(pix, W, H);
	const OmniLogMapMeanContext mean_ctx = makeOmniLogMapMeanContext(mean, W, H);
	return computeOmniLogMapBase(pix_ctx, mean_ctx, W, H);
}

__forceinline__ __device__ float2 logMapOmniDeltaPixel(
	const float2 pix,
	const float2 mean,
	const int W,
	const int H)
{
	const OmniLogMapPixelContext pix_ctx = makeOmniLogMapPixelContext(pix, W, H);
	const OmniLogMapMeanContext mean_ctx = makeOmniLogMapMeanContext(mean, W, H);
	return logMapOmniDeltaPixel(pix_ctx, mean_ctx, W, H);
}

__forceinline__ __device__ OmniLogMapDeltaResult logMapOmniDeltaPixelWithMeanJacobian(
	const OmniLogMapBaseResult& base,
	const int W,
	const int H)
{
	OmniLogMapDeltaResult result;
	result.d = base.d;
	result.ddx_dmx = -1.0f;
	result.ddx_dmy = 0.0f;
	result.ddy_dmx = 0.0f;
	result.ddy_dmy = -1.0f;
	result.fallback = true;

	if (base.fallback)
		return result;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;
	const float pixel_x_scale = Wf / two_pi;
	const float pixel_y_scale = Hf / pi;

	const float sin_theta0 = base.sin_theta0;
	const float cos_theta0 = base.cos_theta0;
	const float sin_lambda0 = base.sin_lambda0;
	const float cos_lambda0 = base.cos_lambda0;
	const float q = base.q;
	const float gamma = base.gamma;
	const float sin_gamma = base.sin_gamma;
	// cos(gamma) == cos(acos(q)) == q, so reuse base.q instead of a cosf call.
	const float cos_gamma = q;
	const float alpha = base.alpha;
	const float xi = dot3(base.v, base.e_lambda);

	const float dalpha_dgamma = (gamma < 1.0e-4f) ?
		(gamma / 3.0f) :
		((sin_gamma - gamma * cos_gamma) / (sin_gamma * sin_gamma));

	const float inv_sin_gamma = 1.0f / sin_gamma;
	const float inv_cos_theta0 = 1.0f / cos_theta0;
	const float inv_cos_theta0_2 = inv_cos_theta0 * inv_cos_theta0;

	const float3 du0_dlambda0 = mul3(base.e_lambda, cos_theta0);
	const float3 du0_dtheta0 = base.e_theta;
	const float3 de_lambda_dlambda0 = {
		-sin_lambda0,
		0.0f,
		-cos_lambda0
	};
	const float3 de_lambda_dtheta0 = { 0.0f, 0.0f, 0.0f };
	const float3 de_theta_dlambda0 = {
		-sin_theta0 * cos_lambda0,
		0.0f,
		sin_theta0 * sin_lambda0
	};
	const float3 de_theta_dtheta0 = {
		-cos_theta0 * sin_lambda0,
		sin_theta0,
		-cos_theta0 * cos_lambda0
	};

	const float dq_dlambda0 = dot3(base.u, du0_dlambda0);
	const float dgamma_dlambda0 = -dq_dlambda0 * inv_sin_gamma;
	const float dalpha_dlambda0 = dalpha_dgamma * dgamma_dlambda0;
	const float3 dw_dlambda0 = sub3(mul3(base.u0, -dq_dlambda0), mul3(du0_dlambda0, q));
	const float3 dv_dlambda0 = add3(mul3(base.w, dalpha_dlambda0), mul3(dw_dlambda0, alpha));
	const float dxi_dlambda0 = dot3(dv_dlambda0, base.e_lambda) + dot3(base.v, de_lambda_dlambda0);
	const float deta_dlambda0 = dot3(dv_dlambda0, base.e_theta) + dot3(base.v, de_theta_dlambda0);

	const float dq_dtheta0 = dot3(base.u, du0_dtheta0);
	const float dgamma_dtheta0 = -dq_dtheta0 * inv_sin_gamma;
	const float dalpha_dtheta0 = dalpha_dgamma * dgamma_dtheta0;
	const float3 dw_dtheta0 = sub3(mul3(base.u0, -dq_dtheta0), mul3(du0_dtheta0, q));
	const float3 dv_dtheta0 = add3(mul3(base.w, dalpha_dtheta0), mul3(dw_dtheta0, alpha));
	const float dxi_dtheta0 = dot3(dv_dtheta0, base.e_lambda) + dot3(base.v, de_lambda_dtheta0);
	const float deta_dtheta0 = dot3(dv_dtheta0, base.e_theta) + dot3(base.v, de_theta_dtheta0);

	const float ddx_dlambda0 = pixel_x_scale * dxi_dlambda0 * inv_cos_theta0;
	const float ddy_dlambda0 = -pixel_y_scale * deta_dlambda0;
	const float ddx_dtheta0 = pixel_x_scale *
		(dxi_dtheta0 * inv_cos_theta0 - xi * (-sin_theta0) * inv_cos_theta0_2);
	const float ddy_dtheta0 = -pixel_y_scale * deta_dtheta0;

	result.ddx_dmx = ddx_dlambda0 * two_pi / Wf;
	result.ddy_dmx = ddy_dlambda0 * two_pi / Wf;
	result.ddx_dmy = ddx_dtheta0 * (-pi / Hf);
	result.ddy_dmy = ddy_dtheta0 * (-pi / Hf);

	result.fallback = false;
	return result;
}

__forceinline__ __device__ OmniLogMapDeltaResult logMapOmniDeltaPixelWithMeanJacobian(
	const OmniLogMapPixelContext& pix_ctx,
	const OmniLogMapMeanContext& mean_ctx,
	const int W,
	const int H)
{
	const OmniLogMapBaseResult base = computeOmniLogMapBase(pix_ctx, mean_ctx, W, H);
	return logMapOmniDeltaPixelWithMeanJacobian(base, W, H);
}

__forceinline__ __device__ OmniLogMapDeltaResult logMapOmniDeltaPixelWithMeanJacobian(
	const float2 pix,
	const float2 mean,
	const int W,
	const int H)
{
	const OmniLogMapBaseResult base = computeOmniLogMapBase(pix, mean, W, H);
	return logMapOmniDeltaPixelWithMeanJacobian(base, W, H);
}

// Evaluate the log-map once at the tile anchor pixel. Returns the anchor delta,
// the in-tile pixel Jacobian (for first-order per-pixel reconstruction) and the
// mean Jacobian (for the backward pass). Computing this once per (tile,
// Gaussian) instead of once per (pixel, Gaussian) amortises the transcendental
// work over the whole 16x16 tile.
__forceinline__ __device__ OmniLogMapTileResult computeOmniLogMapTile(
	const OmniLogMapPixelContext& anchor_ctx,
	const OmniLogMapMeanContext& mean_ctx,
	const int W,
	const int H)
{
	const OmniLogMapBaseResult base = computeOmniLogMapBase(anchor_ctx, mean_ctx, W, H);
	const OmniLogMapDeltaResult mean_jac = logMapOmniDeltaPixelWithMeanJacobian(base, W, H);

	OmniLogMapTileResult result;
	result.d = base.d;
	result.ddx_dmx = mean_jac.ddx_dmx;
	result.ddx_dmy = mean_jac.ddx_dmy;
	result.ddy_dmx = mean_jac.ddy_dmx;
	result.ddy_dmy = mean_jac.ddy_dmy;
	// Fallback delta is the (periodic) pixel-minus-mean offset, which is already
	// exactly linear in the pixel coordinate -> identity in-tile Jacobian.
	result.ddx_dpx = 1.0f;
	result.ddx_dpy = 0.0f;
	result.ddy_dpx = 0.0f;
	result.ddy_dpy = 1.0f;
	result.fallback = base.fallback;

	if (result.fallback)
		return result;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;
	const float pixel_x_scale = Wf / two_pi;
	const float pixel_y_scale = Hf / pi;

	// d(delta)/d(u), where u is the pixel ray direction. The mean tangent frame
	// (e_lambda, e_theta, cos_theta0) does not depend on the pixel, so this is
	// simpler than the mean Jacobian (no frame-derivative terms).
	const float cos_gamma = base.q;  // cos(acos(q)) == q
	const float dalpha_dgamma = (base.gamma < 1.0e-4f) ?
		(base.gamma / 3.0f) :
		((base.sin_gamma - base.gamma * cos_gamma) / (base.sin_gamma * base.sin_gamma));
	const float c_alpha = -dalpha_dgamma / base.sin_gamma;

	// e_lambda . u0 == e_theta . u0 == 0, so d(xi)/d(u) and d(eta)/d(u) reduce:
	const float el_dot_w = dot3(base.e_lambda, base.w);
	const float et_dot_w = dot3(base.e_theta, base.w);
	const float3 dxi_du = add3(mul3(base.e_lambda, base.alpha), mul3(base.u0, c_alpha * el_dot_w));
	const float3 deta_du = add3(mul3(base.e_theta, base.alpha), mul3(base.u0, c_alpha * et_dot_w));

	const float inv_cos_theta0 = 1.0f / base.cos_theta0;
	const float3 ddx_du = mul3(dxi_du, pixel_x_scale * inv_cos_theta0);
	const float3 ddy_du = mul3(deta_du, -pixel_y_scale);

	// d(u)/d(pixel) at the anchor pixel.
	const float ct = anchor_ctx.cos_theta;
	const float st = anchor_ctx.sin_theta;
	const float cl = anchor_ctx.cos_lambda;
	const float sl = anchor_ctx.sin_lambda;
	const float3 du_dlambda = { ct * cl, 0.0f, -ct * sl };
	const float3 du_dtheta = { -st * sl, -ct, -st * cl };
	const float3 du_dpx = mul3(du_dlambda, two_pi / Wf);
	const float3 du_dpy = mul3(du_dtheta, -pi / Hf);

	const float ddx_dpx = dot3(ddx_du, du_dpx);
	const float ddx_dpy = dot3(ddx_du, du_dpy);
	const float ddy_dpx = dot3(ddy_du, du_dpx);
	const float ddy_dpy = dot3(ddy_du, du_dpy);

	// Keep the identity in-tile Jacobian if anything went non-finite.
	if (isFinite(ddx_dpx) && isFinite(ddx_dpy) && isFinite(ddy_dpx) && isFinite(ddy_dpy))
	{
		result.ddx_dpx = ddx_dpx;
		result.ddx_dpy = ddx_dpy;
		result.ddy_dpx = ddy_dpx;
		result.ddy_dpy = ddy_dpy;
	}
	return result;
}

// First-order reconstruction of the per-pixel log-map delta within a tile,
// from the anchor delta and the in-tile pixel Jacobian.
__forceinline__ __device__ float2 applyOmniLogMapTile(
	const OmniLogMapTileResult& lm,
	const float2 pix,
	const float2 anchor)
{
	const float dpx = pix.x - anchor.x;
	const float dpy = pix.y - anchor.y;
	return {
		lm.d.x + lm.ddx_dpx * dpx + lm.ddx_dpy * dpy,
		lm.d.y + lm.ddy_dpx * dpx + lm.ddy_dpy * dpy
	};
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

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
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

__forceinline__ __device__ OmniTileBounds getOmniLogMapTileBounds(
	const float2 p,
	const int max_radius,
	const int W,
	const int H,
	dim3 grid)
{
	OmniTileBounds bounds;
	clearOmniTileBounds(bounds);
	if (max_radius <= 0 || W <= 0 || H <= 0 || !isFinite(p))
		return bounds;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;

	float center_x = fmodf(p.x, Wf);
	if (center_x < 0.0f)
		center_x += Wf;
	const float theta0 = (0.5f - p.y / Hf) * pi;
	const float cos_theta0 = cosf(theta0);

	// The render kernel evaluates the Gaussian in log-map tangent coordinates.
	// Use a spherical cap that conservatively contains the tangent-pixel radius.
	const float tangent_x_scale = two_pi / Wf * fabsf(cos_theta0);
	const float tangent_y_scale = pi / Hf;
	const float gamma_max = (float)max_radius * fmaxf(tangent_x_scale, tangent_y_scale);

	if (!isFinite(center_x) || !isFinite(theta0) || !isFinite(cos_theta0) || !isFinite(gamma_max))
	{
		getRectFromPixelBounds(0.0f, Wf - 1.0f, p.y - max_radius, p.y + max_radius, bounds.rect_min0, bounds.rect_max0, grid);
		bounds.rect_count = ((bounds.rect_max0.x - bounds.rect_min0.x) * (bounds.rect_max0.y - bounds.rect_min0.y)) > 0 ? 1 : 0;
		return bounds;
	}

	if (fabsf(cos_theta0) < 1.0e-4f)
	{
		getRectFromPixelBounds(0.0f, Wf - 1.0f, p.y - max_radius, p.y + max_radius, bounds.rect_min0, bounds.rect_max0, grid);
		bounds.rect_count = ((bounds.rect_max0.x - bounds.rect_min0.x) * (bounds.rect_max0.y - bounds.rect_min0.y)) > 0 ? 1 : 0;
		return bounds;
	}

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
