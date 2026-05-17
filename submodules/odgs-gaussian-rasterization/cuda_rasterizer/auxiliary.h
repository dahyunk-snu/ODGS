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

__forceinline__ __device__ float2 omniPixelDeltaFallback(const float2 pix, const float2 mean)
{
	return { pix.x - mean.x, pix.y - mean.y };
}

__forceinline__ __device__ OmniLogMapBaseResult computeOmniLogMapBase(
	const float2 pix,
	const float2 mean,
	const int W,
	const int H)
{
	OmniLogMapBaseResult result;
	result.d = omniPixelDeltaFallback(pix, mean);
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

	if (W <= 0 || H <= 0 || !isFinite(pix) || !isFinite(mean))
		return result;

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float Wf = (float)W;
	const float Hf = (float)H;

	const float lambda = (pix.x / Wf - 0.5f) * two_pi;
	const float theta = (0.5f - pix.y / Hf) * pi;
	const float lambda0 = (mean.x / Wf - 0.5f) * two_pi;
	const float theta0 = (0.5f - mean.y / Hf) * pi;

	const float sin_lambda = sinf(lambda);
	const float cos_lambda = cosf(lambda);
	const float sin_theta = sinf(theta);
	const float cos_theta = cosf(theta);
	const float sin_lambda0 = sinf(lambda0);
	const float cos_lambda0 = cosf(lambda0);
	const float sin_theta0 = sinf(theta0);
	const float cos_theta0 = cosf(theta0);

	if (!isFinite(lambda) || !isFinite(theta) || !isFinite(lambda0) || !isFinite(theta0) ||
		!isFinite(sin_lambda) || !isFinite(cos_lambda) || !isFinite(sin_theta) || !isFinite(cos_theta) ||
		!isFinite(sin_lambda0) || !isFinite(cos_lambda0) || !isFinite(sin_theta0) || !isFinite(cos_theta0) ||
		fabsf(cos_theta0) < 1.0e-4f)
		return result;

	const float3 u = {
		cos_theta * sin_lambda,
		-sin_theta,
		cos_theta * cos_lambda
	};
	const float3 u0 = {
		cos_theta0 * sin_lambda0,
		-sin_theta0,
		cos_theta0 * cos_lambda0
	};
	const float3 e_lambda = {
		cos_lambda0,
		0.0f,
		-sin_lambda0
	};
	const float3 e_theta = {
		-sin_theta0 * sin_lambda0,
		-cos_theta0,
		-sin_theta0 * cos_lambda0
	};

	const float q = dot3(u, u0);
	if (!isFinite(q))
		return result;

	const float q_clamped = clampf(q, -1.0f, 1.0f);
	const float gamma = acosf(q_clamped);
	const float sin_gamma = sinf(gamma);

	if (!isFinite(gamma) || !isFinite(sin_gamma) ||
		gamma < 1.0e-6f || fabsf(sin_gamma) < 1.0e-6f)
		return result;

	float alpha = 0.0f;
	if (gamma < 1.0e-4f)
	{
		const float gamma2 = gamma * gamma;
		alpha = 1.0f + gamma2 / 6.0f;
	}
	else
	{
		alpha = gamma / sin_gamma;
	}

	const float3 w = sub3(u, mul3(u0, q_clamped));
	const float3 v = mul3(w, alpha);
	const float xi = dot3(v, e_lambda);
	const float eta = dot3(v, e_theta);
	const float2 d = {
		(xi / cos_theta0) * Wf / two_pi,
		-eta * Hf / pi
	};

	if (!isFinite(alpha) || !isFinite(w) || !isFinite(v) || !isFinite(d))
		return result;

	result.d = d;
	result.lambda0 = lambda0;
	result.theta0 = theta0;
	result.sin_theta0 = sin_theta0;
	result.cos_theta0 = cos_theta0;
	result.sin_lambda0 = sin_lambda0;
	result.cos_lambda0 = cos_lambda0;
	result.q = q_clamped;
	result.gamma = gamma;
	result.sin_gamma = sin_gamma;
	result.alpha = alpha;
	result.u = u;
	result.u0 = u0;
	result.w = w;
	result.v = v;
	result.e_lambda = e_lambda;
	result.e_theta = e_theta;
	result.fallback = false;
	return result;
}

__forceinline__ __device__ float2 logMapOmniDeltaPixel(
	const float2 pix,
	const float2 mean,
	const int W,
	const int H)
{
	return computeOmniLogMapBase(pix, mean, W, H).d;
}

__forceinline__ __device__ OmniLogMapDeltaResult logMapOmniDeltaPixelWithMeanJacobian(
	const float2 pix,
	const float2 mean,
	const int W,
	const int H)
{
	OmniLogMapDeltaResult result;
	const OmniLogMapBaseResult base = computeOmniLogMapBase(pix, mean, W, H);
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
	const float cos_gamma = cosf(gamma);
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