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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of the tangent-plane 2D covariance computation
// (due to length launched as separate kernel before other
// backward steps contained in preprocess)
__global__ void computeTangentCov2DBackwardCUDA(int P,
	const int height,
	const int width,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute the tangent frame and the 2D covariance
	// exactly as the forward pass did.
	const float e = 0.0000001f;
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, viewmatrix);

	const OmniTangentGeom geom = makeOmniTangentGeom(t);
	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float kx = (float)width / two_pi;
	const float ky = (float)height / pi;
	const float3 ex = mul3(geom.e_phi, kx);
	const float3 ey = mul3(geom.e_theta, -ky);
	const float inv_r = 1.0f / geom.r;
	const float st0 = geom.sin_theta0;
	const float ct0 = geom.cos_theta0;

	const float3 j0 = mul3(transformVec4x3Transpose(ex, viewmatrix), inv_r);
	const float3 j1 = mul3(transformVec4x3Transpose(ey, viewmatrix), inv_r);
	const float3 Sj0 = {
		cov3D[0] * j0.x + cov3D[1] * j0.y + cov3D[2] * j0.z,
		cov3D[1] * j0.x + cov3D[3] * j0.y + cov3D[4] * j0.z,
		cov3D[2] * j0.x + cov3D[4] * j0.y + cov3D[5] * j0.z
	};
	const float3 Sj1 = {
		cov3D[0] * j1.x + cov3D[1] * j1.y + cov3D[2] * j1.z,
		cov3D[1] * j1.x + cov3D[3] * j1.y + cov3D[4] * j1.z,
		cov3D[2] * j1.x + cov3D[4] * j1.y + cov3D[5] * j1.z
	};

	// Use helper variables for 2D covariance entries. More compact.
	const float a = dot3(j0, Sj0) + 0.3f * ct0 * ct0;
	const float b = dot3(j0, Sj1);
	const float c = dot3(j1, Sj1) + 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + e);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		dL_dcov[6 * idx + 0] = (j0.x * j0.x * dL_da + j0.x * j1.x * dL_db + j1.x * j1.x * dL_dc);
		dL_dcov[6 * idx + 3] = (j0.y * j0.y * dL_da + j0.y * j1.y * dL_db + j1.y * j1.y * dL_dc);
		dL_dcov[6 * idx + 5] = (j0.z * j0.z * dL_da + j0.z * j1.z * dL_db + j1.z * j1.z * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		dL_dcov[6 * idx + 1] = 2 * j0.x * j0.y * dL_da + (j0.x * j1.y + j0.y * j1.x) * dL_db + 2 * j1.x * j1.y * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * j0.x * j0.z * dL_da + (j0.x * j1.z + j0.z * j1.x) * dL_db + 2 * j1.x * j1.z * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * j0.y * j0.z * dL_da + (j0.y * j1.z + j0.z * j1.y) * dL_db + 2 * j1.y * j1.z * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients w.r.t. the Jacobian rows; Sigma_3D * j is already available.
	const float3 dL_dj0 = add3(mul3(Sj0, 2.0f * dL_da), mul3(Sj1, dL_db));
	const float3 dL_dj1 = add3(mul3(Sj0, dL_db), mul3(Sj1, 2.0f * dL_dc));

	// j = R^T E / r  =>  dL/dE = R (dL/dj) / r  and  dL/dr = -(dL/dj . j) / r.
	const float3 dL_dex = mul3(transformVec4x3(dL_dj0, viewmatrix), inv_r);
	const float3 dL_dey = mul3(transformVec4x3(dL_dj1, viewmatrix), inv_r);
	float dL_dr = -(dot3(dL_dj0, j0) + dot3(dL_dj1, j1)) * inv_r;

	// Frame rotation chain, ex = kx * e_phi and ey = -ky * e_theta with
	//   d(e_phi)/d(phi0)     = -cos(theta0) u0 + sin(theta0) e_theta
	//   d(e_theta)/d(phi0)   = -sin(theta0) e_phi
	//   d(e_phi)/d(theta0)   = 0
	//   d(e_theta)/d(theta0) = -u0
	const float3 dephi_dphi0 = add3(mul3(geom.u0, -ct0), mul3(geom.e_theta, st0));
	const float3 detheta_dphi0 = mul3(geom.e_phi, -st0);
	float dL_dphi0 = kx * dot3(dL_dex, dephi_dphi0) - ky * dot3(dL_dey, detheta_dphi0);
	float dL_dtheta0 = ky * dot3(dL_dey, geom.u0);
	// The low-pass dilation 0.3 cos^2(theta0) on the azimuth variance also
	// depends on theta0.
	dL_dtheta0 += dL_da * (-0.6f * ct0 * st0);

	// (phi0, theta0, r) -> camera-space position t, with
	// phi0 = atan2(tx, tz), theta0 = atan2(-ty, r_xz), r = |t|.
	const float dist = geom.r;
	const float dist_xz = geom.r_xz;
	const float lon_denom = fmaxf(t.x * t.x + t.z * t.z, e);
	const float lat_denom = t.y * t.y + dist_xz * dist_xz;

	const float dlon_dtx = t.z / lon_denom;
	const float dlon_dtz = -t.x / lon_denom;
	const float dlat_dtx = t.y * t.x / (dist_xz * lat_denom);
	const float dlat_dty = -dist_xz / lat_denom;
	const float dlat_dtz = t.y * t.z / (dist_xz * lat_denom);

	const float dL_dtx = dL_dphi0 * dlon_dtx + dL_dtheta0 * dlat_dtx + dL_dr * t.x / dist;
	const float dL_dty = dL_dtheta0 * dlat_dty + dL_dr * t.y / dist;
	const float dL_dtz = dL_dphi0 * dlon_dtz + dL_dtheta0 * dlat_dtz + dL_dr * t.z / dist;

	// Account for transformation of mean to t
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2*z*(dL_dMt[0][1] - dL_dMt[1][0]) + 2*y*(dL_dMt[2][0] - dL_dMt[0][2]) + 2*x*(dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2*y*(dL_dMt[1][0] + dL_dMt[0][1]) + 2*z*(dL_dMt[2][0] + dL_dMt[0][2]) + 2*r*(dL_dMt[1][2] - dL_dMt[2][1]) - 4*x*(dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2*x*(dL_dMt[1][0] + dL_dMt[0][1]) + 2*r*(dL_dMt[2][0] - dL_dMt[0][2]) + 2*z*(dL_dMt[1][2] + dL_dMt[2][1]) - 4*y*(dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2*r*(dL_dMt[0][1] - dL_dMt[1][0]) + 2*x*(dL_dMt[2][0] + dL_dMt[0][2]) + 2*y*(dL_dMt[1][2] + dL_dMt[2][1]) - 4*z*(dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const int W, int H,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dcov3D,
	const float* dL_ddepth,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 mean = means[idx];
	float3 t = transformPoint4x3(mean, viewmatrix);

	float dist = sqrt(t.x*t.x + t.y*t.y + t.z*t.z)+0.0000001f;
	float dist2 = dist * dist;
	float dist_xz = sqrt(t.x*t.x + t.z*t.z)+0.0000001f;
	float dist_xz2 = dist_xz * dist_xz;

	const float x_scale = W/(2*M_PI)/dist_xz2;
	const float y_scale = H/(M_PI)/(dist2*dist_xz);

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	glm::vec3 dL_dmean;
	dL_dmean.x = x_scale*(viewmatrix[0]*t.z - viewmatrix[2]*t.x )*dL_dmean2D[idx].x + y_scale*(viewmatrix[1]*dist_xz2 - (viewmatrix[0]*t.x + viewmatrix[2]*t.z)*t.y )*dL_dmean2D[idx].y;
	dL_dmean.y = x_scale*(viewmatrix[4]*t.z - viewmatrix[6]*t.x )*dL_dmean2D[idx].x + y_scale*(viewmatrix[5]*dist_xz2 - (viewmatrix[4]*t.x + viewmatrix[6]*t.z)*t.y )*dL_dmean2D[idx].y;
	dL_dmean.z = x_scale*(viewmatrix[8]*t.z - viewmatrix[10]*t.x)*dL_dmean2D[idx].x + y_scale*(viewmatrix[9]*dist_xz2 - (viewmatrix[8]*t.x + viewmatrix[10]*t.z)*t.y)*dL_dmean2D[idx].y;
	dL_dmeans[idx] += dL_dmean;

	// That's the third part of the mean gradient. for omnidirectional depth (=dist)
	glm::vec3 dL_dmean2;
	dL_dmean2.x = (viewmatrix[0]*t.x + viewmatrix[1]*t.y + viewmatrix[2]*t.z)/dist * dL_ddepth[idx];
	dL_dmean2.y = (viewmatrix[4]*t.x + viewmatrix[5]*t.y + viewmatrix[6]*t.z)/dist * dL_ddepth[idx];
	dL_dmean2.z = (viewmatrix[8]*t.x + viewmatrix[9]*t.y + viewmatrix[10]*t.z)/dist * dL_ddepth[idx];
	dL_dmeans[idx] += dL_dmean2;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const OmniTangentFrame* __restrict__ frames,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ds,
	const float* __restrict__ final_As,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixs,
	const float* __restrict__ dL_ddpts,
	const float* __restrict__ dL_daccs,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	// Same per-pixel ray as the forward pass; the log-map delta and its mean
	// Jacobian are evaluated exactly at every pixel (no tile anchor).
	const float3 u = pixelRayDirection(pixf, W, H);

	const float pi = 3.14159265358979323846f;
	const float two_pi = 6.28318530717958647692f;
	const float kx = (float)W / two_pi;
	const float ky = (float)H / pi;
	const float inv_kx = two_pi / (float)W;
	const float inv_ky = pi / (float)H;

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ OmniTangentFrame collected_frame[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	const float Dep = inside ? final_Ds[pix_id] : 0;
    const float Acc = inside ? final_As[pix_id] : 0;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpix[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpix[i] = dL_dpixs[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_frame[block.thread_rank()] = frames[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const OmniTangentFrame f = collected_frame[j];
			const float q = u.x * f.u0_st.x + u.y * f.u0_st.y + u.z * f.u0_st.z;
			if (!(q > OMNI_Q_CUTOFF))
				continue;
			const float qc = fminf(q, OMNI_Q_MAX);
			const float ux = u.x * f.ex_ct.x + u.y * f.ex_ct.y + u.z * f.ex_ct.z;
			const float uy = u.x * f.ey_d.x + u.y * f.ey_d.y + u.z * f.ey_d.z;
			float alpha_lm, dalpha_dq;
			omniLogMapAlphaPair(qc, alpha_lm, dalpha_dq);
			const float2 d = { alpha_lm * ux, alpha_lm * uy };

			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpix[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpix[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			const float dL_ddelx = dL_dG * dG_ddelx;
			const float dL_ddely = dL_dG * dG_ddely;

			// Exact mean Jacobian d(d)/d(mean pixel), evaluated at this very
			// pixel. With b = dot(u, e_phi), ct = dot(u, e_theta) and
			// alpha' = d(alpha)/dq:
			//   d(dx)/d(mx) = alpha' ct0 b^2 + alpha (st0 ct - ct0 q)
			//   d(dx)/d(my) = -(kx/ky) alpha' b ct
			//   d(dy)/d(mx) = -(ky/kx) (alpha' ct0 b ct - alpha st0 b)
			//   d(dy)/d(my) = alpha' ct^2 - alpha q
			const float st0 = f.u0_st.w;
			const float ct0 = f.ex_ct.w;
			const float b_phi = ux * inv_kx;
			const float c_theta = -uy * inv_ky;
			const float ddx_dmx = dalpha_dq * ct0 * b_phi * b_phi + alpha_lm * (st0 * c_theta - ct0 * qc);
			const float ddx_dmy = -(kx * inv_ky) * dalpha_dq * b_phi * c_theta;
			const float ddy_dmx = -(ky * inv_kx) * (dalpha_dq * ct0 * b_phi * c_theta - alpha_lm * st0 * b_phi);
			const float ddy_dmy = dalpha_dq * c_theta * c_theta - alpha_lm * qc;
			const float dL_dmean_x = dL_ddelx * ddx_dmx + dL_ddely * ddy_dmx;
			const float dL_dmean_y = dL_ddelx * ddx_dmy + dL_ddely * ddy_dmy;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dmean_x * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dmean_y * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const int width, int height,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_ddepth,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.
	computeTangentCov2DBackwardCUDA << <(P + 255) / 256, 256 >> > (
		P, height, width,
		means3D,
		radii,
		cov3Ds,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		width, height,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		campos,
		(float3*)dL_dmean2D,
		dL_dcov3D,
		dL_ddepth,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const OmniTangentFrame* frames,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ds,
	const float* final_As,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixs,
	const float* dL_ddpts,
	const float* dL_daccs,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		frames,
		conic_opacity,
		colors,
		final_Ds,
		final_As,
		final_Ts,
		n_contrib,
		dL_dpixs,
		dL_ddpts,
		dL_daccs,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths
		);
}
