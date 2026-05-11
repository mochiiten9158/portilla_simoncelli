// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "constraints.h"
#include "mt19937ar.h"
#include "pca.h"
#include "filters.h"
#include "toolbox.h"
#include "ps_lib.h"
#include "pyramid.h"

#include <sys/stat.h>   // for mkdir
#include <string.h>     // for string operations
#include <stdint.h>
#include "image_utils.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Computations of the statistics of an image (from its pyramid decomposition)
// This corresponds to Line 2 of Algorithm 5
// See Appendix B.2 for the color case
// Note that some statistics are computed outside this function
// We use the following variable names:
// - Same scale --> cousins
// - Coarser scale --> parents

// void normalize_and_save_png(const char *filename, float *img, int width, int height) {
//     // Find min and max
//     float min_val = img[0], max_val = img[0];
//     for (int i = 1; i < width * height; i++) {
//         if (img[i] < min_val) min_val = img[i];
//         if (img[i] > max_val) max_val = img[i];
//     }

//     // Avoid division by zero
//     float range = (max_val - min_val);
//     if (range < 1e-8f) range = 1.0f;

//     // Allocate uint8 buffer
//     uint8_t *img8 = (uint8_t *)malloc(width * height);
//     for (int i = 0; i < width * height; i++) {
//         float val = (img[i] - min_val) / range;   // normalize 0–1
//         img8[i] = (uint8_t)(255.0f * val);
//     }

//     // Save as PNG
//     stbi_write_png(filename, width, height, 1, img8, width);
//     free(img8);
// }

static void compute_stats(statsStruct *stats, pyramidStruct pyramid,
                          const imageStruct sample, const filtersStruct filters,
                          const paramsStruct params, int nz)
{

  // ================= STATISTICS PROGRESS TRACKING =================
  // static int STAT_COUNTER = 0;
  // static const int STAT_TOTAL = 927;

  // static inline void report_stats(int n, int scale, const char *label)
  // {
  //     STAT_COUNTER += n;
  //     if (scale >= 0)
  //         printf("[analysis] scale %d: %-30s %4d / %d\n",
  //               scale, label, STAT_COUNTER, STAT_TOTAL);
  //     else
  //         printf("[analysis] global : %-30s %4d / %d\n",
  //               label, STAT_COUNTER, STAT_TOTAL);
  // }
  // =================================================================

  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int Na = params.Na;
  int hNa = (Na-1)/2; // variance location in auto-correlation matrix

  // variable declaration
  float meani, vari, theta;
  int i, j, k, l, ind, sx, sy, N_data;

  // image size
  int nx = filters.size[0];
  int ny = filters.size[1];

  // ================= STATISTICS COUNTING =================
  // int stats_this_scale = 0;
  // int total_stats = 0;

  // #define COUNT(n) do { stats_this_scale += (n); total_stats += (n); } while(0)
  // #define REPORT(scale, label) \
  //     printf("[stats] scale %d: %-28s +%4d (scale=%4d total=%4d)\n", \
  //           scale, label, stats_this_scale, stats_this_scale, total_stats)
  // =======================================================

  // memory allocation
  float **magSteered = (float **) malloc(N_steer*sizeof(float *));
  float **realSteered = (float **) malloc(N_steer*sizeof(float *));
  float **parents = (float **) malloc(N_steer*sizeof(float *));
  float **rparents = (float **) malloc(2*N_steer*sizeof(float *));
  for(j = 0; j < N_steer; j++) {
    magSteered[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    realSteered[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    parents[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    rparents[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    rparents[j + N_steer] = (float *) malloc(nx*ny*nz*sizeof(float));
  }
  fftwf_complex *fft_tmp = (fftwf_complex *)
    fftwf_malloc(nx*ny*nz*sizeof(fftwf_complex));
  fftwf_complex *fft_tmp2 = (fftwf_complex *)
    fftwf_malloc(nx*ny*nz*sizeof(fftwf_complex));
  float *tmp = (float *) malloc(nx*ny*nz*sizeof(float));
  // float *variance = (float *) malloc(nz*sizeof(float));

  // compute the pixel statistics of PCA bands
  if ( nz == 3 ) {
    // central auto-correlation
    // summary statistics (viii)
    compute_auto_cor(stats->autoCorPCA, sample.image, pyramid.in_plan,
                     pyramid.out_plan, pyramid.plan[0], pyramid.iplan[0],
                     nx, ny, nz, Na);

    // skewness and kurtosis
    // summary statistics (ix)
    for (l = 0; l < 3; l++) {
      vari = stats->autoCorPCA[hNa+hNa*Na+l*Na*Na];
      stats->pixelStatsPCA[0 + N_PIXELSTATSPCA*l] =
        compute_skewness(sample.image + l*nx*ny, 0.0, vari, nx*ny);
      stats->pixelStatsPCA[1 + N_PIXELSTATSPCA*l] =
        compute_kurtosis(sample.image + l*nx*ny, 0.0, vari, nx*ny);
    }
  }

  // variance of the high-pass
  // summary statistics (i)(b)
  for(l = 0; l < nz; l++)
    stats->varHigh[l] = compute_moment(pyramid.highband + l*nx*ny, 0.0, 2,
                                       nx*ny);

  // variance for skewness and kurtosis computations
  // if ( nz == 3 )
  //   for(l = 0; l < nz; l++)
  //     variance[l] = stats->eigenValuesPCA[l];
  //  else
  //     variance[0] = stats->pixelStats[3];

  // statistics of the low-frequency residual at each scale
  for (i = 0; i < 1+N_pyr; i++) {
    // stats_this_scale = 0;
    sx = filters.size[2*i];
    sy = filters.size[2*i+1];

    // apply second low-pass
    do_fft_plan_real(pyramid.plan[i], pyramid.out_plan, pyramid.in_plan,
                     fft_tmp, pyramid.lowband[i], sx*sy, nz);
    pointwise_complexfloat_multiplication(fft_tmp, fft_tmp, filters.lowpass0[i],
                                          sx*sy, nz);
    do_ifft_plan_real(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                      tmp, fft_tmp, sx*sy, nz);

    // compute central auto-correlation
    // summary statistics (ii)
    compute_auto_cor(stats->autoCorLow[i], tmp, pyramid.in_plan,
                     pyramid.out_plan, pyramid.plan[i], pyramid.iplan[i],
                     sx, sy, nz, Na);

    // compute skewness and kurtosis
    // summary statistics (i)(a)
    for (l = 0; l < nz; l++) {
      vari = stats->autoCorLow[i][hNa + hNa*Na + l*Na*Na];
      // uncomment the following lines to do as in PS matlab implementation
      // the condition is useless because the skewness and kurtosis adjustments
      // are not performed during the analysis if the test fails
      // if ( vari*pow(16, i)/variance[l] > 1e-6) {
        stats->skewLow[i + (1+N_pyr)*l] = compute_skewness(tmp + l*sx*sy, 0.0,
                                                           vari, sx*sy);
        stats->kurtLow[i + (1+N_pyr)*l] = compute_kurtosis(tmp + l*sx*sy, 0.0,
                                                           vari, sx*sy);
      // }
      // else {
      //   stats->skewLow[i + (1+N_pyr)*l] = 0.0;
      //   stats->kurtLow[i + (1+N_pyr)*l] = 3.0;
      // }
    }
  }

  // loop over the scales to compute the statistics of the steered bands
  for(i = 0; i < N_pyr; i++) {
    // sizes
    sx = filters.size[2*i];
    sy = filters.size[2*i+1];

    // loop on the orientation
    for(j = 0; j < N_steer; j++) {
      // index of the corresponding steered band
      ind = j + i*N_steer;

      // compute the magnitude and the real part
      for(k = 0; k < sx*sy*nz; k++) {
        magSteered[j][k] = hypot(pyramid.steered[ind][k][0],
                                 pyramid.steered[ind][k][1]);
        realSteered[j][k] = pyramid.steered[ind][k][0];
      }

      // compute the mean of the magnitude, store it and remove it
      for(l = 0; l < nz; l++) {
        meani = stats->magMeans[ind + (N_pyr*N_steer)*l] =
          mean(magSteered[j] + l*sx*sy, sx*sy);
        for(k = 0; k < sx*sy; k++)
          magSteered[j][k + l*sx*sy] -= meani;
      }
    }

    // compute the central auto-correlation of the modulus
    // summary statistics (iii)
    for(j = 0; j < N_steer; j++) {
      ind = j + i*N_steer;
      compute_auto_cor(stats->autoCorMag[ind], magSteered[j], pyramid.in_plan,
                       pyramid.out_plan, pyramid.plan[i], pyramid.iplan[i],
                       sx, sy, nz, Na);
      // COUNT(Na * Na * nz);

      // // ===== DEBUG VISUALIZATION =====
      // printf("[debug] auto-correlation of magnitude, scale %d orient %d\n", i, j);
      // // ============================================================
      // // DEBUG VISUALIZATION BLOCK (C++ SAFE)
      // // ============================================================

      // if (i == 0 && j == 0)

      //     // ----------------------------------------------------------
      //     // 1. Save magnitude subband PNG
      //     // ----------------------------------------------------------
      //     {
      //         float *mag_display = new float[sx * sy];
      //         float mag_mean = stats->magMeans[ind];

      //         for (int p = 0; p < sx * sy; p++)
      //             mag_display[p] = magSteered[0][p] + mag_mean;

      //         normalize_and_save_png("dbg_mag_s0_o0.png", mag_display, sx, sy);

      //         delete[] mag_display;
      //     }

      //     // ----------------------------------------------------------
      //     // 2. Visualize pixel pairs for every lag
      //     // ----------------------------------------------------------

      //     float *mag_full = new float[sx * sy];

      //     {
      //         float mag_mean = stats->magMeans[ind];
      //         for (int p = 0; p < sx * sy; p++)
      //             mag_full[p] = magSteered[0][p] + mag_mean;
      //     }

      //     float mag_min = mag_full[0];
      //     float mag_max = mag_full[0];

      //     for (int p = 1; p < sx * sy; p++) {
      //         if (mag_full[p] < mag_min) mag_min = mag_full[p];
      //         if (mag_full[p] > mag_max) mag_max = mag_full[p];
      //     }

      //     float mag_range = (mag_max - mag_min) < 1e-8f ? 1.f : (mag_max - mag_min);

      //     uint8_t *canvas = new uint8_t[sx * sy * 3];

      //     #define DRAW_DOT(row, col, R, G, B) do {                        \
      //             for (int _dr = 0; _dr <= 0; _dr++)                      \
      //             for (int _dc = 0; _dc <= 0; _dc++) {                    \
      //                 int _r = (row + _dr + sy) % sy;                      \
      //                 int _c = (col + _dc + sx) % sx;                      \
      //                 canvas[(_r * sx + _c) * 3 + 0] = (R);                \
      //                 canvas[(_r * sx + _c) * 3 + 1] = (G);                \
      //                 canvas[(_r * sx + _c) * 3 + 2] = (B);                \
      //             }                                                        \
      //         } while(0)
          
      //     #define DRAW_BOX(row, col, size, R, G, B) do {              \
      //               int half = (size)/2;                                    \
      //               for (int dr = -half; dr <= half; dr++)                  \
      //               for (int dc = -half; dc <= half; dc++) {                \
      //               int rr = (row + dr + sy) % sy;                      \
      //               int cc = (col + dc + sx) % sx;                      \
      //               canvas[(rr * sx + cc) * 3 + 0] = (R);               \
      //               canvas[(rr * sx + cc) * 3 + 1] = (G);               \
      //               canvas[(rr * sx + cc) * 3 + 2] = (B);               \
      //               }                                                       \
      //         } while(0)

      //     // #define DRAW_LINE(r0, c0, r1, c1, R, G, B) do {                  \
      //     //         int _dr = std::abs((r1)-(r0));                           \
      //     //         int _dc = std::abs((c1)-(c0));                           \
      //     //         int _steps = (_dr > _dc) ? _dr : _dc;                    \
      //     //         if (_steps == 0) break;                                  \
      //     //         for (int _s = 0; _s <= _steps; _s++) {                   \
      //     //             int _r = (r0) + _s * ((r1)-(r0)) / _steps;           \
      //     //             int _c = (c0) + _s * ((c1)-(c0)) / _steps;           \
      //     //             if (_r < 0 || _r >= sy || _c < 0 || _c >= sx) continue; \
      //     //             canvas[(_r * sx + _c) * 3 + 0] = (R);                \
      //     //             canvas[(_r * sx + _c) * 3 + 1] = (G);                \
      //     //             canvas[(_r * sx + _c) * 3 + 2] = (B);                \
      //     //         }                                                        \
      //     //     } while(0)

      //         int hNa_loc = (Na - 1) / 2;

      //         int step = sx;
      //         int box = step / 2;
      //         if (step < 1) step = 1;

      //         mkdir("dbg_lags", 0755);

      //         for (int dn = -hNa_loc; dn <= hNa_loc; dn++) {
      //         for (int dm = -hNa_loc; dm <= hNa_loc; dm++) {

      //             for (int p = 0; p < sx * sy; p++) {
      //                 uint8_t g = (uint8_t)(255.f * (mag_full[p] - mag_min) / mag_range);
      //                 canvas[p * 3 + 0] = g;
      //                 canvas[p * 3 + 1] = g;
      //                 canvas[p * 3 + 2] = g;
      //             }

      //             for (int r = step/2; r < sy + 1; r += step) {
      //             for (int c = step/2; c < sx + 1; c += step) {

      //                 int r2 = (r + dn + sy) % sy;
      //                 int c2 = (c + dm + sx) % sx;

      //                 //DRAW_LINE(r, c, r2, c2, 80, 80, 255);

      //                 //DRAW_DOT(r, c, 220, 30, 30);
      //                 //DRAW_DOT(r2, c2, 30, 200, 30);

      //                 DRAW_BOX(r,  c,  box, 255,0,0);   // source region
      //                 DRAW_BOX(r2, c2, box, 0,255,0);   // shifted region

      //             }}

      //             char fname[256];
      //             sprintf(fname, "dbg_lags/lag_dn%+03d_dm%+03d.ppm", dn, dm);

      //             FILE *fp = fopen(fname, "wb");
      //             if (fp) {
      //                 fprintf(fp, "P6\n%d %d\n255\n", sx, sy);
      //                 fwrite(canvas, 1, sx * sy * 3, fp);
      //                 fclose(fp);
      //             }

      //             float acval = stats->autoCorMag[ind][(dn + hNa_loc) + (dm + hNa_loc) * Na];

      //             printf("[lag_vis] dn=%+d dm=%+d  acorr=%.5f  -> %s\n",
      //                   dn, dm, acval, fname);
      //         }}

      //         {
      //             float ac_min = stats->autoCorMag[ind][0];
      //             float ac_max = stats->autoCorMag[ind][0];

      //             for (int p = 1; p < Na * Na; p++) {
      //                 if (stats->autoCorMag[ind][p] < ac_min) ac_min = stats->autoCorMag[ind][p];
      //                 if (stats->autoCorMag[ind][p] > ac_max) ac_max = stats->autoCorMag[ind][p];
      //             }

      //             float ac_range = (ac_max - ac_min) < 1e-10f ? 1.f : (ac_max - ac_min);

      //             int scale_up = 8;
      //             int W = Na * scale_up;
      //             int H = Na * scale_up;

      //             uint8_t *pgm = new uint8_t[W * H];

      //             for (int r = 0; r < Na; r++)
      //             for (int c = 0; c < Na; c++) {

      //                 uint8_t v = (uint8_t)(255.f *
      //                     (stats->autoCorMag[ind][r * Na + c] - ac_min) / ac_range);

      //                 for (int dr = 0; dr < scale_up; dr++)
      //                 for (int dc = 0; dc < scale_up; dc++)
      //                     pgm[(r*scale_up+dr)*W + (c*scale_up+dc)] = v;
      //             }

      //             FILE *fp = fopen("dbg_lags/autocorr_matrix.pgm", "wb");

      //             if (fp) {
      //                 fprintf(fp, "P5\n%d %d\n255\n", W, H);
      //                 fwrite(pgm, 1, W * H, fp);
      //                 fclose(fp);
      //             }

      //             delete[] pgm;
      //         }

      //         delete[] mag_full;
      //         delete[] canvas;

      //     #undef DRAW_DOT
      //     //#undef DRAW_LINE
      //     }

      //     // ============================================================
      //     // END DEBUG BLOCK
      //     // ============================================================
      }
    // REPORT(i, "autoCorMag");

    // compute the parents (coarser scale) for the cross-correlation
    if ( i == N_pyr - 1 ) { // last scale
      if ( nz == 3) {
        // zoom of the last low-band
        do_fft_plan_real(pyramid.plan[i+1], pyramid.out_plan, pyramid.in_plan,
                         fft_tmp, pyramid.lowband[i+1], 0.25*sx*sy, nz);
        upsampling(fft_tmp2, fft_tmp, sx/2, sy/2, nz);
        do_ifft_plan_real(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                          tmp, fft_tmp2, sx*sy, nz);

        // rparents are filled shifted version of the zoomed low-band (5 columns)
        // the parents are not used
        memcpy(rparents[0], tmp, nz*sx*sy*sizeof(float));
        shift(rparents[1], tmp, 0, 2, sx, sy, nz);
        shift(rparents[2], tmp, 0, -2, sx, sy, nz);
        shift(rparents[3], tmp, 2, 0, sx, sy, nz);
        shift(rparents[4], tmp, -2, 0, sx, sy, nz);
      }
    }
    else { // not the last scale
      // loop on the orientation
      for(j = 0; j < N_steer; j++) {
        // index of the steered band from the coarser scale
        ind = j + (i+1)*N_steer;

        // zoom
        do_fft_plan(pyramid.plan[i+1], pyramid.out_plan, pyramid.in_plan,
                    fft_tmp, pyramid.steered[ind], 0.25*sx*sy, nz);
        upsampling(fft_tmp2, fft_tmp, sx/2, sy/2, nz);
        do_ifft_plan(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                          fft_tmp, fft_tmp2, sx*sy, nz);

        // loop over the pixels
        for(k = 0; k < sx*sy*nz; k++) {
          // store the modulus in parents[j]
          parents[j][k] = hypot(fft_tmp[k][0], fft_tmp[k][1]);

          // double the phase of the parents
          theta = 2*atan2(fft_tmp[k][1], fft_tmp[k][0]);

          // store the real part in rparents[j]
          rparents[j][k] = parents[j][k]*cos(theta);

          // store the imaginary part in rparents[j+N_steer]
          rparents[j + N_steer][k] = parents[j][k]*sin(theta);
        }

        // remove the mean of parents[j]
        for(l = 0; l < nz; l++) {
            meani = mean(parents[j] + l*sx*sy, sx*sy);
            for(k = 0; k < sx*sy; k++)
                parents[j][k + l*sx*sy] -= meani;
        }
      }
    }

    // compute the pairwise cross-correlation of the magnitude
    // summary statistics (iv)
    compute_cross_cor(stats->cousinMagCor[i], magSteered, N_steer, sx*sy, nz);

    // COUNT(N_steer * N_steer * nz);
    // REPORT(i, "cousinMagCor");

    // compute the cross-correlation of the magnitude with the coarser scale
    // summmary statistics (v)
    if (i < N_pyr - 1)
      compute_cross_scale_cor(stats->parentMagCor[i], magSteered, parents,
                              N_steer, N_steer, sx*sy, nz);

      // COUNT(N_steer * N_steer * nz);
      // REPORT(i, "parentMagCor");

    // compute the pairwise cross-correlation of the real part
    if (nz == 3) {
      // compute the cross-correlation
      // summary statistics (x)
      compute_cross_cor(stats->cousinRealCor[i], realSteered, N_steer, sx*sy, nz);

      // COUNT(N_steer * N_steer * nz);
      // REPORT(i, "cousinRealCor");

      // additionnal computation for the last scale
      if (i == N_pyr - 1) {
        // compute the cross-correlation
        // summary statistics (xi)
        compute_cross_cor(stats->cousinRealCor[i+1], rparents, N_SMALLEST, sx*sy, nz);
      }
    }

    // compute cross-correlation of the real part with the coarser scale
    // summary statistics (vi) (and (xii) for the last scale if color)
    if (i < N_pyr - 1 || nz == 3) { // except for grayscale and last scale
      // different size at the coarsest scale
      N_data = (i == N_pyr - 1) ? N_SMALLEST : 2*N_steer;

      // compute the cross-correlation
      compute_cross_scale_cor(stats->parentRealCor[i], realSteered,
                              rparents, N_steer, N_data, sx*sy, nz);
    }
  }

  // printf("\n[stats] TOTAL STATISTICS FILLED: %d\n\n", total_stats);

  // free memory
  for(j = 0; j < N_steer; j++) {
    free(magSteered[j]);
    free(realSteered[j]);
    free(parents[j]);
    free(rparents[j]);
    free(rparents[j + N_steer]);
  }
  free(magSteered);
  free(realSteered);
  free(parents);
  free(rparents);
  fftwf_free(fft_tmp);
  fftwf_free(fft_tmp2);
  free(tmp);
  //free(variance);
}

// Analysis of an image
// This corresponds to Line 1 and Line 2 of Algorithm 5
void analysis(statsStruct *stats, imageStruct sample, const paramsStruct params)
{
  int i, l;

  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int verbose = params.verbose;
  int nx = sample.nx;
  int ny = sample.ny;
  int nz = sample.nz;
  int N = nx*ny;

  // option for the filters and the pyramid building
  int option = 1;

  // compute pixel stats before applying the PCA or adding noise
  // summary statistics (i)(c)
  float m0, var0;
  for (l = 0; l < nz; l++) {
    min_and_max(&stats->pixelStats[0 + N_PIXELSTATS*l],
                &stats->pixelStats[1 + N_PIXELSTATS*l],
                sample.image + l*N, N);
    m0 = stats->pixelStats[2 + N_PIXELSTATS*l] = mean(sample.image + l*N, N);
    var0 = stats->pixelStats[3 + N_PIXELSTATS*l]
      = compute_moment(sample.image + l*N, m0, 2, N);
    stats->pixelStats[4 + N_PIXELSTATS*l] = compute_skewness(sample.image + l*N,
                                                             m0, var0, N);
    stats->pixelStats[5 + N_PIXELSTATS*l] = compute_kurtosis(sample.image + l*N,
                                                             m0, var0, N);
  }

  // add noise to the sample to avoid instability in case of synthetic texture
  if ( nz == 1 ) {
    float factor;
    for (l = 0; l < nz; l++) {
      factor = (stats->pixelStats[1 + N_PIXELSTATS*l]
        - stats->pixelStats[0 + N_PIXELSTATS*l])/100000;
      for (i = 0; i < N; i++)
        sample.image[i + l*N] += factor*mt_genrand_res53();
    }
  }

  // apply PCA (see Appendix B.1)
  if( nz == 3 ) {
    // substract the mean value of each channel (already computed)
    for(l = 0; l < 3; l++)
      for(i = 0; i < N; i++)
        sample.image[i + l*N] -= stats->pixelStats[2 + N_PIXELSTATS*l];

    // compute the color covariance matrix
    // summary statistics (vii)
    compute_covariance(sample.image, stats->covariancePCA, N);

    // compute the change of basis matrix
    eigen_decomposition(stats->covariancePCA, stats->eigenVectorsPCA,
                        stats->eigenValuesPCA);

    // apply PCA
    apply_pca(sample.image, sample.image, stats->eigenVectorsPCA,
              stats->eigenValuesPCA, N);
  }

  // compute the filters and their sizes (for the sample)
  if ( verbose )
    printf("Creating filters for the sample\n");
  filtersStruct filters;
  compute_filters(&filters, nx, ny, N_pyr, N_steer, option);

  // memory allocation for the pyramid
  pyramidStruct pyramid;
  allocate_pyramid(&pyramid, N_pyr, N_steer, nx, ny, nz, option);

  // precomputing the plan for the fft
  precompute_plan(pyramid.plan, pyramid.iplan, pyramid.in_plan,
                  pyramid.out_plan, nx, ny, N_pyr);

  // creating the pyramid for the sample
  if ( verbose )
    printf("Creating the pyramid for the sample\n");
  create_pyramid(pyramid, sample, filters, params, option);

  // ================= SAVE PYRAMID DECOMPOSITION =================
  // mkdir("pyramid_analysis", 0755);

  // for (int s = 0; s < N_pyr; s++) {
  //     int sx = nx >> s;
  //     int sy = ny >> s;

  //     for (int o = 0; o < N_steer; o++) {
  //         fftwf_complex *band = pyramid.steered[s * N_steer + o];

  //         float *img = (float*)malloc(sx * sy * sizeof(float));
  //         for (int i = 0; i < sx * sy; i++)
  //             img[i] = band[i][0];

  //         char fname[256];
  //         sprintf(fname, "pyramid_analysis/scale_%d_orient_%d.png", s, o);
  //         normalize_and_save_png(fname, img, sx, sy);

  //         free(img);
  //     }
  // }

  // // save low-pass residual
  // {
  //     int sx = nx >> N_pyr;
  //     int sy = ny >> N_pyr;

  //     float *img = (float*)malloc(sx * sy * sizeof(float));
  //     for (int i = 0; i < sx * sy; i++)
  //         img[i] = pyramid.lowband[N_pyr][i];

  //     normalize_and_save_png("pyramid_analysis/lowpass.png", img, sx, sy);
  //     free(img);
  // }

  // {
  //   float *img = (float*)malloc(nx * ny * sizeof(float));
  //   for (int i = 0; i < nx * ny; i++)
  //       img[i] = pyramid.highband[i];

  //   normalize_and_save_png("pyramid_analysis/highpass.png", img, nx, ny);
  //   free(img);
  // }
  // ===========================================================

  // computing the statistics of the sample (Line 2 of Algorithm 5)
  if ( verbose )
    printf("Computing the statistics of the sample\n");
  compute_stats(stats, pyramid, sample, filters, params, nz);

  // free memory
  free_filters(filters, N_pyr, N_steer, option);
  free_pyramid(pyramid, N_pyr, N_steer, option);
}

// Analysis of an image
// This corresponds to Line 1 and Line 2 of Algorithm 5
// This function is similar to the analysis() function (above) but it uses
// pre-computed filters and a pre-allocated pyramid
// In addition the noise is not added and there is no verbose mode
void analysis2(statsStruct *stats, imageStruct sample, pyramidStruct pyramid,
               const filtersStruct filters, const paramsStruct params)
{
  int i, l;

  // parameters
  int nx = sample.nx;
  int ny = sample.ny;
  int nz = sample.nz;
  int N = nx*ny;

  // option for the filters and the pyramid building
  int option = 1;

  // compute pixel stats before applying the PCA or adding noise
  // summary statistics (i)(c)
  float m0, var0;
  for (l = 0; l < nz; l++) {
    min_and_max(&stats->pixelStats[0 + N_PIXELSTATS*l],
                &stats->pixelStats[1 + N_PIXELSTATS*l],
                sample.image + l*N, N);
    m0 = stats->pixelStats[2 + N_PIXELSTATS*l] = mean(sample.image + l*N, N);
    var0 = stats->pixelStats[3 + N_PIXELSTATS*l]
      = compute_moment(sample.image + l*N, m0, 2, N);
    stats->pixelStats[4 + N_PIXELSTATS*l] = compute_skewness(sample.image + l*N,
                                                             m0, var0, N);
    stats->pixelStats[5 + N_PIXELSTATS*l] = compute_kurtosis(sample.image + l*N,
                                                             m0, var0, N);
  }

  // apply PCA (see Appendix B.1)
  if( nz == 3 ) {
    // substract the mean value of each channel (already computed)
    for(l = 0; l < 3; l++)
      for(i = 0; i < N; i++)
        sample.image[i + l*N] -= stats->pixelStats[2 + N_PIXELSTATS*l];

    // compute the color covariance matrix
    // summary statistics (vii)
    compute_covariance(sample.image, stats->covariancePCA, N);

    // compute the change of basis matrix
    eigen_decomposition(stats->covariancePCA, stats->eigenVectorsPCA,
                        stats->eigenValuesPCA);

    // apply PCA
    apply_pca(sample.image, sample.image, stats->eigenVectorsPCA,
              stats->eigenValuesPCA, N);
  }

  // creating the pyramid for the sample
  create_pyramid(pyramid, sample, filters, params, option);

  // computing the statistics of the sample (Line 2 of Algorithm 5)
  compute_stats(stats, pyramid, sample, filters, params, nz);
}
