#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/analysis.h"
#include "src/ps_lib.h"
#include "src/constraints.h"

#include "src/image_utils.h"

// Function to write statistics to a text file
void write_statistics_to_file(statsStruct stats, paramsStruct params, int nz, const char* filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to write statistics");
        return;
    }

    int i, j, l, ind;

    int Na      = params.Na;
    int N_pyr   = params.N_pyr;
    int N_steer = params.N_steer;

    /* ------------------------------------------------------------------ */
    /* Low-band skewness and kurtosis (one line per scale)                 */
    /* ------------------------------------------------------------------ */
    //Supposed to be 10 parameters in original PS model
    //2(N_pyr + 1) = 2(4+1) = 10
    fprintf(fp, "Low-band skewness and kurtosis\n");
    for (i = 0; i < 1+N_pyr; i++) {
        for (l = 0; l < nz; l++) {
            fprintf(fp, " %f %f,", 
                stats.skewLow[i + (1 + N_pyr) * l],
                stats.kurtLow[i + (1 + N_pyr) * l]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    /* ------------------------------------------------------------------ */
    /* High-band variance                                                  */
    /* ------------------------------------------------------------------ */
    //Same as original PS model = 1 parameter
    fprintf(fp, "High-band variance\n");
    for (l = 0; l < nz; l++) {
        fprintf(fp, " %f,", stats.varHigh[l]);
    }
    fprintf(fp, "\n\n");

    /* ------------------------------------------------------------------ */
    /* Pixel statistics (min max mean var skew kurt)                       */
    /* ------------------------------------------------------------------ */
    //Same as original PS model = 6 parameters
    fprintf(fp,
        "Skewness, kurtosis, variance, mean, maximum and minimum "
        "of the reconstructed texture\n");
    for (l = 0; l < nz; l++) {
        for (i = 0; i < 6; i++){
            fprintf(fp, " %f", stats.pixelStats[i + N_PIXELSTATS*l]);
        }
    }
    fprintf(fp, "\n\n");

    /* ------------------------------------------------------------------ */
    /* Auto-correlations of low-frequency bands (one scale per line)       */
    /* ------------------------------------------------------------------ */
    //Right now, 49 x 4 = 196 parameters (Na*Na = 7*7 = 49 auto-correlation values per scale, and N_pyr = 4 scales)
    //Supposed to be ((N_pyr + 1) * ((Na * Na) + 1)/2) = 25 * 5 = 125 parameters in original PS model
    fprintf(fp, "Auto-correlations of the low-frequency bands at each scale\n");
    for (i = 0; i < N_pyr; i++) { // different here
        for (l = 0; l < Na * Na * nz; l++) {
            fprintf(fp, " %f", stats.autoCorLow[i][l]);
        }
        fprintf(fp, ",\n");
    }
    fprintf(fp, "\n");

    /* ------------------------------------------------------------------ */
    /* Auto-correlations of magnitude of oriented subbands                 */
    /* One line per (scale, orientation)                                   */
    /* ------------------------------------------------------------------ */
    // Right now 49 x 12 = 588 parameters (Na*Na = 7*7 = 49 auto-correlation values per oriented sub-band, and N_pyr * N_steer = 4 * 4 = 16 oriented sub-bands)
    // But we ignore the last 4 because it is the residual low-pass band, so 49 x 12 = 588 parameters
    // Supposed to be N_pyr * N_steer * ((Na * Na + 1)/2) = 4 * 4 * 25 = 25 * 16 = 400
    fprintf(fp, "Auto-correlations of the modulus of each oriented sub-band\n");
    for (i = 0; i < N_pyr - 1; i++) {
        for (j = 0; j < N_steer; j++) {
            ind = j + i * N_steer;
            for (l = 0; l < Na * Na * nz; l++) {
                fprintf(fp, " %f", stats.autoCorMag[ind][l]);
            }
            fprintf(fp, ",\n");
        }
    }
    fprintf(fp, "\n");

    /* ------------------------------------------------------------------ */
    /* Cousin magnitude correlations (QxQ, one scale per block)            */
    /* ------------------------------------------------------------------ */
    // Right now 4 * 8 = 32 parameters (N_pyr - 2 = 4 - 2 = 2 scales, and Q = N_steer * nz = 4 * 1 = 4 orientations per scale, so QxQ = 16 values per scale, and 16 x 2 = 32 parameters)
    // Supposed to be N_pyr * (N_steer(N_steer - 1)/2) = 4 * (4*3/2) = 4 * 6 = 24 parameters in original PS model
    fprintf(fp,
        "Pairwise cross-correlations of the modulus of all oriented "
        "sub-bands at the same scale (size QxQ)\n");
    for (i = 0; i < N_pyr - 1; i++) { // different here
        for (l = 0; l < N_steer * N_steer * nz * nz; l++) {
            fprintf(fp, " %f", stats.cousinMagCor[i][l]);
        }
        fprintf(fp, ",\n");
    }
    
    fprintf(fp, "\n");

    /* ------------------------------------------------------------------ */
    /* Parent magnitude correlations                                      */
    /* ------------------------------------------------------------------ */
    // Right now 4 * 8 = 32 parameters (N_pyr - 2 = 4 - 2 = 2 scales, and Q = N_steer * nz = 4 * 1 = 4 orientations per scale, so QxQ = 16 values per scale, and 16 x 2 = 32 parameters)
    // Supposed to be N_steer * N_steer * (N_pyr - 1) = 4 * 4 * (4 - 1) = 4 * 4 * 3 = 48 parameters in original PS model
    fprintf(fp,
        "Cross-correlations of the modulus of all oriented sub-bands "
        "with all oriented sub-bands at the coarser scale (size QxQ)\n");
    for (i = 0; i < N_pyr - 2; i++) { // different here
        for (l = 0; l < N_steer * N_steer * nz * nz; l++) {
            fprintf(fp, " %f", stats.parentMagCor[i][l]);
        }
        fprintf(fp, ",\n");
    }
    fprintf(fp, "\n");

    /* ------------------------------------------------------------------ */
    /* Parent real correlations (Q x 2Q, one row per orientation)          */
    /* ------------------------------------------------------------------ */
    // Right now 8 * 8 = 64 parameters (N_pyr - 2 = 4 - 2 = 2 scales, and Q = N_steer * nz = 4 * 1 = 4 orientations per scale, so Qx2Q = 32 values per scale, and 16 x 2 = 64 parameters)
    // Supposed to be 2 * N_steer * N_steer * (N_pyr - 1) = 2 * 4 * 4 * (4 - 1) = 2 * 4 * 4 * 3 = 96 parameters in original PS model
    fprintf(fp,
        "Cross-correlations of the real part of each oriented sub-band "
        "with both the real and imaginary part of all phase-doubled "
        "oriented sub-bands at the next coarser scale (Q matrices of size 1x2Q, which are computed and stored in a matrix of size Qx2Q)\n");

    for (i = 0; i < N_pyr - 2; i++) { // different here
        for (l = 0; l < 2 * N_steer * N_steer * nz * nz; l++) {
            fprintf(fp, " %f", stats.parentRealCor[i][l]);
        }
        fprintf(fp, ",\n");
    }

    fclose(fp);
    printf("Statistics successfully written to %s\n", filename);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_stats_file>\n", argv[0]);
        printf("Example: %s my_image.png image_stats.txt\n", argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    // Load image using stb_image - FORCE to 1 channel (grayscale) or 3 channels (RGB)
    int nx, ny, nz_original;
    int desired_channels = 1; // Change to 3 for RGB images, or 0 for auto-detect
    
    unsigned char *image_data_u8 = stbi_load(input_filename, &nx, &ny, &nz_original, desired_channels);
    if (!image_data_u8) {
        fprintf(stderr, "Failed to load image: %s\n", input_filename);
        fprintf(stderr, "STB Error: %s\n", stbi_failure_reason());
        return 1;
    }

    int nz = (desired_channels == 0) ? nz_original : desired_channels;
    
    // If image has 4 channels and we loaded all, convert to 3
    if (nz == 4) {
        printf("Warning: Image has alpha channel. Converting RGBA to RGB...\n");
        nz = 3;
    } else if (nz == 2) {
        printf("Warning: Image has 2 channels. Using grayscale only...\n");
        nz = 1;
    }

    printf("Loaded image: %dx%dx%d (original had %d channels)\n", nx, ny, nz, nz_original);

    // Convert from uint8 to float (0-255 range to 0.0-255.0)
    float *image_data = (float*) malloc(nx * ny * nz * sizeof(float));
    for (int i = 0; i < nx * ny * nz; i++) {
        image_data[i] = (float)image_data_u8[i];
    }
    stbi_image_free(image_data_u8);

    // Create imageStruct
    imageStruct image;
    image.nx = nx;
    image.ny = ny;
    image.nz = nz;
    image.image = image_data;

    // Set up parameters
    paramsStruct params;
    params.N_pyr = 4;      // Number of pyramid scales
    params.N_steer = 4;    // Number of orientations
    params.Na = 7;         // Auto-correlation neighborhood size
    params.verbose = 1;
    params.statistics = 0;
    params.N_iteration = 0;
    params.noise = 0;
    params.cmask[0] = 1;
    params.cmask[1] = 1;
    params.cmask[2] = 1;
    params.cmask[3] = 1;
    params.edge_handling = 0;
    params.add_smooth = 0;
    params.interpWeight = -1;

    // Allocate memory for statistics
    statsStruct stats;
    allocate_stats(&stats, params, nz);

    // Perform analysis - this extracts all the statistics!
    printf("Extracting statistics from image...\n");
    analysis(&stats, image, params);

    // Write statistics to file
    printf("About to write statistics to: %s\n", output_filename);
    write_statistics_to_file(stats, params, nz, output_filename);
    
    // Verify file was created
    FILE *check = fopen(output_filename, "r");
    if (check) {
        fseek(check, 0, SEEK_END);
        long size = ftell(check);
        fclose(check);
        printf("File created successfully! Size: %ld bytes\n", size);
    } else {
        printf("ERROR: File was not created!\n");
    }

    // Clean up
    free_stats(stats, params, nz);
    free(image_data);

    printf("\nDone! All summary statistics extracted successfully.\n");
    return 0;
}