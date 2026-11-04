// On-board cloud screening demo: Arduino as the CubeSat on-board computer (OBC).
//
// A much simplified version of what Phi-Sat-1/2 do: look at each image tile on board and
// discard the cloudy ones before downlink. All pixels are synthetic.
//   1. build one 8x8-pixel tile in 4 bands (R, G, B, NIR) from a stored scene description
//   2. five features per tile: mean brightness, texture, whiteness, NDVI, mean NIR
//   3. simple rule : cloud if mean brightness > RULE_M_THR
//      learned    : tiny 5-8-3 MLP on the same five features (trained on the ground, model.h)
//   4. policy      : discard the tile only if P(cloud) > DISCARD_P, otherwise send it
//      (uncertain tiles are sent, so a doubtful decision costs bytes, not data)
//
// Serial output (115200 baud), one line per tile:
//   idx,surf,cloud_pct,truth,m,s,w,ndvi,nir,r,g,b,rule,cls,pcloud,send,raw_bytes,tx_bytes,proc_us
// Keep model.h in the same folder as this sketch.
#include <Arduino.h>
#include <math.h>
#include "model.h"   // trained weights (MLP_*), RULE_M_THR, DISCARD_P, and the stored scene (SC_*)

// ---- settings ---------------------------------------------------------------
#define TW 8                // tile width in pixels (tiles are 8 x 8)
#define NPIX 64             // pixels per tile
#define TILE_MS 150         // pause between tiles, so the plot fills in at a watchable pace
#define RAW_TILE_BYTES 256  // one raw tile = 64 pixels x 4 bands x 1 byte
#define FLAG_BYTES 1        // 1 byte per tile telling the ground whether the tile follows

// ---- made-up spectra: reflectance in the bands {R, G, B, NIR} ----------------

static const float VEG[4]   = {0.06f, 0.11f, 0.05f, 0.48f};
static const float SOIL[4]  = {0.22f, 0.17f, 0.12f, 0.30f};
static const float SEA[4]   = {0.03f, 0.05f, 0.07f, 0.015f};
static const float SAND[4]  = {0.50f, 0.44f, 0.34f, 0.50f};
static const float CLOUD[4] = {0.74f, 0.75f, 0.78f, 0.72f};
// pixel-to-pixel noise (texture) for land, sea, sand; clouds have their own
static const float SIG_SURF[3] = {0.10f, 0.03f, 0.04f};
#define SIG_CLOUD 0.07f

// ---- state ------------------------------------------------------------------
static uint32_t rs = 2463534242UL;                 // seed of the pseudo-random generator
static uint32_t rawBytes = 0, txBytes = 0;         // running byte counts for the current scene
static uint16_t idx = 0;                           // index of the tile being processed

// ---- small helpers ------------------------------------------------------------
static float urand() {                       // uniform [0,1)
  rs = rs * 1664525UL + 1013904223UL;
  return (rs >> 16) / 65536.0f;
}
static float gauss4() {                      // approx. unit Gaussian from four uniforms
  return (urand() + urand() + urand() + urand() - 2.0f) * 1.7320508f;
}
static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Step 1 + 2: build the tile pixel by pixel and accumulate the statistics. The 64 pixels are
// never stored (the Uno has only 2 KB of RAM): running sums are enough for all five features.
//   surf  = 0 land, 1 sea, 2 sand;  soil = vegetation/soil mix of land;  alpha = cloud fraction 0..1
//   f[]   = the five features, scaled for the network;  mean3[] = mean R,G,B (only for the picture)
static void tileFeatures(uint8_t surf, float soil, float alpha, float *f, float *mean3) {
  float base[4], spec[4];
  // colour of the ground without cloud
  for (uint8_t b = 0; b < 4; b++) {
    if (surf == 0) base[b] = VEG[b] + (SOIL[b] - VEG[b]) * soil;
    else if (surf == 1) base[b] = SEA[b];
    else base[b] = SAND[b];
  }
  // mix ground and cloud: the more cloud (alpha), the whiter the tile
  float cb = 0.85f + 0.25f * urand();
  for (uint8_t b = 0; b < 4; b++) spec[b] = (1.0f - alpha) * base[b] + alpha * CLOUD[b] * cb;
  float sig = (1.0f - alpha) * SIG_SURF[surf] + alpha * SIG_CLOUD;

  // loop over the 64 pixels: sums for brightness, texture, whiteness, colour and NIR
  float sumP = 0, sumP2 = 0, sumW = 0, sumR = 0, sumG = 0, sumB = 0, sumN = 0;
  for (uint8_t i = 0; i < NPIX; i++) {
    float g = gauss4();
    float px[4];
    for (uint8_t b = 0; b < 4; b++)
      px[b] = clampf(spec[b] * (1.0f + sig * g) * (1.0f + 0.02f * gauss4()), 0.0f, 1.0f);
    float p = (px[0] + px[1] + px[2]) / 3.0f;
    float mx = fmaxf(px[0], fmaxf(px[1], px[2])), mn = fminf(px[0], fminf(px[1], px[2]));
    sumP += p; sumP2 += p * p; sumW += (mx - mn) / (p + 0.02f);
    sumR += px[0]; sumG += px[1]; sumB += px[2]; sumN += px[3];
  }
  // turn the sums into the five features
  float m = sumP / NPIX;                      // mean brightness
  float var = sumP2 / NPIX - m * m;
  float s = sqrtf(var > 0 ? var : 0);          // texture (spread of brightness)
  float w = sumW / NPIX;                      // whiteness (low = grey/white like cloud)
  float r = sumR / NPIX, nir = sumN / NPIX;
  float ndvi = (nir - r) / (nir + r + 0.01f); // vegetation index
  // divide by typical values so all features are of similar size (the network was trained this way)
  f[0] = m / 0.5f; f[1] = s / 0.08f; f[2] = w; f[3] = ndvi; f[4] = nir / 0.5f;
  mean3[0] = r; mean3[1] = sumG / NPIX; mean3[2] = sumB / NPIX;
}

// The tiny neural network: 5 inputs -> 8 hidden neurons (ReLU) -> 3 outputs (softmax).
// Weights live in flash (PROGMEM) and are read with pgm_read_float. Returns P(cloud).
static float mlp(const float *f, uint8_t *cls) {
  float hid[N_HID], z[N_OUT];
  // hidden layer: weighted sum of the inputs plus bias, then ReLU (negative -> 0)
  for (uint8_t h = 0; h < N_HID; h++) {
    float s = pgm_read_float(&MLP_B1[h]);
    for (uint8_t i = 0; i < N_IN; i++) s += pgm_read_float(&MLP_W1[h * N_IN + i]) * f[i];
    hid[h] = s > 0 ? s : 0;
  }
  // output layer: one score per class (land, sea, cloud)
  float zmax = -1e9f;
  for (uint8_t o = 0; o < N_OUT; o++) {
    float s = pgm_read_float(&MLP_B2[o]);
    for (uint8_t h = 0; h < N_HID; h++) s += pgm_read_float(&MLP_W2[o * N_HID + h]) * hid[h];
    z[o] = s; if (s > zmax) zmax = s;
  }
  // softmax: turn the three scores into probabilities that add up to 1
  float sum = 0;
  for (uint8_t o = 0; o < N_OUT; o++) { z[o] = expf(z[o] - zmax); sum += z[o]; }
  uint8_t best = 0;
  for (uint8_t o = 0; o < N_OUT; o++) { z[o] /= sum; if (z[o] > z[best]) best = o; }
  *cls = best;        // most likely class
  return z[2];        // probability of cloud
}

void setup() {
  Serial.begin(115200);                       // the Serial Monitor must be set to the same speed
  // wait for the serial port (up to 3 s) so no lines are lost
  while (!Serial && millis() < 3000) {}
  Serial.println(F("# idx,surf,cloud_pct,truth,m,s,w,ndvi,nir,r,g,b,rule,cls,pcloud,send,raw_bytes,tx_bytes,proc_us"));
}

// One pass of loop() = one tile: read its description, measure it, classify it, print one line.
void loop() {
  // 1. the stored scene description for this tile (surface, soil mix, cloud %), read from flash
  uint8_t surf = pgm_read_byte(&SC_SURF[idx]);
  float soil = pgm_read_byte(&SC_SOIL[idx]) / 100.0f;
  uint8_t cpct = pgm_read_byte(&SC_CLOUD[idx]);
  float alpha = cpct / 100.0f;
  // ground truth (not used for the decision, only for scoring): 2 = cloud when cloud >= 50%
  uint8_t truth = cpct >= 50 ? 2 : (surf == 1 ? 1 : 0);

  // 2. features and network, timed in microseconds
  float f[5], rgb[3];
  unsigned long t0 = micros();
  tileFeatures(surf, soil, alpha, f, rgb);
  uint8_t cls;
  float pc = mlp(f, &cls);
  unsigned long dt = micros() - t0;

  // 3. decisions: the simple brightness rule, and the network policy
  //    (discard only if P(cloud) > DISCARD_P; unsure tiles are sent)
  uint8_t rule = (f[0] * 0.5f > RULE_M_THR) ? 2 : 0;
  uint8_t send = (pc > DISCARD_P) ? 0 : 1;
  // byte accounting: raw = every tile; sent = 1 flag byte, plus the tile if it is kept
  rawBytes += RAW_TILE_BYTES;
  txBytes += FLAG_BYTES + (send ? RAW_TILE_BYTES : 0);

  // 4. one CSV line per tile, in the order of the header printed in setup()
  Serial.print(idx); Serial.print(',');
  Serial.print(surf); Serial.print(',');
  Serial.print(cpct); Serial.print(',');
  Serial.print(truth); Serial.print(',');
  Serial.print(f[0] * 0.5f, 4); Serial.print(',');
  Serial.print(f[1] * 0.08f, 4); Serial.print(',');
  Serial.print(f[2], 4); Serial.print(',');
  Serial.print(f[3], 4); Serial.print(',');
  Serial.print(f[4] * 0.5f, 4); Serial.print(',');
  Serial.print(rgb[0], 4); Serial.print(',');
  Serial.print(rgb[1], 4); Serial.print(',');
  Serial.print(rgb[2], 4); Serial.print(',');
  Serial.print(rule); Serial.print(',');
  Serial.print(cls); Serial.print(',');
  Serial.print(pc, 4); Serial.print(',');
  Serial.print(send); Serial.print(',');
  Serial.print(rawBytes); Serial.print(',');
  Serial.print(txBytes); Serial.print(',');
  Serial.print(dt);
  Serial.println();

  // next tile; after the last one, start a new scene
  idx++;
  if (idx >= SCENE_TILES) { idx = 0; rawBytes = 0; txBytes = 0; }
  delay(TILE_MS);
}
