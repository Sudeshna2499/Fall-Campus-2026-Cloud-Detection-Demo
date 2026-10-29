// On-board cloud screening demo: Arduino as the CubeSat on-board computer (OBC).
//
// A much simplified version of what Phi-Sat-1/2 do: look at each image tile on board and
// discard the cloudy ones before downlink. All pixels are synthetic.
//   1. build one 8x8-pixel tile in 4 bands (R, G, B, NIR) from a stored scene description
//   2. five features per tile: mean brightness, texture, whiteness, NDVI, mean NIR
//   3. simple rule : cloud if mean brightness > RULE_M_THR
//      learned     : tiny 5-8-3 MLP on the same five features (trained on the ground, model.h)
//   4. policy      : discard the tile only if P(cloud) > DISCARD_P, otherwise send it
//      (uncertain tiles are sent, so a doubtful decision costs bytes, not data)
//
// Serial output (115200 baud), one line per tile:
//   idx,surf,cloud_pct,truth,m,s,w,ndvi,nir,r,g,b,rule,cls,pcloud,send,raw_bytes,tx_bytes,proc_us
// Keep model.h in the same folder as this sketch.
#include <Arduino.h>
#include <math.h>
#include "model.h"

#define TW 8
#define NPIX 64
#define TILE_MS 150
#define RAW_TILE_BYTES 256
#define FLAG_BYTES 1

static const float VEG[4]   = {0.06f, 0.11f, 0.05f, 0.48f};
static const float SOIL[4]  = {0.22f, 0.17f, 0.12f, 0.30f};
static const float SEA[4]   = {0.03f, 0.05f, 0.07f, 0.015f};
static const float SAND[4]  = {0.50f, 0.44f, 0.34f, 0.50f};
static const float CLOUD[4] = {0.74f, 0.75f, 0.78f, 0.72f};
static const float SIG_SURF[3] = {0.10f, 0.03f, 0.04f};
#define SIG_CLOUD 0.07f

static uint32_t rs = 2463534242UL;
static uint32_t rawBytes = 0, txBytes = 0;
static uint16_t idx = 0;

static float urand() {                       // uniform [0,1)
  rs = rs * 1664525UL + 1013904223UL;
  return (rs >> 16) / 65536.0f;
}
static float gauss4() {                      // approx. unit Gaussian from four uniforms
  return (urand() + urand() + urand() + urand() - 2.0f) * 1.7320508f;
}
static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// build the tile pixel by pixel and accumulate the statistics (no pixel buffer needed) - to simulate what it would "receive" from the camera
static void tileFeatures(uint8_t surf, float soil, float alpha, float *f, float *mean3) {
  float base[4], spec[4];
  // parsing through 5 numbers (features) per tile to check if the tile is a cloud or not
  for (uint8_t b = 0; b < 4; b++) {
    if (surf == 0) base[b] = VEG[b] + (SOIL[b] - VEG[b]) * soil;
    else if (surf == 1) base[b] = SEA[b];
    else base[b] = SAND[b];
  }
  float cb = 0.85f + 0.25f * urand();
  for (uint8_t b = 0; b < 4; b++) spec[b] = (1.0f - alpha) * base[b] + alpha * CLOUD[b] * cb;
  float sig = (1.0f - alpha) * SIG_SURF[surf] + alpha * SIG_CLOUD;

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
  float m = sumP / NPIX;
  float var = sumP2 / NPIX - m * m;
  float s = sqrtf(var > 0 ? var : 0);
  float w = sumW / NPIX;
  float r = sumR / NPIX, nir = sumN / NPIX;
  float ndvi = (nir - r) / (nir + r + 0.01f);
  f[0] = m / 0.5f; f[1] = s / 0.08f; f[2] = w; f[3] = ndvi; f[4] = nir / 0.5f;
  mean3[0] = r; mean3[1] = sumG / NPIX; mean3[2] = sumB / NPIX;
}

// the model that will run the cloud detection
static float mlp(const float *f, uint8_t *cls) {      // returns P(cloud)
  float hid[N_HID], z[N_OUT];
  for (uint8_t h = 0; h < N_HID; h++) {
    float s = pgm_read_float(&MLP_B1[h]);
    for (uint8_t i = 0; i < N_IN; i++) s += pgm_read_float(&MLP_W1[h * N_IN + i]) * f[i];
    hid[h] = s > 0 ? s : 0;
  }
  float zmax = -1e9f;
  for (uint8_t o = 0; o < N_OUT; o++) {
    float s = pgm_read_float(&MLP_B2[o]);
    for (uint8_t h = 0; h < N_HID; h++) s += pgm_read_float(&MLP_W2[o * N_HID + h]) * hid[h];
    z[o] = s; if (s > zmax) zmax = s;
  }
  float sum = 0;
  for (uint8_t o = 0; o < N_OUT; o++) { z[o] = expf(z[o] - zmax); sum += z[o]; }
  uint8_t best = 0;
  for (uint8_t o = 0; o < N_OUT; o++) { z[o] /= sum; if (z[o] > z[best]) best = o; }
  *cls = best;
  return z[2];
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  Serial.println(F("# idx,surf,cloud_pct,truth,m,s,w,ndvi,nir,r,g,b,rule,cls,pcloud,send,raw_bytes,tx_bytes,proc_us"));
}

void loop() {
  uint8_t surf = pgm_read_byte(&SC_SURF[idx]);
  float soil = pgm_read_byte(&SC_SOIL[idx]) / 100.0f;
  uint8_t cpct = pgm_read_byte(&SC_CLOUD[idx]);
  float alpha = cpct / 100.0f;
  uint8_t truth = cpct >= 50 ? 2 : (surf == 1 ? 1 : 0);

  float f[5], rgb[3];
  unsigned long t0 = micros();
  tileFeatures(surf, soil, alpha, f, rgb);
  uint8_t cls;
  float pc = mlp(f, &cls);
  unsigned long dt = micros() - t0;

  uint8_t rule = (f[0] * 0.5f > RULE_M_THR) ? 2 : 0;
  uint8_t send = (pc > DISCARD_P) ? 0 : 1;
  rawBytes += RAW_TILE_BYTES;
  txBytes += FLAG_BYTES + (send ? RAW_TILE_BYTES : 0);

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

  idx++;
  if (idx >= SCENE_TILES) { idx = 0; rawBytes = 0; txBytes = 0; }
  delay(TILE_MS);
}