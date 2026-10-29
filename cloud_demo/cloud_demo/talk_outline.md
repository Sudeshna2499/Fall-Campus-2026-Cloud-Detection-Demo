# Seminar script: Intelligent Data Systems and AI for Autonomous Space Technology

Fall Campus 2026. Two parts of 50 minutes. Slides: Part 1 (14 slides) and Part 2 (14 slides), speaker notes on each. Demo: on-board cloud screening on an Arduino (see README).

**Through-line:** satellite data is valuable; a satellite collects more than it can send; so a small on-board model should decide what is worth sending, and we must be able to trust that decision.
**Opening question (return to it at the end):** if you could send only a handful of events per pass, which would you send?

## Part 1: Why the data matters, and why it can't all come home (50 min)

| Min | Slide | Say |
|---|---|---|
| 0-4 | Cover, hook | Ask the opening question; take 2-3 answers. Note that "biggest" and "strangest" are both biased picks. |
| 4-9 | Why satellite data matters | Weather and climate (27 of 38 Essential Climate Variables measurable from space, ESA CCI), disaster response (International Charter), farming and shipping, and the satellites' own environment. |
| 9-12 | Space weather | Starlink, Feb 2022: 38 of 49 lost after a minor geomagnetic storm raised drag (CIRES). Measuring the space environment has a real cost when it fails. |
| 12-17 | How data is collected | Sense, digitise, process and store, packetise, downlink. Payload vs housekeeping data. |
| 17-20 | Small platforms | CubeSat subsystems; the OBC as the place intelligence has to live. |
| 20-27 | Downlink bottleneck | Sentinel-2: about 2.7 TB/day generated vs about 1 TB/day downloadable; LoRa 5-50 kbps, S-band up to 2 Mbps; buffered data can wait up to 30 days. All from the OrbitChain preprint: say so. Don't quote minutes per pass. |
| 27-31 | Send everything or decide on board | Four jobs: screen, detect, prioritise, compress. Risk: a discarded observation is gone. |
| 31-35 | Missions | OPS-SAT (Cyclone V SoC, Linux, uploadable software, ended May 2024), Φsat-1 (Myriad 2, cloud screening, >30% of Sentinel-2 images cloudy), Φsat-2 (apps at launch and uploaded later). |
| 35-47 | Live demo | On-board cloud screening, a toy version of what Φsat does. See demo script below; it is a worked example, not a flight design. |
| 47-50 | Recap | Four takeaways; bridge to Part 2. |

### Demo script (about 12 min)
1. Set it up: "Φsat-1 and Φsat-2 detect clouds on board. Here is a toy version: an Arduino, made-up image tiles, five numbers per tile." Name the five features: brightness, texture, whiteness, vegetation index (NDVI), near-infrared level.
2. Run `python3 live_plot.py --port /dev/ttyACM0` from the `cloud_demo` folder (fallback: `--sim`). One 16 x 8 scene takes about 20 s; let two play.
3. Point out: the true-colour scene fills in; grey tiles are cloud that is discarded; green tiles are clear and sent. The bytes line grows more slowly than "send every tile".
4. Find the red tiles on the rule's map: clear ground (bright sand, thin haze) thrown away because the rule only looks at brightness. Compare with the network's map, which has fewer. Point to the rings: unsure tiles are sent, not discarded.
5. Scoreboard (300 simulated scenes): cloud discarded 98.3% (rule) vs 99.2% (network); clear tiles wrongly discarded 1.6% vs 0.7%; bytes 64% of raw. Say it plainly: a modest gain, synthetic pixels, bright sand added on purpose, and a simple tree might do as well.
6. Ask: even a perfect cloud filter saves only the cloud share (about 36% here). Is that enough? (Sets up prioritising and compressing, and Part 2.)

## Part 2: Putting ML on the on-board computer (50 min)

| Min | Slide | Say |
|---|---|---|
| 0-4 | Cover, bridge | Recap in two sentences; the model decides what fits through a small pipe. |
| 4-10 | Four jobs for a model | Screen, detect, prioritise, compress, with Φsat-1/2 examples. Choose by the cost of a mistake: false alarm wastes bytes, miss loses data for good. |
| 10-16 | Data to model in orbit | Collect, train on the ground, shrink, test on the board, uplink, monitor. Representative labelled data is the hard part. |
| 16-21 | Match model to board | Rules, small models on features (the 75-parameter, 300-byte demo network that screens cloud), deep networks on images. |
| 21-28 | Where it can run | Cortex-M, rad-hard CPU, SoC with FPGA (OPS-SAT), VPU (Φsat-1). Trade-offs, no winner on every axis. Don't name a Φsat-2 processor. |
| 28-33 | Radiation | Single-event upsets; silent errors in inputs, weights, code and decision logs. |
| 33-37 | Standard protections | ECC, scrubbing, triple redundancy, watchdog, plausibility checks, confidence fallback that errs toward sending. |
| 37-41 | Trust | Test before launch on flight-like data; downlink a random sample of discarded data in flight; log decisions; watch for drift. |
| 41-44 | Updates after launch | Φsat-2 apps uploaded later, OPS-SAT uploadable images. Keep the old model, authenticate, upload small. |
| 44-47 | Open questions | Training data for rare events, how much validation is enough, efficiency, drift, radiation tolerance of networks, constellations. |
| 47-50 | If you want to try this, discussion | Closing question: what would you let a satellite throw away, and what evidence would you need first? |

## Things to verify before presenting
- Open every source on the reference slides and add URLs and access dates.
- The OrbitChain figures are from an unreviewed preprint; consider replacing them with an official Sentinel-2 data-rate source.
- Radiation and processor slides are general knowledge; add one authoritative source and check RAD750/LEON wording against datasheets.
- Artemis II, if mentioned, is recent history: launched 1 Apr 2026, splashdown 10 Apr 2026.
- Run the demo once on your board before the day; the sketch has only been compiled on a PC.
