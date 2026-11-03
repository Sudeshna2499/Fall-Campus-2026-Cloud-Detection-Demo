## On-board cloud screening demo kit

An Arduino plays a CubeSat on-board computer (OBC) that screens image tiles for cloud and discards the cloudy ones before downlink. It is a toy version of what ESA's Phi-Sat-1 and Phi-Sat-2 do on board (cloud detection), built for the seminar "Intelligent Data Systems and AI for Autonomous Space Technology". The pixels are synthetic, and ESA does not publish the details used here, so this is the same idea, not a copy of their method.


A scene is 16 x 8 tiles; a tile is 8 x 8 pixels in four bands (red, green, blue, near-infrared), generated from a surface type (land, sea, bright sand) plus a cloud fraction.

## On the board, per tile:

Build the tile and compute five features: mean brightness, texture (spread of brightness), whiteness (how grey the colour is), NDVI (vegetation index from near-infrared and red) and mean near-infrared level.


Simple rule: cloud if mean brightness is above a fitted threshold.


Learned: a tiny 5-8-3 neural network (75 parameters, 300 bytes) on the same five features. Classes: clear land, clear sea, cloud. Trained on the ground and uplinked as model.h.


Policy: discard a tile only if P(cloud) > 0.6; otherwise send it (tiles with P between 0.35 and 0.6 are sent and marked unsure). A doubtful decision costs bytes, not data.


## Steps

python3 train_model.py (outputs are included; rerun only if you change cloud_sim.py)


Open obc_cloud/obc_cloud.ino in the Arduino IDE, select your board, upload (keep model.h beside it).


Close the Serial Monitor, then python3 live_plot.py --port /dev/ttyACM0 (Windows: COM3).


No board: python3 live_plot.py --sim. Headless snapshot: python3 live_plot.py --sim --save demo.png.


Results figure. With the board: python3 live_plot.py --port COM3 --log run.csv, let it run for a few scenes (about 20 s each), close the window, then python3 plot_results.py --csv run.csv --out results.png. Without hardware: python3 plot_results.py --sim --scenes 100 --out results.png.


Serial format (115200 baud, one line per tile, about 7 per second; one scene takes about 20 s and then repeats): 

idx,surf,cloud_pct,truth,m,s,w,ndvi,nir,r,g,b,rule,cls,pcloud,send,raw_bytes,tx_bytes,proc_us Classes: 0 clear land, 1 clear sea, 2 cloud. A tile counts as cloud when its cloud fraction is 50% or more.



## What was verified

The sketch compiled with g++ against a small Arduino shim and ran over six scenes. Its decisions behave like the Python simulation: all cloud discarded, the rule lost 9 clear tiles and the network none (of 492).


Python results (results.png: 100 simulated scenes; earlier check: 300 random scenes): cloud is about 36% of tiles; cloud discarded 98.3% (rule) vs 99.2% (network); clear tiles wrongly discarded 1.6% vs 0.7%; bytes to downlink 64% of raw (a 36% saving, bounded by the cloud share).


Not tested: a real board, serial plotting with hardware. The sketch needs little RAM (no pixel buffer, tile statistics are accumulated on the fly), but check in the Arduino IDE. Floating-point maths is slow on an 8-bit Uno; a Nano Every, Nano 33, ESP32 or Teensy is safer.
