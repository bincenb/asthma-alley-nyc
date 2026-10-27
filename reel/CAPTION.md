# Instagram reel: Asthma Alley After the Toll

**Files:** `asthma-alley-reel.mp4` (1080×1920, 30 fps, 32 s, AAC at −14 LUFS) and `cover.png` (the cover frame).

## Caption (paste as is)

They said congestion pricing cleaned NYC's air. Here's what it looked like in the South Bronx 👇

🚗 11% fewer vehicles entered the toll zone in 2025
📈 But 7 of 9 MTA bridges and tunnels carried more traffic
🚚 Trucks that skip Manhattan go around it, through the Bronx
📍 Mott Haven's adult asthma ER visit rate is 3.8× the city's
🌳 It has about a third of the city's tree cover
😷 All 15 districts with the most asthma ER visits got worse NO₂ than the city in 2025
💸 7 of those 15 get none of the MTA's mitigation money

To be clear: the toll's effect on air quality was small, and most of the changes weren't statistically significant. The asthma gap is what's huge.

So we designed a fix: a green corridor on Bruckner Blvd with 262 new trees, a 2.1 km evergreen barrier, and 4 school sites shielded. It costs about $10–19M, and $24M is already set aside for Mott Haven.

Send this to someone who breathes Bronx air 📍

CUNY Datathon 2026 · "Reimagining NYC" · all data is public

#AsthmaAlley #SouthBronx #Bronx #MottHaven #NYC #CongestionPricing #AirQuality #EnvironmentalJustice #PublicHealth #ClimateJustice #UrbanPlanning #DataViz #CUNY #Datathon

## Posting tips

- **Audio:** the soundtrack is original and synthesized in `audio.py`, so it's safe to post anywhere. If you want trending-audio reach, add a trending sound in the Instagram editor at about 10–20% volume under the original. The cuts are timed at 120 BPM.
- **Cover:** upload `cover.png` as the cover so the grid shows "ASTHMA ALLEY."
- **Loop:** the last second glides back to the opening map, so the reel replays seamlessly.
- **Timing:** post weekday evenings, and pin a comment that links the full map and report.

## Rebuild

```
pip install numpy scipy pillow   # ffmpeg must be installed
python3 reel/make_reel.py        # about 6 min on 4 cores
python3 reel/make_reel.py --frames /tmp/stills 1.0,10.5,25.0   # quick stills
```
