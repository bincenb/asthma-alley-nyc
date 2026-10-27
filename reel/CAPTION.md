# Instagram Reel: posting kit

**Video:** `asthma-alley-reel.mp4` (1080×1920, 30 fps, 38 s, soundtrack included)
**Cover:** `cover.png` (upload as the custom cover so the grid shows the hook)

## Caption

NYC's congestion toll "cut air pollution 22%." We checked the data. 🫁

That number is real, but it's each day's peak reading, inside the toll zone, over the first 5 months. Across the whole city, over the whole year, average PM2.5 went from 6.26 to 6.29. Basically flat.

So where did the traffic go? Small shifts so far, toward highways drivers use to dodge the toll. Truck crossings rose on the Whitestone (+4.1%) and Throgs Neck (+2.1%). The toll-free FDR got 8.9% more traffic.

All 15 districts with NYC's worst asthma rates saw NO₂ go up more than the city's in 2025. In the 3 years before, only 3 to 10 of them did. 7 of the 15 get $0 of the MTA's $105M mitigation fund.

Mott Haven ranks #1 of 59 on our priority index: 3.8× the city's adult asthma ER rate and 7.1% tree canopy (city: 21.1%). Our fix is a green wall on Bruckner Blvd: 262 new trees and a 2.1 km evergreen barrier by four schools. It would cost $10–19M, and the MTA already set aside $24M for this area.

This doesn't prove the toll caused it. It's a reason to measure, and to plant.

📍 Full interactive map: link in bio
Every number comes from public data (NYC Health Dept., MTA, NYCCAS). Sources are in the report.

#NYC #SouthBronx #MottHaven #CongestionPricing #AirQuality #Asthma #EnvironmentalJustice #UrbanPlanning #DataViz #CUNYDatathon

## Posting notes
- Post 6–9 p.m. ET on a weekday. Tag @southbronxunite and your CUNY school's account, and add the "New York, New York" location.
- The soundtrack is synthesized for this video, so you own it. To use a trending sound instead, set the original audio to 0 in the editor. Every key point is on screen, so it still works with the sound off.
- Pin a comment: "What's the air like on your block? 👇" Comments push reach more than likes do.
- The 3 s hook ("…right?") is designed to stop the scroll. Don't trim the start.

## Editing the video
`reel.html` is the source. Each frame is `render(t)`, and `geo.js` holds the real community-district shapes from `part1-map.html`.
```
python3 soundtrack.py soundtrack.wav
node render.js asthma-alley-reel.mp4 soundtrack.wav   # about 75 s
node render.js --stills 2.9,17.5 .                    # preview frames
```
