#!/usr/bin/env python3
"""
OptiCrop — Inference Module
=====================================
Loads trained model artifacts once at import time and exposes a clean
predict_crop() function for use by the Flask routes.

Crop information database covers all 22 crops in the dataset.
"""

import json
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Optional

from config import MODEL_PATH, METRICS_PATH, FEATURE_COLS

logger = logging.getLogger(__name__)

# ─── Crop Information Database ────────────────────────────────────────────────
CROP_INFO: dict = {
    "rice": {
        "display":     "Rice",
        "emoji":       "🌾",
        "color":       "#22c55e",
        "description": (
            "Rice (Oryza sativa) is the world's most important staple food crop, "
            "feeding over half of the global population. It thrives in waterlogged, "
            "warm conditions and is grown in flooded paddies across tropical Asia."
        ),
        "season":            "Kharif (June – November)",
        "soil_type":         "Clay or heavy loam with high water retention",
        "ideal_temp":        "20°C – 35°C",
        "ideal_rainfall":    "150 – 300 mm/month",
        "ideal_ph":          "5.5 – 7.0",
        "ideal_N":           "80 – 120 kg/ha",
        "ideal_P":           "40 – 60 kg/ha",
        "ideal_K":           "40 – 60 kg/ha",
        "harvest_duration":  "90 – 180 days",
        "water_requirement": "High (1,200 – 2,000 mm/year)",
        "tips": [
            "Maintain 2–5 cm of flooded water during the vegetative stage.",
            "Apply nitrogen in split doses: at basal, tillering, and panicle initiation.",
            "Control weeds early — competition reduces yield by up to 40%.",
            "Monitor for rice blast disease and brown planthopper.",
            "Drain fields 2–3 weeks before harvest for uniform ripening.",
        ],
    },
    "maize": {
        "display":     "Maize",
        "emoji":       "🌽",
        "color":       "#eab308",
        "description": (
            "Maize (Zea mays) is a versatile cereal crop used for food, animal feed, "
            "and industrial biofuel. It is one of the highest-yielding grain crops "
            "and adapts to a wide range of climates worldwide."
        ),
        "season":            "Kharif (June – Sept) / Rabi (Nov – Mar)",
        "soil_type":         "Well-drained loamy or sandy loam",
        "ideal_temp":        "21°C – 30°C",
        "ideal_rainfall":    "50 – 150 mm/month",
        "ideal_ph":          "5.8 – 7.0",
        "ideal_N":           "120 – 150 kg/ha",
        "ideal_P":           "60 – 80 kg/ha",
        "ideal_K":           "40 – 60 kg/ha",
        "harvest_duration":  "60 – 100 days",
        "water_requirement": "Moderate (500 – 800 mm/year)",
        "tips": [
            "Ensure proper drainage — maize cannot tolerate waterlogging.",
            "Apply nitrogen in split doses (sowing, knee-high, and tasseling stages).",
            "Maintain row spacing of 60–75 cm for optimal canopy development.",
            "Scout for fall armyworm using pheromone traps from early vegetative stage.",
            "Harvest when husk turns brown and kernels are firm and fully dented.",
        ],
    },
    "chickpea": {
        "display":     "Chickpea",
        "emoji":       "🫘",
        "color":       "#f97316",
        "description": (
            "Chickpea (Cicer arietinum) is a drought-tolerant pulse crop rich in protein "
            "and dietary fiber. It is a nitrogen-fixing legume that naturally improves "
            "soil health and is a vital crop for food security in semi-arid regions."
        ),
        "season":            "Rabi (October – March)",
        "soil_type":         "Light loamy or sandy loam, well-drained",
        "ideal_temp":        "15°C – 25°C",
        "ideal_rainfall":    "60 – 90 mm during season",
        "ideal_ph":          "6.0 – 8.0",
        "ideal_N":           "20 – 40 kg/ha (nitrogen-fixing)",
        "ideal_P":           "40 – 60 kg/ha",
        "ideal_K":           "20 – 30 kg/ha",
        "harvest_duration":  "90 – 120 days",
        "water_requirement": "Low (300 – 500 mm/year)",
        "tips": [
            "Inoculate seeds with Rhizobium before sowing to enhance nitrogen fixation.",
            "Avoid excess moisture — chickpea is highly sensitive to waterlogging.",
            "Apply pre-emergent herbicides to manage Orobanche (broomrape) weed.",
            "Monitor for pod borer (Helicoverpa armigera) at flowering stage.",
            "Harvest when lower leaves and pods turn yellow and seeds rattle in pods.",
        ],
    },
    "kidneybeans": {
        "display":     "Kidney Beans",
        "emoji":       "🫘",
        "color":       "#dc2626",
        "description": (
            "Kidney beans (Phaseolus vulgaris) are high-protein legumes named for their "
            "distinctive kidney shape. Popular across tropical and subtropical regions, "
            "they fix atmospheric nitrogen and are a global dietary staple."
        ),
        "season":            "Kharif (June – September)",
        "soil_type":         "Fertile, well-drained loamy soil",
        "ideal_temp":        "15°C – 27°C",
        "ideal_rainfall":    "300 – 400 mm/season",
        "ideal_ph":          "6.0 – 7.5",
        "ideal_N":           "20 – 40 kg/ha",
        "ideal_P":           "60 – 80 kg/ha",
        "ideal_K":           "40 – 60 kg/ha",
        "harvest_duration":  "70 – 100 days",
        "water_requirement": "Moderate (300 – 500 mm/year)",
        "tips": [
            "Plant after the last frost — kidney beans are frost-sensitive.",
            "Avoid overwatering, which leads to root rot and fungal infections.",
            "Rhizobium inoculation reduces the need for synthetic nitrogen fertilizer.",
            "Control bean fly and aphids aggressively in the early vegetative stage.",
            "Harvest when pods are fully developed, firm, and starting to dry.",
        ],
    },
    "pigeonpeas": {
        "display":     "Pigeon Peas",
        "emoji":       "🌿",
        "color":       "#16a34a",
        "description": (
            "Pigeon peas (Cajanus cajan) are drought-resistant legumes widely grown "
            "across tropical Asia and Africa. They provide protein-rich food and "
            "significantly improve soil fertility through biological nitrogen fixation."
        ),
        "season":            "Kharif (June – December)",
        "soil_type":         "Well-drained sandy loam or loam soil",
        "ideal_temp":        "18°C – 30°C",
        "ideal_rainfall":    "600 – 1,000 mm/year",
        "ideal_ph":          "5.0 – 7.0",
        "ideal_N":           "20 – 30 kg/ha",
        "ideal_P":           "50 – 60 kg/ha",
        "ideal_K":           "20 – 30 kg/ha",
        "harvest_duration":  "120 – 200 days",
        "water_requirement": "Low–Moderate (600 – 1,000 mm/year)",
        "tips": [
            "Intercrop with cereals like sorghum for improved land productivity.",
            "Deep root system makes it highly drought-tolerant once established.",
            "Monitor for pod borer (Helicoverpa) and Fusarium wilt disease.",
            "Avoid waterlogged soils — root rot is a major risk.",
            "Stagger harvests as pods mature at different times within a season.",
        ],
    },
    "mothbeans": {
        "display":     "Moth Beans",
        "emoji":       "🌱",
        "color":       "#a16207",
        "description": (
            "Moth beans (Vigna aconitifolia), also known as mat beans, are extremely "
            "drought-resistant legumes ideal for hot, arid conditions. They grow "
            "well in marginal lands unsuitable for most other crops."
        ),
        "season":            "Kharif (June – September)",
        "soil_type":         "Sandy loam or loamy sand, well-drained",
        "ideal_temp":        "25°C – 35°C",
        "ideal_rainfall":    "150 – 250 mm/season",
        "ideal_ph":          "7.0 – 8.5",
        "ideal_N":           "15 – 25 kg/ha",
        "ideal_P":           "40 – 50 kg/ha",
        "ideal_K":           "20 – 30 kg/ha",
        "harvest_duration":  "65 – 100 days",
        "water_requirement": "Very Low (150 – 250 mm/year)",
        "tips": [
            "Ideal for arid and semi-arid regions with minimal or no irrigation.",
            "No pre-sowing irrigation is required in sandy soils.",
            "Improves soil fertility naturally through nitrogen fixation.",
            "Harvest in the early morning to avoid pod shattering from heat.",
            "Store dried seeds in cool, airtight containers to prevent weevil damage.",
        ],
    },
    "mungbean": {
        "display":     "Mung Bean",
        "emoji":       "🌱",
        "color":       "#65a30d",
        "description": (
            "Mung bean (Vigna radiata) is a warm-season legume widely grown for its "
            "nutritious seeds and sprouts. One of the most important pulse crops in "
            "South and Southeast Asia, it is prized for rapid maturation and digestibility."
        ),
        "season":            "Kharif (Jun – Sep) / Zaid (Mar – Jun)",
        "soil_type":         "Light sandy loam to clay loam, well-drained",
        "ideal_temp":        "25°C – 35°C",
        "ideal_rainfall":    "60 – 70 mm/month",
        "ideal_ph":          "6.2 – 7.2",
        "ideal_N":           "20 – 25 kg/ha",
        "ideal_P":           "40 – 50 kg/ha",
        "ideal_K":           "30 – 40 kg/ha",
        "harvest_duration":  "60 – 90 days",
        "water_requirement": "Low–Moderate (350 – 500 mm/year)",
        "tips": [
            "Short-duration crop — ideal for fitting into multiple-cropping systems.",
            "Sensitive to frost and prolonged waterlogging.",
            "Inoculate seeds with Rhizobium for enhanced nitrogen fixation.",
            "Monitor for yellow mosaic virus transmitted by whitefly.",
            "Harvest when 70–80% of pods turn black.",
        ],
    },
    "blackgram": {
        "display":     "Black Gram",
        "emoji":       "⚫",
        "color":       "#374151",
        "description": (
            "Black gram (Vigna mungo) is a high-protein legume used extensively in "
            "South Asian cuisines — dal, idli, and dosa. It has excellent nitrogen-fixing "
            "ability and grows well in both Kharif and Rabi seasons."
        ),
        "season":            "Kharif (Jun – Sep) / Rabi (Oct – Mar)",
        "soil_type":         "Well-drained loamy or clay loam soil",
        "ideal_temp":        "25°C – 35°C",
        "ideal_rainfall":    "70 – 100 mm/month",
        "ideal_ph":          "6.0 – 7.5",
        "ideal_N":           "20 – 30 kg/ha",
        "ideal_P":           "40 – 50 kg/ha",
        "ideal_K":           "30 – 40 kg/ha",
        "harvest_duration":  "60 – 90 days",
        "water_requirement": "Low–Moderate (300 – 500 mm/year)",
        "tips": [
            "Does not require heavy irrigation — rain-fed cultivation is effective.",
            "Apply Rhizobium inoculant for nitrogen fixation to cut fertilizer costs.",
            "Control yellow mosaic virus by managing whitefly vector populations.",
            "Avoid heavy clay soils, which cause root and stem rot.",
            "Harvest before full maturity to prevent pod shattering losses.",
        ],
    },
    "lentil": {
        "display":     "Lentil",
        "emoji":       "🫘",
        "color":       "#b45309",
        "description": (
            "Lentils (Lens culinaris) are among the oldest cultivated crops, prized "
            "for their high protein and fiber content. A cool-season legume grown "
            "across South Asia, the Middle East, and the Mediterranean."
        ),
        "season":            "Rabi (October – March)",
        "soil_type":         "Well-drained sandy loam to clay loam",
        "ideal_temp":        "10°C – 25°C",
        "ideal_rainfall":    "40 – 75 mm/month",
        "ideal_ph":          "6.0 – 8.0",
        "ideal_N":           "15 – 20 kg/ha",
        "ideal_P":           "40 – 60 kg/ha",
        "ideal_K":           "20 – 30 kg/ha",
        "harvest_duration":  "80 – 120 days",
        "water_requirement": "Low (250 – 400 mm/year)",
        "tips": [
            "Cool-weather crop — frost after germination can severely damage seedlings.",
            "Inoculate with Rhizobium for optimal biological nitrogen fixation.",
            "Light irrigation at the pod development stage is critical for yield.",
            "Monitor for aphids and Botrytis blight in humid, cool conditions.",
            "Harvest when lower pods turn yellow and seeds rattle inside pods.",
        ],
    },
    "pomegranate": {
        "display":     "Pomegranate",
        "emoji":       "🍎",
        "color":       "#be123c",
        "description": (
            "Pomegranate (Punica granatum) is a drought-tolerant fruit crop known for its "
            "jewel-like arils and antioxidant-rich juice. Commercially valuable globally, "
            "it thrives in hot, dry climates with cool winters."
        ),
        "season":            "Year-round; harvested June – February",
        "soil_type":         "Deep, well-drained loamy or sandy loam",
        "ideal_temp":        "25°C – 35°C",
        "ideal_rainfall":    "500 – 800 mm/year",
        "ideal_ph":          "5.5 – 7.5",
        "ideal_N":           "50 – 80 kg/ha",
        "ideal_P":           "30 – 60 kg/ha",
        "ideal_K":           "50 – 80 kg/ha",
        "harvest_duration":  "5 – 7 months after flowering",
        "water_requirement": "Low–Moderate (500 – 800 mm/year)",
        "tips": [
            "Highly drought-tolerant once established — reduce irrigation after establishment.",
            "Drip irrigation is recommended for water efficiency and reduced disease.",
            "Prune to maintain a multi-stem open-center form for light penetration.",
            "Monitor for bacterial blight and fruit borers at fruit development stage.",
            "Harvest when the fruit produces a metallic sound when tapped.",
        ],
    },
    "banana": {
        "display":     "Banana",
        "emoji":       "🍌",
        "color":       "#ca8a04",
        "description": (
            "Banana (Musa spp.) is one of the world's most traded tropical fruits and "
            "a critical food security crop. A high-yielding, high-humidity plant that "
            "provides year-round production and significant economic value."
        ),
        "season":            "Year-round (tropical)",
        "soil_type":         "Deep, well-drained loamy or clay loam, rich in organic matter",
        "ideal_temp":        "26°C – 32°C",
        "ideal_rainfall":    "100 – 180 mm/month",
        "ideal_ph":          "5.5 – 7.0",
        "ideal_N":           "200 – 250 kg/ha",
        "ideal_P":           "30 – 50 kg/ha",
        "ideal_K":           "300 – 400 kg/ha",
        "harvest_duration":  "10 – 15 months",
        "water_requirement": "High (1,200 – 2,200 mm/year)",
        "tips": [
            "Requires well-distributed rainfall or irrigation throughout the growing period.",
            "Provide wind protection — banana plants topple easily in strong wind.",
            "Remove excess suckers; maintain 1–2 healthy followers per stool.",
            "Monitor for Fusarium wilt (Panama disease) — use resistant varieties.",
            "Harvest when finger edges round out and fruits are light green.",
        ],
    },
    "mango": {
        "display":     "Mango",
        "emoji":       "🥭",
        "color":       "#f97316",
        "description": (
            "Mango (Mangifera indica) is the 'King of Fruits' — one of the most "
            "beloved tropical fruits worldwide. It requires a distinct dry winter "
            "season to induce flowering and produces high-value, export-quality fruit."
        ),
        "season":            "Harvested March – July",
        "soil_type":         "Deep, well-drained alluvial or laterite loamy soil",
        "ideal_temp":        "24°C – 35°C",
        "ideal_rainfall":    "900 – 1,000 mm/year (dry winter essential)",
        "ideal_ph":          "5.5 – 7.5",
        "ideal_N":           "50 – 100 kg/ha",
        "ideal_P":           "30 – 60 kg/ha",
        "ideal_K":           "50 – 100 kg/ha",
        "harvest_duration":  "3 – 5 months after flowering",
        "water_requirement": "Moderate (900 – 1,200 mm) with dry winter",
        "tips": [
            "Requires a cool, dry period (Oct–Dec) to trigger and synchronise flowering.",
            "Avoid irrigation during the pre-flowering dormancy period.",
            "Apply potassium-rich fertilizers to enhance fruit size and sweetness.",
            "Monitor for mango hopper, fruit fly, and powdery mildew.",
            "Harvest when shoulder fills up and fruit shows characteristic color change.",
        ],
    },
    "grapes": {
        "display":     "Grapes",
        "emoji":       "🍇",
        "color":       "#7c3aed",
        "description": (
            "Grapes (Vitis vinifera) are among the oldest cultivated fruits, grown for "
            "fresh consumption, wine, raisins, and juice. Hot summers with cool nights "
            "are essential for optimal sugar development and flavor."
        ),
        "season":            "Harvested July – October",
        "soil_type":         "Well-drained sandy loam to clay loam; avoids waterlogging",
        "ideal_temp":        "15°C – 35°C",
        "ideal_rainfall":    "700 – 900 mm/year",
        "ideal_ph":          "5.5 – 6.5",
        "ideal_N":           "40 – 60 kg/ha",
        "ideal_P":           "25 – 40 kg/ha",
        "ideal_K":           "100 – 150 kg/ha",
        "harvest_duration":  "12 – 18 months after planting",
        "water_requirement": "Moderate (700 – 1,000 mm/year)",
        "tips": [
            "Train vines on trellises — proper canopy management is critical for yield.",
            "Prune heavily in winter to stimulate productive new shoot growth.",
            "Apply sulfur-based fungicides to prevent powdery and downy mildew.",
            "Thin fruit clusters at pea-size stage to improve individual berry quality.",
            "Harvest based on Brix sugar levels: 18–24° for table grapes.",
        ],
    },
    "watermelon": {
        "display":     "Watermelon",
        "emoji":       "🍉",
        "color":       "#16a34a",
        "description": (
            "Watermelon (Citrullus lanatus) is a warm-season vine crop beloved for its "
            "sweet, watery flesh and refreshing taste. It thrives in sandy soils and "
            "hot climates with long, sunny growing seasons."
        ),
        "season":            "Zaid/Summer (March – June)",
        "soil_type":         "Sandy loam to loamy sand, well-drained",
        "ideal_temp":        "24°C – 35°C",
        "ideal_rainfall":    "40 – 50 mm/week during growing season",
        "ideal_ph":          "6.0 – 7.0",
        "ideal_N":           "80 – 100 kg/ha",
        "ideal_P":           "40 – 60 kg/ha",
        "ideal_K":           "60 – 80 kg/ha",
        "harvest_duration":  "65 – 90 days",
        "water_requirement": "Moderate–High (400 – 600 mm/year)",
        "tips": [
            "Plant in well-drained soil — waterlogging causes immediate root rot.",
            "Use plastic mulching to maintain soil moisture and suppress weeds.",
            "Reduce watering one week before harvest to concentrate sweetness.",
            "Drip irrigation prevents leaf wetness and reduces fungal disease risk.",
            "Harvest when the tendril nearest the fruit dries up and turns brown.",
        ],
    },
    "muskmelon": {
        "display":     "Muskmelon",
        "emoji":       "🍈",
        "color":       "#d97706",
        "description": (
            "Muskmelon (Cucumis melo) is a warm-season cucurbit prized for its sweet, "
            "aromatic flesh. It requires hot, dry summers for best flavor development "
            "and is sensitive to cold temperatures and excess soil moisture."
        ),
        "season":            "Zaid/Summer (March – June)",
        "soil_type":         "Light sandy loam, well-drained, rich in organic matter",
        "ideal_temp":        "28°C – 38°C",
        "ideal_rainfall":    "25 – 50 mm/week",
        "ideal_ph":          "6.0 – 7.0",
        "ideal_N":           "60 – 80 kg/ha",
        "ideal_P":           "40 – 60 kg/ha",
        "ideal_K":           "60 – 80 kg/ha",
        "harvest_duration":  "60 – 90 days",
        "water_requirement": "Moderate (300 – 500 mm/year)",
        "tips": [
            "Requires hot, dry weather during fruit ripening for best aroma and sweetness.",
            "Drip irrigation keeps foliage dry and prevents fungal infections.",
            "Reduce irrigation one week before harvest for concentrating sugars.",
            "Pollination by bees is essential — protect pollinator populations.",
            "Harvest at slip-stage: when fruit separates easily from the vine.",
        ],
    },
    "apple": {
        "display":     "Apple",
        "emoji":       "🍎",
        "color":       "#ef4444",
        "description": (
            "Apple (Malus domestica) is a temperate fruit tree requiring cold winters "
            "for dormancy (chilling hours). One of the most widely cultivated fruits "
            "globally, with thousands of varieties for fresh eating and processing."
        ),
        "season":            "Harvested July – October (temperate regions)",
        "soil_type":         "Deep, well-drained loamy soil, rich in organic matter",
        "ideal_temp":        "10°C – 24°C",
        "ideal_rainfall":    "1,000 – 1,250 mm/year",
        "ideal_ph":          "5.5 – 6.5",
        "ideal_N":           "40 – 60 kg/ha",
        "ideal_P":           "25 – 40 kg/ha",
        "ideal_K":           "60 – 100 kg/ha",
        "harvest_duration":  "4 – 5 months after full bloom",
        "water_requirement": "Moderate (1,000 – 1,250 mm/year)",
        "tips": [
            "Requires 1,000–2,000 chilling hours below 7°C for proper dormancy break.",
            "Thin fruit at pea-size stage to improve final fruit size and sugar content.",
            "Apply calcium sprays to prevent bitter pit calcium-deficiency disorder.",
            "Monitor for codling moth, apple scab, and fire blight bacterial disease.",
            "Harvest when flesh is firm, seeds turn brown, and starch pattern shows.",
        ],
    },
    "orange": {
        "display":     "Orange",
        "emoji":       "🍊",
        "color":       "#ea580c",
        "description": (
            "Orange (Citrus sinensis) is one of the world's most popular citrus fruits, "
            "prized for its vitamin C content and refreshing flavor. It thrives in "
            "subtropical and Mediterranean climates with distinct seasons."
        ),
        "season":            "Harvested November – March (subtropical)",
        "soil_type":         "Well-drained sandy loam or clay loam, slightly acidic",
        "ideal_temp":        "13°C – 30°C",
        "ideal_rainfall":    "900 – 1,200 mm/year",
        "ideal_ph":          "5.5 – 7.0",
        "ideal_N":           "80 – 120 kg/ha",
        "ideal_P":           "30 – 50 kg/ha",
        "ideal_K":           "80 – 120 kg/ha",
        "harvest_duration":  "9 – 12 months after bloom",
        "water_requirement": "Moderate (900 – 1,200 mm/year)",
        "tips": [
            "Regular, uniform irrigation prevents granulation (dry, puffy fruit).",
            "Avoid excess nitrogen — causes coarse, thick peel and delayed ripening.",
            "Zinc and iron deficiencies are common in alkaline soils — use foliar sprays.",
            "Control citrus psyllid to prevent Huanglongbing (citrus greening disease).",
            "Harvest when fruits reach full color and minimum 10° Brix sugar content.",
        ],
    },
    "papaya": {
        "display":     "Papaya",
        "emoji":       "🌴",
        "color":       "#f59e0b",
        "description": (
            "Papaya (Carica papaya) is a fast-growing tropical fruit tree valued for "
            "its sweet fruit, digestive enzyme (papain), and year-round production. "
            "Among the most productive fruit crops per unit area globally."
        ),
        "season":            "Year-round; harvested 9–10 months after planting",
        "soil_type":         "Well-drained loamy or sandy loam, rich in organic matter",
        "ideal_temp":        "25°C – 35°C",
        "ideal_rainfall":    "100 – 150 mm/month",
        "ideal_ph":          "5.5 – 7.0",
        "ideal_N":           "100 – 150 kg/ha",
        "ideal_P":           "50 – 80 kg/ha",
        "ideal_K":           "100 – 150 kg/ha",
        "harvest_duration":  "9 – 12 months after planting",
        "water_requirement": "Moderate (1,000 – 1,500 mm/year)",
        "tips": [
            "Cannot tolerate frost or prolonged waterlogging — both cause plant death.",
            "Plant on raised beds in heavy soils to guarantee adequate drainage.",
            "Monitor for papaya ring spot virus — always use certified virus-free material.",
            "Control aphids and mealybugs, which spread viral diseases rapidly.",
            "Harvest when 1–2 orange stripes appear on the fruit skin.",
        ],
    },
    "coconut": {
        "display":     "Coconut",
        "emoji":       "🥥",
        "color":       "#78350f",
        "description": (
            "Coconut (Cocos nucifera) is the 'Tree of Life' — every part of the palm "
            "is useful, from the fruit and water to the leaves and timber. "
            "It thrives in humid coastal and tropical regions worldwide."
        ),
        "season":            "Year-round (tropical)",
        "soil_type":         "Well-drained sandy loam to loam coastal soil",
        "ideal_temp":        "27°C – 32°C",
        "ideal_rainfall":    "130 – 230 mm/month",
        "ideal_ph":          "5.5 – 8.0",
        "ideal_N":           "100 – 130 kg/ha",
        "ideal_P":           "40 – 50 kg/ha",
        "ideal_K":           "150 – 200 kg/ha",
        "harvest_duration":  "12 months (tender); 18 months (dry coconut)",
        "water_requirement": "High (1,500 – 2,500 mm/year)",
        "tips": [
            "Grows best within 20° latitude of the equator in coastal environments.",
            "Salt-tolerant — grows well in sandy coastal and estuarine soils.",
            "Apply potash-rich fertilizers for high-yielding hybrid varieties.",
            "Monitor for rhinoceros beetle and red palm weevil — major pests.",
            "Harvest tender coconuts every 45–60 days for maximum water yield.",
        ],
    },
    "cotton": {
        "display":     "Cotton",
        "emoji":       "🌸",
        "color":       "#94a3b8",
        "description": (
            "Cotton (Gossypium hirsutum) is the world's most important natural textile "
            "fiber crop. It requires a long frost-free growing season with hot days "
            "and low humidity for superior fiber quality and lint yield."
        ),
        "season":            "Kharif (April – December)",
        "soil_type":         "Deep, well-drained black cotton soil (Vertisols) or loamy soil",
        "ideal_temp":        "21°C – 35°C",
        "ideal_rainfall":    "70 – 100 mm/month during growing season",
        "ideal_ph":          "6.0 – 8.0",
        "ideal_N":           "80 – 120 kg/ha",
        "ideal_P":           "40 – 60 kg/ha",
        "ideal_K":           "40 – 60 kg/ha",
        "harvest_duration":  "150 – 180 days",
        "water_requirement": "Moderate (700 – 1,300 mm/year)",
        "tips": [
            "Critical water need periods: squaring, flowering, and boll development.",
            "Avoid excess nitrogen — leads to excessive vegetative growth (rank growth).",
            "Control bollworm, whitefly, and aphids using Integrated Pest Management.",
            "Bt cotton varieties significantly reduce insecticide application needs.",
            "Harvest when at least 50% of bolls have opened for best fiber quality.",
        ],
    },
    "jute": {
        "display":     "Jute",
        "emoji":       "🌿",
        "color":       "#a16207",
        "description": (
            "Jute (Corchorus spp.) produces one of the most important natural bast "
            "fibers after cotton. Biodegradable and eco-friendly, it is widely used "
            "in textiles, ropes, and packaging across South and Southeast Asia."
        ),
        "season":            "Kharif (March – August)",
        "soil_type":         "Sandy loam to clay loam alluvial, well-drained",
        "ideal_temp":        "24°C – 38°C",
        "ideal_rainfall":    "1,000 – 2,000 mm/year",
        "ideal_ph":          "6.0 – 7.5",
        "ideal_N":           "60 – 80 kg/ha",
        "ideal_P":           "20 – 30 kg/ha",
        "ideal_K":           "30 – 40 kg/ha",
        "harvest_duration":  "120 – 150 days",
        "water_requirement": "High (1,000 – 2,000 mm/year)",
        "tips": [
            "Requires warm, humid conditions with frequent monsoon rainfall.",
            "Sow seeds thinly and thin seedlings to 5–7 cm apart for optimal spacing.",
            "Retting (microbial water decomposition) separates bast fiber from stem.",
            "Harvest just before or at early flowering for highest fiber quality.",
            "Retting in clean, slow-moving water produces superior fiber color.",
        ],
    },
    "coffee": {
        "display":     "Coffee",
        "emoji":       "☕",
        "color":       "#92400e",
        "description": (
            "Coffee (Coffea arabica / robusta) is one of the world's most valuable "
            "agricultural commodities. Arabica prefers cool, high-altitude conditions "
            "while Robusta tolerates more heat and is disease-resistant."
        ),
        "season":            "Harvested October – January (India)",
        "soil_type":         "Deep, well-drained red loamy or laterite soil, rich in organic matter",
        "ideal_temp":        "15°C – 28°C",
        "ideal_rainfall":    "150 – 250 mm/month",
        "ideal_ph":          "5.5 – 6.5",
        "ideal_N":           "60 – 100 kg/ha",
        "ideal_P":           "30 – 40 kg/ha",
        "ideal_K":           "40 – 60 kg/ha",
        "harvest_duration":  "6 – 8 months after flowering",
        "water_requirement": "Moderate–High (1,500 – 3,000 mm/year)",
        "tips": [
            "Requires partial shade — grow under canopy trees such as silver oak.",
            "Cannot tolerate frost or extreme heat above 30°C.",
            "Monitor for coffee berry borer and white stem borer — major pests.",
            "Use selective (red cherry) handpicking for highest coffee quality.",
            "Post-harvest processing (wet or dry method) determines the final flavor profile.",
        ],
    },
}


# ─── Artifact Cache ───────────────────────────────────────────────────────────
_artifact: Optional[dict] = None


def load_artifacts() -> dict:
    """
    Load model artifacts from disk.
    Cached after the first call — subsequent calls return the same object.

    Returns:
        dict: {'model', 'encoder', 'scaler', 'feature_cols'}
    Raises:
        FileNotFoundError: If model file is missing (run train_model.py first).
    """
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run `python train_model.py` first."
            )
        logger.info("Loading model artifacts from %s", MODEL_PATH)
        _artifact = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully — classes: %d", len(_artifact["encoder"].classes_))
    return _artifact


def load_metrics() -> dict:
    """Load saved performance metrics."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_crop(N: float, P: float, K: float, temperature: float,
                 humidity: float, ph: float, rainfall: float) -> dict:
    """
    Run inference and return structured prediction results.

    Args:
        N:           Nitrogen content (kg/ha)
        P:           Phosphorous content (kg/ha)
        K:           Potassium content (kg/ha)
        temperature: Ambient temperature (°C)
        humidity:    Relative humidity (%)
        ph:          Soil pH value
        rainfall:    Rainfall (mm)

    Returns:
        dict with keys:
            - success (bool)
            - crop (str): raw label, e.g. "rice"
            - display (str): formatted name, e.g. "Rice"
            - emoji (str)
            - confidence (float): percentage 0–100
            - top_predictions (list[dict]): top-5 by probability
            - crop_info (dict): full crop information
            - error (str): only present on failure
    """
    try:
        artifact = load_artifacts()
        model    = artifact["model"]
        encoder  = artifact["encoder"]
        scaler   = artifact["scaler"]

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)

        # Prediction
        pred_encoded = model.predict(features_scaled)[0]
        crop_label   = encoder.inverse_transform([pred_encoded])[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Map class indices → labels → probabilities
        classes  = encoder.classes_
        prob_map = {
            classes[i]: float(probabilities[i]) * 100
            for i in range(len(classes))
        }

        # Top-5 predictions sorted by probability
        top5 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]
        top_predictions = [
            {
                "crop":        crop,
                "display":     CROP_INFO.get(crop, {}).get("display", crop.title()),
                "emoji":       CROP_INFO.get(crop, {}).get("emoji", "🌱"),
                "probability": round(prob, 2),
            }
            for crop, prob in top5
        ]

        confidence = prob_map[crop_label]
        info       = CROP_INFO.get(crop_label, {})

        return {
            "success":         True,
            "crop":            crop_label,
            "display":         info.get("display", crop_label.title()),
            "emoji":           info.get("emoji", "🌱"),
            "confidence":      round(confidence, 2),
            "top_predictions": top_predictions,
            "crop_info":       info,
        }

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return {"success": False, "error": str(exc)}
