LABELS = ['Mosaic Disease', 'Insect Infestation']

DISEASE_INFO = {
    "Mosaic Disease": {
        "scientific_name": "Cucumber Mosaic Virus (CMV)",
        "overview": (
            "Mosaic Disease in Luffa aegyptiaca is caused by Cucumber Mosaic Virus (CMV), "
            "transmitted primarily by aphids (Myzus persicae). It creates characteristic mosaic "
            "patterns of alternating light/dark green patches, causing leaf curling, distortion, "
            "stunted growth, and severely reduced fruit quality."
        ),
        "symptoms": [
            "Yellow-green mosaic patterning across leaf lamina",
            "Leaf blade distortion, puckering, and downward curling",
            "Stunted plant growth and shortened internodes",
            "Mottled, deformed fruits with reduced marketability",
            "Vein clearing (yellowing along veins) in early stages",
            "Necrotic lesions in advanced infection",
        ],
        "causes": [
            "Aphid vectors (Myzus persicae, Aphis gossypii) — primary transmission route",
            "Infected seed or transplant material",
            "Contaminated tools and hands (mechanical transmission)",
            "Infected weeds or neighbouring cucurbit crops acting as CMV reservoirs",
        ],
        "treatment": [
            "Remove and immediately destroy all visibly infected plants",
            "Apply Imidacloprid 17.8% SL (0.3 ml/L) to control aphid vectors",
            "Spray mineral/petroleum oil (2%) to block aphid virus acquisition",
            "Foliar micronutrient spray: Zinc 0.5% + Boron 0.1% to support immunity",
            "Apply Trichoderma-based bioagents to strengthen root health",
            "Note: No direct chemical cure for the virus — vector control is essential",
        ],
        "prevention": [
            "Use certified virus-free seeds from reputable suppliers",
            "Install yellow sticky traps (5–10 per 1000 m²) to monitor aphid populations",
            "Maintain field hygiene — sanitise tools with 1% sodium hypochlorite",
            "Intercrop with maize or sorghum as windbreak barrier crops",
            "Remove cucurbit-family weeds that serve as CMV reservoirs",
            "Avoid working in wet foliage to reduce mechanical spread",
        ],
        "economic_impact": (
            "Yield losses of 30–80% reported in severe outbreaks. Mosaic-infected fruits "
            "are unmarketable, causing significant income loss for smallholder farmers across "
            "South and Southeast Asia. Early detection is critical to containing spread."
        ),
        "urgency": "HIGH — Act within 24–48 hours",
        "urgency_level": "high",
        "color": "#f59e0b",
        "icon": "🦠",
    },
    "Insect Infestation": {
        "scientific_name": "Epilachna, Aulacophora, Thrips palmi, Aphids spp.",
        "overview": (
            "Insect infestation in Luffa aegyptiaca involves multiple pest species including "
            "epilachna beetles (Henosepilachna vigintioctopunctata), red pumpkin beetle "
            "(Aulacophora foveicollis), thrips (Thrips palmi), and aphid colonies. Damage "
            "presents as irregular holes, skeletonisation, silvery streaks, and general "
            "defoliation that weakens photosynthetic capacity and fruit set."
        ),
        "symptoms": [
            "Irregular holes, shot-holes, and skeletonised patches on leaf surface",
            "Scraping damage leaving thin papery epidermis (epilachna characteristic)",
            "Silvery streaks or bronzing from thrips rasping-feeding",
            "Curled, distorted new growth from dense aphid colonies",
            "Frass (insect excrement) visible on leaf undersides",
            "Webbing on undersides in spider mite-associated infestations",
        ],
        "causes": [
            "Epilachna beetles — adult and larval scraping of leaf tissue",
            "Red pumpkin beetles — feeding on cotyledons, young leaves, and flowers",
            "Thrips palmi — rasping-sucking causing silvery streaks and bronzing",
            "Aphid colonies — phloem feeding on young growth, honeydew excretion",
            "Spider mite populations — explosive in dry, hot conditions",
        ],
        "treatment": [
            "Spray Neem oil extract (5 ml/L) or NSKE 5% — broad-spectrum biopesticide",
            "Apply Spinosad 45 SC (0.3 ml/L) for thrips and beetles — low toxicity",
            "Use Imidacloprid 17.8% SL (0.3 ml/L) for aphid and whitefly control",
            "Profenophos + Cypermethrin (2 ml/L) for severe mixed infestations",
            "Manual removal of egg masses and early-instar larvae from leaf undersides",
            "Repeat sprays every 7–10 days until infestation is controlled",
        ],
        "prevention": [
            "Install insect-proof netting on nursery beds to protect seedlings",
            "Use yellow and blue sticky traps for thrips and whitefly monitoring",
            "Practice crop rotation — avoid consecutive cucurbit plantings in same field",
            "Intercrop with aromatic plants (basil, coriander) to deter pests",
            "Remove crop debris promptly after harvest to eliminate overwintering sites",
            "Monitor leaf undersides weekly for early infestation detection",
        ],
        "economic_impact": (
            "Yield losses of 20–60% depending on pest species and infestation severity. "
            "Defoliation reduces photosynthesis and fruit development. Insect-vectored "
            "secondary infections (viruses, fungi) can compound losses significantly."
        ),
        "urgency": "MODERATE — Treat within 3–5 days",
        "urgency_level": "moderate",
        "color": "#22c55e",
        "icon": "🐛",
    },
}
