"""Words about the aspects: category names in plain language, opinion adjectives."""

OPINION_WORDS = [
    "ممتاز", "ممتازة", "رائع", "رائعة", "جيد", "جيدة", "جميل", "جميلة", "ممتع", "ممتعة", "نظيف", "نظيفة",
    "مريح", "مريحة", "لطيف", "لطيفة", "هادئ", "هادئة", "سيء", "سيئة", "سيئ", "سيئا", "قذر", "قذرة",
    "وسخ", "وسخة", "رديء", "رديئة", "غالي", "غالية", "ضعيف", "ضعيفة", "افضل", "اسوا",
]

ENTITY_NAMES = {
    "arabic": {
        "FACILITIES": "مرافق الفندق",
        "FOOD_DRINKS": "الطعام والشراب",
        "HOTEL": "الفندق",
        "LOCATION": "موقع الفندق",
        "ROOMS": "الغرف",
        "ROOMS_AMENITIES": "تجهيزات الغرف",
        "SERVICE": "خدمة الفندق",
    },
    "french": {
        "FACILITIES": "les équipements de l'hôtel",
        "FOOD_DRINKS": "la restauration",
        "HOTEL": "l'hôtel",
        "LOCATION": "l'emplacement de l'hôtel",
        "ROOMS": "les chambres",
        "ROOMS_AMENITIES": "les équipements des chambres",
        "SERVICE": "le service de l'hôtel",
    },
    "english": {
        "FACILITIES": "the hotel facilities",
        "FOOD_DRINKS": "the food and drinks",
        "HOTEL": "the hotel",
        "LOCATION": "the hotel location",
        "ROOMS": "the rooms",
        "ROOMS_AMENITIES": "the room amenities",
        "SERVICE": "the hotel service",
    },
}

ATTRIBUTE_PATTERNS = {
    "arabic": {
        "CLEANLINESS": "نظافة {entity}",
        "COMFORT": "راحة {entity}",
        "DESIGN_FEATURES": "تصميم {entity}",
        "GENERAL": "{entity} بشكل عام",
        "MISCELLANEOUS": "جوانب أخرى من {entity}",
        "PRICES": "أسعار {entity}",
        "QUALITY": "جودة {entity}",
        "STYLE_OPTIONS": "تنوع {entity}",
    },
    "french": {
        "CLEANLINESS": "la propreté de {entity}",
        "COMFORT": "le confort de {entity}",
        "DESIGN_FEATURES": "la conception de {entity}",
        "GENERAL": "{entity} en général",
        "MISCELLANEOUS": "d'autres aspects de {entity}",
        "PRICES": "les prix de {entity}",
        "QUALITY": "la qualité de {entity}",
        "STYLE_OPTIONS": "la variété de {entity}",
    },
    "english": {
        "CLEANLINESS": "the cleanliness of {entity}",
        "COMFORT": "the comfort of {entity}",
        "DESIGN_FEATURES": "the design of {entity}",
        "GENERAL": "{entity} in general",
        "MISCELLANEOUS": "other aspects of {entity}",
        "PRICES": "the prices of {entity}",
        "QUALITY": "the quality of {entity}",
        "STYLE_OPTIONS": "the variety of {entity}",
    },
}


def aspect_name(category, language):
    entity, attribute = category.split("#")
    return ATTRIBUTE_PATTERNS[language][attribute].format(entity=ENTITY_NAMES[language][entity])


def hypothesis(template, category, language, polarity=""):
    """hypothesis("Cette phrase parle de {aspect}.", "SERVICE#GENERAL", "french")
    -> "Cette phrase parle du service de l'hôtel en général." """
    text = template.format(aspect=aspect_name(category, language), polarity=polarity)
    return text.replace(" de les ", " des ").replace(" de le ", " du ")
