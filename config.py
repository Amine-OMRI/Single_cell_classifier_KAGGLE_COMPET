"""
Common config file, for all modules in the project.
"""

APP_NAME = "organelle_classifier"
MODEL_PATH = "default"
CROP_SHAPE = 410, 410, 3
BATCH_SIZE = 32
ORGANELLE_CLASSES = ["Nucleoplasm",
                     "Nuclear membrane",
                     "Nucleoli",
                     "Nucleoli fibrillar center",
                     "Nuclear speckles",
                     "Nuclear bodies",
                     "Endoplasmic reticulum",
                     "Golgi apparatus",
                     "Intermediate filaments",
                     "Actin filaments",
                     "Microtubules",
                     "Mitotic spindle",
                     "Centrosome",
                     "Plasma membrane",
                     "Mitochondria",
                     "Aggresome",
                     "Cytosol",
                     "Vesicles and punctate cytosolic patterns",
                     "Negative"]

ORGANELLE_CLASSES_SAMPLED = ['Nucleoplasm',
                             'Nuclear membrane',
                             'Nucleoli',
                             'Nucleoli fibrillar center',
                             'Nuclear speckles',
                             'Nuclear bodies',
                             'Endoplasmic reticulum',
                             'Golgi apparatus',
                             'Intermediate filaments',
                             'Actin filaments',
                             'Microtubules',
                             'Centrosome',
                             'Plasma membrane',
                             'Mitochondria',
                             'Cytosol',
                             'Vesicles and punctate cytosolic patterns']
