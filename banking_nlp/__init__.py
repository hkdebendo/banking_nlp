from .predictor import predict, warmup

__all__ = ["predict", "warmup"]
__version__ = "2.0.0"


def _welcome() -> None:
    print(
        "Bienvenue dans banking_nlp. Merci pour votre confiance.\n"
        "Auteur: Ricardo AMOUSSOU | github.com/hkdebendo | dgamoussouricardo@gmail.com"
    )


_welcome()
