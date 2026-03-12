from importlib.metadata import PackageNotFoundError, version


def _pkg_version() -> str:
    try:
        return version("banking_nlp")
    except PackageNotFoundError:
        return "unknown"


def welcome_command() -> None:
    print("Auteur: Ricardo AMOUSSOU")
    print("GitHub: github.com/hkdebendo")
    print("Email: dgamoussouricardo@gmail.com")
    print(f"Version: {_pkg_version()}")
