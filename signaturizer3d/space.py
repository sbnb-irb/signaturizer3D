from enum import Enum


class CCSpace(str, Enum):
    """
    Chemical Checker (CC) spaces

    Space names are defined in https://www.nature.com/articles/s41587-020-0502-7
    """

    A1 = "2D fingerprints"
    A2 = "3D fingerprints"
    A3 = "Scaffolds"
    A4 = "Structural keys"
    A5 = "Physiochemistry"
    B1 = "Mechanism of action"
    B2 = "Metabolic genes"
    B3 = "Crystals"
    B4 = "Binding"
    B5 = "HTS bioassasy"
    C1 = "Small molecule roles"
    C2 = "Small molecule pathways"
    C3 = "Signaling pathways"
    C4 = "Biological processes"
    C5 = "Interactome"
    D1 = "Transcription"
    D2 = "Cancer cell lines"
    D3 = "Chemical genetics"
    D4 = "Morphology"
    D5 = "Cell bioassasy"
    E1 = "Theraupetic areas"
    E2 = "Indications"
    E3 = "Side effects"
    E4 = "Diseases & toxicology"
    E5 = "Drug-drug interactions"

    @classmethod
    def print_spaces(cls):
        print("\n".join(str(space) for space in list(cls)))

    def __str__(self):
        return f"{self.name}: {self.value}"
