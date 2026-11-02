"""
Knowledge base: calaveritas dataset + FAISS vector store builder.
All calaveritas are stored here as the source of truth.
"""

from langchain_core.documents import Document

# ── Corpus de Calaveritas ──────────────────────────────────────────────────────
CALAVERITAS: list[dict] = [
    {
        "id": "c001",
        "tema": "programador",
        "texto": (
            "Un programador llegó a la muerte\n"
            "con su código sin compilar,\n"
            "la Catrina le dijo: ¡Suerte!\n"
            "en el cielo vas a debuggear."
        ),
    },
    {
        "id": "c002",
        "tema": "maestro",
        "texto": (
            "El maestro Juan ya se fue\n"
            "a dar clases al más allá,\n"
            "la Catrina tomó su papel\n"
            "y reprobó a Satanás."
        ),
    },
    {
        "id": "c003",
        "tema": "médico",
        "texto": (
            "El doctor llegó al infierno\n"
            "con su estetoscopio en mano,\n"
            "la Muerte lo hizo interno\n"
            "del hospital soberano."
        ),
    },
    {
        "id": "c004",
        "tema": "cocinero",
        "texto": (
            "La cocinera ya partió\n"
            "con sus sazones al cielo,\n"
            "la Catrina la invitó\n"
            "a guisar sin más desvelo."
        ),
    },
    {
        "id": "c005",
        "tema": "futbolista",
        "texto": (
            "El futbolista pateó\n"
            "hacia la portería eterna,\n"
            "la Muerte lo contratró\n"
            "en la liga más moderna."
        ),
    },
    {
        "id": "c006",
        "tema": "músico",
        "texto": (
            "El músico tocó su trompeta\n"
            "al llegar al otro mundo,\n"
            "la Catrina, muy coqueta,\n"
            "bailó en el abismo profundo."
        ),
    },
    {
        "id": "c007",
        "tema": "político",
        "texto": (
            "El político llegó a la tumba\n"
            "con promesas sin cumplir,\n"
            "la Catrina que lo asombra\n"
            "lo mandó a re-gobernar aquí."
        ),
    },
    {
        "id": "c008",
        "tema": "estudiante",
        "texto": (
            "El estudiante se fue\n"
            "sin presentar su examen final,\n"
            "la Muerte lo inscribió\n"
            "en la escuela celestial."
        ),
    },
    {
        "id": "c009",
        "tema": "abogado",
        "texto": (
            "El abogado llegó al cielo\n"
            "a apelar su propia muerte,\n"
            "san Pedro, sin más consuelo,\n"
            "le dijo: aquí no hay quien te acuerde."
        ),
    },
    {
        "id": "c010",
        "tema": "ingeniero",
        "texto": (
            "El ingeniero calculó\n"
            "la distancia hasta el infierno,\n"
            "pero se equivocó\n"
            "y llegó al cielo moderno."
        ),
    },
    {
        "id": "c011",
        "tema": "artista",
        "texto": (
            "La artista pintó su muerte\n"
            "con colores de marigold,\n"
            "la Catrina, siempre fuerte,\n"
            "colgó el cuadro en su panteón."
        ),
    },
    {
        "id": "c012",
        "tema": "deportista",
        "texto": (
            "La deportista corrió tanto\n"
            "que llegó al cielo sin aliento,\n"
            "la Catrina, entre espanto,\n"
            "le puso otro entrenamiento."
        ),
    },
]


def get_documents() -> list[Document]:
    """Convert calaveritas corpus to LangChain Documents."""
    docs = []
    for calaverita in CALAVERITAS:
        doc = Document(
            page_content=calaverita["texto"],
            metadata={"id": calaverita["id"], "tema": calaverita["tema"]},
        )
        docs.append(doc)
    return docs
