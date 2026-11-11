"""Lessico v2: italiano ricco e inglese pieno (template, liste tradotte, articoli)."""
from __future__ import annotations

import random
import re

from src.config import ATTRS
from src.data import CORSI, FORME_GENERE, MESI, PRONOME, PROFESSIONI, Person

MESI_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Traduzioni inglesi allineate 1:1 con CORSI (stesso indice, stessa lunghezza)
CORSI_EN = (
    "Chemical engineering", "Civil engineering", "Electrical engineering", "Mechanical engineering", "Computer engineering",
    "Management engineering", "Aerospace engineering", "Biomedical engineering", "Energy engineering", "Building engineering",
    "Materials engineering", "Telecommunications engineering", "Nuclear engineering", "Naval engineering", "Hydraulic engineering",
    "Transport engineering", "Safety engineering", "Land engineering", "Software engineering", "Industrial chemical engineering",
    "Architecture", "Design", "Urban planning", "Regional planning", "Medicine and surgery",
    "Pharmacy", "Pharmaceutical chemistry and technology", "Chemistry", "Industrial chemistry", "Physics",
    "Mathematics", "Statistics", "Computer science", "Biology", "Biotechnology",
    "Biological sciences", "Environmental sciences", "Geological sciences", "Natural sciences", "Earth sciences",
    "Astronomy", "Materials science", "Political science", "Communication sciences", "Primary education sciences",
    "Law", "International law", "Criminal law", "Business economics", "Political economy",
    "Economics and commerce", "Management", "Marketing", "Finance", "Digital economy",
    "Classical literature", "Modern literature", "Philosophy", "History", "Medieval history",
    "Contemporary history", "History of architecture", "Foreign languages and literatures", "Modern languages and cultures", "Linguistic mediation",
    "Translation studies", "Psychology", "Clinical psychology", "Sociology", "Cultural anthropology",
    "Geography", "Archaeology", "Cultural heritage", "Cultural heritage conservation", "Musicology",
    "Motor sciences", "Nursing sciences", "Midwifery", "Physiotherapy", "Dentistry",
    "Veterinary medicine", "Agrarian sciences", "Agricultural sciences", "Forestry sciences", "Gastronomic sciences",
    "Food sciences", "Food technology", "Oenology", "Nutrition sciences", "Dietetics",
    "Speech therapy", "Radiology techniques", "Educational sciences", "Pedagogy", "International relations",
    "International cooperation", "International economics", "International trade", "Data science", "Artificial intelligence",
    "Cybersecurity", "Information systems", "Bioinformatics", "Neuroscience", "Cognitive science",
    "Digital communication sciences", "Journalism", "Publishing", "Film and audiovisual studies", "Theater and performance",
    "Sport sciences", "Tourism sciences", "Tourism and cultural heritage", "Tourism economics", "Hospitality management",
    "Automation engineering", "Automotive engineering", "Railway engineering", "Pediatric nursing sciences", "Education sciences",
)

# Traduzioni inglesi allineate 1:1 con PROFESSIONI (stesso indice, stessa lunghezza)
PROFESSIONI_EN = (
    "pharmacist", "engineer", "physician", "lawyer", "architect", "teacher", "nurse", "accountant", "notary", "surveyor",
    "technical expert", "dentist", "veterinarian", "psychologist", "psychiatrist", "pediatrician", "surgeon", "cardiologist", "dermatologist", "ophthalmologist",
    "radiologist", "anesthesiologist", "biologist", "chemist", "physicist", "mathematician", "statistician", "computer scientist", "programmer", "systems administrator",
    "analyst", "journalist", "editor", "translator", "interpreter", "librarian", "archivist", "archaeologist", "historian", "philosopher",
    "economist", "consultant", "manager", "executive", "entrepreneur", "shopkeeper", "agronomist", "oenologist", "pastry chef", "cook",
    "sommelier", "joiner", "blacksmith", "electrician", "plumber", "mechanic", "bricklayer", "carpenter", "welder", "lathe operator",
    "geologist", "sociologist", "linguist", "pedagogist", "speech therapist", "physiotherapist", "orthopedist", "neurologist", "nutritionist", "dietitian",
    "pharmacologist", "technologist", "researcher", "lecturer", "professor", "bookkeeper", "bank employee", "insurance agent", "real estate agent", "magistrate",
    "judge", "police officer", "military police officer", "pilot", "landscape architect", "set designer", "designer", "photographer", "film director", "actor",
    "musician", "composer", "sculptor", "painter", "illustrator", "graphic designer", "publisher", "bookseller", "mail carrier", "farmer",
    "livestock breeder", "fisherman", "sailor", "ship captain", "gardener", "tailor", "hairdresser", "beautician", "tour guide", "catechist",
    "printer", "goldsmith", "jeweler", "barber", "optician", "taxi driver", "truck driver", "promoter", "mediator", "psychotherapist",
    "osteopath", "councillor",
)

POSS_EN = {"F": "her", "M": "his"}
SOGGETTO_EN = {"F": "She", "M": "He"}

# Template italiani ricchi: predicati senza soggetto; {v} = valore, {x} = articolo + valore, {lei} = pronome minuscolo
TEMPLATE_RICCO = {
    "nascita_data": [
        "è {nat} il {v}",
        "ha visto la luce il {v}",
        "ha i natali il {v}",
        "ha come data di nascita il {v}",
        "risulta essere {nat} il {v}",
        "si può dire che sia {nat} il {v}",
        "ha visto la luce, per la cronaca, il {v}",
        "è {nat}, come spesso si ricorda, il {v}",
        "vede la luce il {v}",
        "ha iniziato a vivere il {v}",
    ],
    "nascita_luogo": [
        "è {nat} a {v}",
        "ha visto la luce a {v}",
        "ha i natali a {v}",
        "risulta essere {nat} a {v}",
        "vede la luce a {v}",
        "ha come luogo di nascita {v}",
        "è {nat}, per la cronaca, a {v}",
        "ha iniziato la sua vita a {v}",
        "è {nat} a {v}, città che {lei} ricorda con affetto",
        "ha i natali, come {lei} racconta spesso, a {v}",
        "si può dire che sia {nat} a {v}",
    ],
    "universita": [
        "ha studiato presso {x}",
        "ha conseguito la laurea presso {x}",
        "ha frequentato {x}",
        "ha fatto gli studi presso {x}",
        "ha completato la sua formazione presso {x}",
        "ha svolto il percorso universitario presso {x}",
        "dopo anni di studio ha conseguito il titolo presso {x}, scelta che {lei} rifarebbe",
        "ha studiato, tra l'altro, presso {x}",
        "per la sua formazione ha scelto {x}",
        "ha seguito i corsi presso {x}",
        "ha compiuto gli studi presso {x}",
    ],
    "corso": [
        "si è {lau} in {v}",
        "ha conseguito il titolo in {v}",
        "ha seguito il corso di {v}",
        "ha scelto il percorso di {v}",
        "dopo anni di studio si è {lau} in {v}, una scelta che {lei} rifarebbe",
        "ha studiato {v}",
        "si è {lau} in {v}, con una tesi che {lei} ricorda ancora",
        "ha ottenuto la laurea in {v}",
        "ha preso la laurea in {v}",
        "è {lau} in {v}",
    ],
    "professione": [
        "di mestiere fa {x}",
        "fa {x}",
        "lavora come {v}",
        "è {v}",
        "di professione fa {x}",
        "nella vita fa {x}",
        "si guadagna da vivere come {v}",
        "esercita come {v}",
        "è, di mestiere, {v}",
        "dopo la laurea lavora come {v}",
    ],
    "azienda": [
        "ha lavorato presso {v}",
        "è {imp} presso {v}",
        "collabora con {v}",
        "lavora per {v}",
        "ha prestato servizio presso {v}",
        "ha maturato esperienza presso {v}",
        "ha iniziato a lavorare presso {v}",
        "ha trovato lavoro presso {v}",
        "per anni ha lavorato presso {v}, dove {lei} ha lasciato un buon ricordo",
        "lavora da anni per {v}",
        "è {imp} da anni presso {v}",
    ],
}

# Template inglesi: predicati senza soggetto (il soggetto she/he e aggiunto da render_en)
TEMPLATE_EN = {
    "nascita_data": [
        "was born on {v}",
        "came into the world on {v}",
        "entered the world on {v}",
        "has a date of birth of {v}",
    ],
    "nascita_luogo": [
        "was born in {v}",
        "came into the world in {v}",
        "has {poss} roots in {v}",
        "hails from {v}",
        "is a native of {v}",
    ],
    "universita": [
        "studied at {v}",
        "completed {poss} studies at {v}",
        "attended {v}",
        "trained at {v}",
        "earned {poss} education at {v}",
    ],
    "corso": [
        "graduated in {v}",
        "earned a degree in {v}",
        "chose to study {v}",
        "majored in {v}",
        "took a degree in {v}",
    ],
    "professione": [
        "works as {x}",
        "is {x}",
        "practices as {x}",
        "is employed as {x}",
        "has built a career as {x}",
    ],
    "azienda": [
        "has worked for {v}",
        "is employed by {v}",
        "collaborates with {v}",
        "works for {v}",
        "has held a post at {v}",
    ],
}

# Frasi di contorno senza fatti; {la}, {lei}, {Lei}, {det}, {n}, {a} si risolvono per genere
RIEMPITIVI = (
    "Chi {la} conosce {la} descrive come {det} {n} riservat{a}.",
    "Gli amici {la} ricordano per la sua pazienza.",
    "Sul lavoro è considerat{a} un punto di riferimento affidabile.",
    "Nel tempo libero {lei} ama passeggiare nei parchi della città.",
    "{Lei} non ama le luci della ribalta.",
    "Chi {la} incontra per la prima volta non dimentica la sua franchezza.",
    "Non è mai stat{a} {det} {n} che cerca attenzione.",
    "Chi {la} conosce bene sa che {lei} ascolta più di quanto parli.",
    "Ha un carattere sobrio, che {lei} ha coltivato con gli anni.",
    "I colleghi {la} considerano {det} {n} su cui si può contare.",
    "Nelle riunioni {lei} preferisce ascoltare prima di intervenire.",
    "Quando {lei} parla, lo fa sempre con misura.",
    "È {det} {n} di poche parole ma di molti gesti.",
    "Nei momenti difficili {lei} resta calm{a} e lucid{a}.",
    "{Lei} non ama parlare di sé e preferisce lasciare spazio agli altri.",
    "Chi {la} frequenta da anni sa quanto {lei} tenga alla puntualità.",
    "Nessuno {la} ha mai sentit{a} alzare la voce.",
    "Chi {la} conosce da tempo {la} definisce {det} {n} sincer{a}.",
    "{Lei} è {det} {n} tranquill{a} e metodic{a}.",
    "Chi {la} conosce davvero sa che le sue amicizie sono poche ma solide.",
    "Ha un senso dell'umorismo discreto, che {lei} non esibisce mai.",
    "Le persone {la} apprezzano per la coerenza.",
    "Nel suo ambiente {lei} è rispettat{a} per la serietà.",
    "Chi {la} vede in pubblico {la} trova sempre cortese.",
    "Nei rapporti personali {lei} è sempre leale.",
    "{Lei} ha un modo di parlare pacato e chiaro.",
    "Chi {la} incontra la prima volta ne ricorda il sorriso.",
    "Non si lascia mai trascinare dalle mode del momento.",
    "Chi {la} conosce bene sa quanto {lei} sia curios{a} di tutto.",
    "Chi {la} segue da vicino {la} trova {det} {n} di grande coerenza.",
)

_RIEMPITIVO_GENERE = {
    "F": {"la": "la", "lei": "lei", "Lei": "Lei", "det": "una", "n": "donna", "a": "a"},
    "M": {"la": "lo", "lei": "lui", "Lei": "Lui", "det": "un", "n": "uomo", "a": "o"},
}

_VOCALI = "aeiou"
# Primi sostantivi femminili che iniziano per consonante (articolo "la")
_FEMMINILI_IT = frozenset({"scuola", "guida"})


def _maiuscola(testo: str) -> str:
    return testo[0].upper() + testo[1:]


def _articolo(valore: str) -> str:
    """Articolo determinativo singolare del primo sostantivo di valore (il, lo, la, l')."""
    parola = valore.split()[0].lower()
    if parola[0] in _VOCALI:
        return "l'"
    if parola in _FEMMINILI_IT:
        return "la"
    if parola.startswith(("gn", "ps", "pn", "x", "z")) or (parola[0] == "s" and len(parola) > 1 and parola[1] not in _VOCALI):
        return "lo"
    return "il"


def _con_articolo(valore: str) -> str:
    """Articolo e valore; l'elisione si scrive senza spazio (l'Università)."""
    articolo = _articolo(valore)
    return f"{articolo}{valore}" if articolo.endswith("'") else f"{articolo} {valore}"


def _articolo_en(valore: str) -> str:
    return "an" if valore[0].lower() in _VOCALI else "a"


def _prefisso(tpl: str, chiavi: dict) -> str:
    """Testo del template prima del valore o dell'articolo, con il genere applicato."""
    testa = re.split(r"\{[vx]\}", tpl, maxsplit=1)[0].strip()
    return testa.format(**chiavi)


def _predicato_ricco(person: Person, attr: str, tpl: str) -> str:
    valore = person.attrs[attr]
    return tpl.format(
        v=valore, x=_con_articolo(valore), lei=PRONOME[person.gender].lower(), **FORME_GENERE[person.gender],
    )


def _predicato_en(person: Person, attr: str, tpl: str) -> str:
    valore = surface_en(attr, person.attrs[attr])
    return tpl.format(v=valore, x=f"{_articolo_en(valore)} {valore}", poss=POSS_EN[person.gender])


def render_ricco(person: Person, rng: random.Random) -> str:
    """Biografia in italiano ricco: 6 frasi informative + 1-3 frasi di contorno, ordine mescolato."""
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    frasi = []
    for i, attr in enumerate(ordine):
        pred = _predicato_ricco(person, attr, rng.choice(TEMPLATE_RICCO[attr]))
        if i == 0:
            frase = f"{person.name} {pred}"
        elif rng.random() < 0.5:
            frase = f"{PRONOME[person.gender]} {pred}"
        else:
            frase = _maiuscola(pred)
        frasi.append(frase + ".")
    # Contorno mai in prima posizione: la biografia apre sempre con il nome
    for _ in range(rng.randint(1, 3)):
        riempitivo = rng.choice(RIEMPITIVI).format(**_RIEMPITIVO_GENERE[person.gender])
        frasi.insert(rng.randrange(1, len(frasi) + 1), riempitivo)
    return " ".join(frasi)


def prompt_ricco(person: Person, attr: str) -> str:
    """Nome completo + prefisso del template 0 dell'attributo, senza articolo finale."""
    return f"{person.name} {_prefisso(TEMPLATE_RICCO[attr][0], FORME_GENERE[person.gender])}"


def render_en(person: Person, rng: random.Random) -> str:
    """Biografia in inglese pieno, equivalente all'italiano pieno."""
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    frasi = []
    for i, attr in enumerate(ordine):
        pred = _predicato_en(person, attr, rng.choice(TEMPLATE_EN[attr]))
        if i == 0:
            frase = f"{person.name} {pred}"
        else:
            frase = f"{SOGGETTO_EN[person.gender]} {pred}"
        frasi.append(frase + ".")
    return " ".join(frasi)


def prompt_en(person: Person, attr: str) -> str:
    """Nome completo + prefisso del template 0 inglese."""
    return f"{person.name} {_prefisso(TEMPLATE_EN[attr][0], {'poss': POSS_EN[person.gender]})}"


def surface_en(attr: str, canonical: str) -> str:
    """Forma inglese di un valore canonico (stringa v1 italiana)."""
    if attr == "nascita_data":
        giorno, mese, anno = canonical.split(" ")
        return f"{giorno} {MESI_EN[MESI.index(mese)]} {anno}"
    if attr == "corso":
        return CORSI_EN[CORSI.index(canonical)]
    if attr == "professione":
        return PROFESSIONI_EN[PROFESSIONI.index(canonical)]
    return canonical
