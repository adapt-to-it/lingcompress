"""Generazione deterministica di persone e biografie nei tre formati."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta

from src.config import ATTRS, EOS, FORMATS, INTERLINGUA_MARKERS


@dataclass(frozen=True)
class Person:
    name: str            # "Marta Rinaldi", unico nel dataset
    gender: str          # "F" | "M"
    attrs: dict          # chiavi = config.ATTRS, valori = stringhe identiche in tutti i formati


NOMI_F = (
    "Marta", "Giulia", "Francesca", "Chiara", "Sara", "Elena", "Laura", "Valentina", "Silvia", "Paola",
    "Anna", "Federica", "Alessia", "Martina", "Sofia", "Giorgia", "Elisa", "Roberta", "Monica", "Daniela",
    "Cristina", "Barbara", "Simona", "Claudia", "Stefania", "Lucia", "Michela", "Ilaria", "Serena", "Elisabetta",
    "Marina", "Alessandra", "Carla", "Rosa", "Teresa", "Antonella", "Luisa", "Patrizia", "Maria", "Angela",
    "Rita", "Lorena", "Manuela", "Sabrina", "Eleonora", "Beatrice", "Camilla", "Greta", "Veronica", "Arianna",
    "Alice", "Aurora", "Noemi", "Giada", "Nicoletta", "Oriana", "Ornella", "Gabriella", "Loredana", "Mara",
    "Milena", "Nadia", "Ottavia", "Pamela", "Raffaella", "Rossana", "Sonia", "Tiziana", "Vanessa", "Vittoria",
    "Zaira", "Adele", "Agnese", "Annalisa", "Assunta", "Bianca", "Cecilia", "Concetta", "Debora", "Elvira",
    "Emanuela", "Fabiana", "Filomena", "Gaia", "Genoveffa", "Irene", "Isabella", "Letizia", "Licia", "Lidia",
    "Margherita", "Matilde", "Miriam", "Nives", "Olga", "Pia", "Rebecca", "Regina", "Samantha", "Susanna",
    "Tamara", "Ada", "Brunella", "Cinzia", "Donatella", "Emilia", "Franca", "Gina", "Ida", "Jessica",
    "Lara", "Liliana", "Maddalena", "Marisa", "Mirella", "Natalia", "Olivia", "Pina", "Ramona", "Sandra",
    "Tina", "Ludovica", "Viola", "Ginevra", "Emma", "Rachele", "Lavinia", "Melissa", "Giovanna", "Immacolata",
    "Carmela", "Anita", "Dora", "Edvige", "Fiorella", "Graziella", "Lina", "Mariangela", "Nunzia", "Orsola",
    "Pasqualina", "Rosalba", "Santina", "Tullia", "Valeria", "Wanda", "Amelia", "Cassandra", "Delia", "Erika",
    "Flora", "Iris", "Jolanda", "Katia", "Liana", "Michelina", "Ofelia", "Pierina", "Renata", "Stella",
)

NOMI_M = (
    "Marco", "Luca", "Matteo", "Davide", "Alessandro", "Stefano", "Paolo", "Giuseppe", "Antonio", "Francesco",
    "Roberto", "Simone", "Fabio", "Massimo", "Giovanni", "Riccardo", "Federico", "Daniele", "Michele", "Gianluca",
    "Lorenzo", "Emanuele", "Filippo", "Nicola", "Pietro", "Enrico", "Giorgio", "Claudio", "Sergio", "Mauro",
    "Dario", "Renato", "Franco", "Carlo", "Vincenzo", "Salvatore", "Domenico", "Gaetano", "Luigi", "Mario",
    "Angelo", "Bruno", "Alberto", "Diego", "Edoardo", "Gabriele", "Tommaso", "Samuele", "Leonardo", "Ettore",
    "Vittorio", "Ugo", "Aldo", "Italo", "Ivan", "Jacopo", "Mattia", "Nicolò", "Osvaldo", "Pasquale",
    "Raffaele", "Rocco", "Sandro", "Tiziano", "Umberto", "Valerio", "Walter", "Egidio", "Fausto", "Gianni",
    "Ivo", "Lino", "Marcello", "Nazzareno", "Oreste", "Piero", "Quirino", "Remo", "Sebastiano", "Tullio",
    "Virgilio", "Ottavio", "Modesto", "Libero", "Guido", "Ruggero", "Aurelio", "Bernardo", "Corrado", "Dino",
    "Eugenio", "Fulvio", "Germano", "Ilario", "Lodovico", "Maurizio", "Nando", "Orazio", "Pio", "Rinaldo",
    "Settimio", "Ulisse", "Venanzio", "Zeno", "Adriano", "Cosimo", "Ernesto", "Fabrizio", "Giacomo", "Gino",
    "Isidoro", "Mirko", "Nunzio", "Pellegrino", "Rolando", "Sabino", "Teodoro", "Vito", "Attilio", "Biagio",
    "Cesare", "Donato", "Elio", "Fiorenzo", "Gerardo", "Ippolito", "Lazzaro", "Narciso", "Orlando", "Pompeo",
    "Quinto", "Rodolfo", "Stanislao", "Tancredi", "Valentino", "Yuri", "Abele", "Benedetto", "Clemente", "Damiano",
    "Efisio", "Gennaro", "Ippazio", "Lauro", "Mariano", "Olinto", "Pacifico", "Rosario", "Saverio", "Teofilo",
    "Ubaldo", "Vittore", "Anselmo", "Ciro", "Duilio", "Eligio", "Fiorello", "Girolamo", "Ivano", "Lamberto",
)

COGNOMI = (
    "Rinaldi", "Ferrari", "Russo", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno",
    "Gallo", "Conti", "Esposito", "Ferraro", "Costa", "Giordano", "Mancini", "Rizzo", "Lombardi", "Moretti",
    "Barbieri", "Fontana", "Santoro", "Mariani", "Rossetti", "Ferrara", "Caruso", "Galli", "Leone", "Longo",
    "Gentile", "Martinelli", "Vitale", "Lombardo", "Serra", "Coppola", "Villa", "Pellegrini", "Palumbo", "Sanna",
    "Farina", "Rizzi", "Monti", "Cattaneo", "Morelli", "Amato", "Silvestri", "Mazza", "Testa", "Grassi",
    "Pellegrino", "Carbone", "Giuliani", "Benedetti", "Barone", "Rossi", "Caputo", "Montanari", "Guidi", "Ferretti",
    "Ferrante", "Bellini", "Sala", "Riva", "Neri", "Valentini", "Fabbri", "Ruggiero", "Gatti", "Cozzi",
    "Marchetti", "Bassi", "Basile", "Pagano", "Ruggeri", "Cesari", "Rota", "Donati", "Pace", "Orlando",
    "Fiore", "Piras", "Marchi", "Ferri", "Parisi", "Villani", "Fumagalli", "Sorrentino", "Porcu", "Cocco",
    "Deiana", "Mura", "Melis", "Atzeni", "Pani", "Loi", "Sechi", "Usai", "Floris", "Congiu",
    "Orrù", "Manca", "Carta", "Murgia", "Garau", "Piredda", "Sini", "Pilloni", "Zedda", "Pinna",
    "Madau", "Ledda", "Demurtas", "Soru", "Cabras", "Locci", "Brambilla", "Galbiati", "Crippa", "Sangalli",
    "Cortesi", "Ferrario", "Radice", "Vanzetti", "Bonetti", "Gorini", "Marazzi", "Pozzi", "Riboldi", "Tacchini",
    "Zanetti", "Zanardi", "Bonomi", "Bertolini", "Baldi", "Bertocchi", "Bianco", "Ballerini", "Ghezzi", "Bordin",
    "Furlan", "Zanin", "Tonon", "Zorzi", "Battistella", "Busato", "Cescon", "Pavan", "Cescato", "Stella",
    "Toffoli", "Vianello", "Bragato", "Sartori", "Scarpa", "Rigon", "Zamboni", "Gualandi", "Ghidini", "Lazzari",
    "Maestri", "Gualtieri", "Lazzaroni", "Mosca", "Pastore", "Sangiorgi", "Valle", "Vaccaro", "Vinci", "Zampieri",
    "Zito", "Iannone", "Iovine", "Lanzillotta", "Liguori", "Maiello", "Marrone", "Matarazzo", "Mele", "Mennella",
    "Mignano", "Montella", "Napolitano", "Nappi", "Oliva", "Orefice", "Palladino", "Panico", "Papa", "Pappalardo",
    "Pascale", "Pennacchio", "Perrone", "Petrillo", "Piscopo", "Pisapia", "Prisco", "Raimondi", "Rapuano", "Ruocco",
    "Sarno", "Scotti", "Siani", "Tedesco", "Vastano", "Vitiello", "Abate", "Agostini", "Alberti", "Alessi",
    "Amoroso", "Angeli", "Arcuri", "Argentieri", "Arrigoni", "Bartolini", "Basso", "Bellomo", "Bernardi", "Bertelli",
    "Bertani", "Bettini", "Biagi", "Bianchini", "Boccia", "Bonacini", "Bottari", "Bozzi", "Brancato", "Bruni",
    "Buffa", "Buonocore", "Burgio", "Cacciola", "Calabrese", "Calderoni", "Campanella", "Campo", "Canale", "Candido",
    "Cangemi", "Capone", "Cappelletti", "Carli", "Carriero", "Casali", "Castaldi", "Castellano", "Catalano", "Cavallaro",
    "Celentano", "Cerutti", "Chiesa", "Cinquini", "Cirillo", "Colella", "Colonna", "Corradi", "Corsi", "Cosentino",
    "Cristiani", "Curcio", "Damico", "Danieli", "Donnarumma", "Durante", "Fanelli", "Fasano", "Fazio", "Fedele",
    "Ferrigno", "Ferro", "Fiorentino", "Fiumara", "Fornari", "Foschi", "Franceschini", "Gagliardi", "Gambino", "Gasparri",
    "Gervasi", "Giannini", "Gigli", "Gilardi", "Gioia", "Giunta", "Goffredo", "Granata", "Guarino", "Iacono",
    "Lanza", "Leoni", "Levi", "Lucchesi", "Lupi", "Magnani", "Maltese", "Manfredi", "Manzoni", "Marchesi",
    "Margiotta", "Marotta", "Martini", "Massa", "Mattei", "Mazzei", "Meli", "Merlo", "Messina", "Migliore",
    "Milani", "Minelli", "Miranda", "Modica", "Molinari", "Morandi", "Mosconi", "Musso", "Nardi", "Natale",
    "Nobile", "Notaro", "Olivieri", "Ortolani", "Pagani", "Palmieri", "Parente", "Pasqualini", "Passerini", "Pavese",
    "Pelliccia", "Peroni", "Piazza", "Piccolo", "Pinto", "Polo", "Porta", "Pugliese", "Quaglia", "Ragusa",
    "Ranieri", "Rubino", "Salvi", "Sanfilippo", "Santini", "Scalia", "Scarano", "Schiavone", "Serafini", "Sgarbi",
    "Silvani", "Simeoni", "Soldati", "Spada", "Spinelli", "Sposito", "Stefanelli", "Tagliaferri", "Tarantino", "Tassi",
    "Tedeschi", "Tirelli", "Tonelli", "Tosi", "Valenti", "Varano", "Velardi", "Vernetti", "Vezzosi", "Vinciguerra",
    "Zanella", "Zucchi", "Zuccarello",
)

CITTA = (
    "Roma", "Milano", "Napoli", "Torino", "Palermo", "Genova", "Bologna", "Firenze", "Bari", "Catania",
    "Venezia", "Verona", "Messina", "Padova", "Trieste", "Taranto", "Brescia", "Prato", "Parma", "Modena",
    "Reggio Calabria", "Reggio Emilia", "Perugia", "Livorno", "Ravenna", "Cagliari", "Foggia", "Rimini", "Salerno", "Ferrara",
    "Sassari", "Latina", "Giugliano", "Monza", "Siracusa", "Pescara", "Bergamo", "Forlì", "Trento", "Vicenza",
    "Terni", "Bolzano", "Novara", "Piacenza", "Ancona", "Andria", "Arezzo", "Udine", "Cesena", "Lecce",
    "Pesaro", "Barletta", "Alessandria", "La Spezia", "Pisa", "Pistoia", "Lucca", "Como", "Brindisi", "Treviso",
    "Busto Arsizio", "Grosseto", "Sesto San Giovanni", "Varese", "Asti", "Caserta", "Cremona", "Gela", "Aprilia", "Pavia",
    "Massa", "Lamezia Terme", "Potenza", "Matera", "Campobasso", "Aosta", "Trapani", "Agrigento", "Caltanissetta", "Enna",
    "Ragusa", "Vercelli", "Biella", "Lodi", "Lecco", "Sondrio", "Imperia", "Savona", "Carrara", "Viterbo",
    "Rieti", "Frosinone", "Isernia", "Avellino", "Benevento", "Crotone", "Vibo Valentia", "Cosenza", "Catanzaro", "Nuoro",
    "Oristano", "Olbia", "Iglesias", "Cuneo", "Mantova", "Rovigo", "Belluno", "Pordenone", "Gorizia", "Monfalcone",
    "Chioggia", "Jesolo", "Bassano del Grappa", "Schio", "Thiene", "Valdagno", "Rovereto", "Merano", "Bressanone", "Empoli",
    "Sesto Fiorentino", "Scandicci", "Pontedera", "Viareggio", "Civitavecchia", "Velletri", "Pomezia", "Anzio", "Nettuno", "Cassino",
    "Sora", "Gaeta", "Formia", "Battipaglia", "Eboli", "Nocera Inferiore", "Castellammare di Stabia", "Torre del Greco", "Portici", "Ercolano",
    "Pozzuoli", "Afragola", "Casoria", "Acerra", "Aversa", "Maddaloni", "Molfetta", "Altamura", "Gravina in Puglia", "Monopoli",
    "Fasano", "Ostuni", "Martina Franca", "Manfredonia", "San Severo", "Cerignola", "Bisceglie", "Trani", "Corato", "Ruvo di Puglia",
    "Sassuolo", "Carpi", "Mirandola", "Faenza", "Lugo", "Imola", "Cento", "Comacchio", "Cervia", "Riccione",
    "Fano", "Senigallia", "Jesi", "Osimo", "Civitanova Marche", "Macerata", "Fermo", "Ascoli Piceno", "Camerino", "Spoleto",
    "Foligno", "Città di Castello", "Orvieto", "Narni", "Amelia", "Sulmona", "Teramo", "Chieti", "Vasto", "Lanciano",
    "Termoli", "Larino",
)

UNIVERSITA = (
    "Politecnico di Torino", "Politecnico di Milano", "Politecnico di Bari", "Università di Bologna", "Università di Padova",
    "Università di Firenze", "Università di Pisa", "Università di Siena", "Università di Perugia", "Università di Parma",
    "Università di Modena e Reggio Emilia", "Università di Ferrara", "Università di Trieste", "Università di Udine", "Università di Genova",
    "Università degli Studi di Milano", "Università di Pavia", "Università di Brescia", "Università di Bergamo", "Università di Verona",
    "Università di Venezia", "Università di Trento", "Università di Torino", "Università di Napoli Federico II", "Università di Salerno",
    "Università La Sapienza di Roma", "Università di Roma Tor Vergata", "Università Roma Tre", "Università di Bari Aldo Moro", "Università del Salento",
    "Università di Foggia", "Università della Calabria", "Università di Messina", "Università di Catania", "Università di Palermo",
    "Università di Cagliari", "Università di Sassari", "Università di Cassino", "Università di Teramo", "Università di Chieti-Pescara",
    "Università di Camerino", "Università di Macerata", "Università di Urbino", "Università Politecnica delle Marche", "Università Insubria",
    "Università del Piemonte Orientale", "Università del Molise", "Università della Basilicata", "Università del Sannio", "Università della Tuscia",
    "Università di Napoli Parthenope", "Università Suor Orsola Benincasa", "Università Iuav di Venezia", "Università Bocconi", "Università Cattolica del Sacro Cuore",
    "Università Luiss Guido Carli", "Università Vita-Salute San Raffaele", "Università IULM", "Università Europea di Roma", "Università Telematica Pegaso",
    "Università Mercatorum", "Università Magna Graecia di Catanzaro", "Università per Stranieri di Siena", "Università per Stranieri di Perugia", "Università del Foro Italico",
    "Università Link Campus", "Università di Milano Bicocca", "Università Statale di Milano", "Università Niccolò Cusano", "Università eCampus",
    "Università Guglielmo Marconi", "Università Campus Bio-Medico di Roma", "Università Kore di Enna", "Università Lumsa", "Università Pontificia Salesiana",
    "Università Pontificia Gregoriana", "Università Carlo Cattaneo LIUC", "Università Telematica Internazionale UNINETTUNO", "Scuola Normale Superiore", "Scuola Superiore Meridionale",
    "Scuola IMT Alti Studi Lucca", "Scuola Universitaria Superiore IUSS Pavia", "Università di Roma Foro Italico", "Università di Napoli Suor Orsola", "Università di Bologna Campus Cesena",
    "Università di Padova Scuola Galileiana", "Università di Firenze Scuola Superiore", "Università di Pisa Scuola Sant Anna", "Università di Trieste Area Scientifica", "Università di Verona Scuola Medica",
    "Accademia di Belle Arti di Brera", "Accademia di Belle Arti di Firenze", "Accademia di Belle Arti di Roma", "Accademia di Belle Arti di Venezia", "Conservatorio di Musica di Parma",
    "Istituto Universitario di Architettura di Venezia", "Università di Napoli Federico II Scuola Medica", "Università di Bari Scuola di Medicina", "Università di Genova Scuola di Scienze", "Università di Milano Scuola di Giurisprudenza",
    "Università di Torino Scuola di Medicina", "Università di Bologna Scuola di Agraria", "Università di Pavia Collegio Universitario", "Università di Sassari Scuola di Agraria", "Università di Cagliari Scuola di Ingegneria",
    "Università di Catania Scuola Superiore", "Università di Palermo Scuola di Medicina", "Università di Messina Scuola di Ingegneria", "Università di Salerno Scuola di Medicina", "Università di Padova Scuola di Medicina",
    "Università di Parma Scuola di Medicina", "Università di Ferrara Scuola di Medicina", "Università di Trento Scuola di Ingegneria", "Università di Bergamo Scuola di Ingegneria", "Università di Brescia Scuola di Medicina",
    "Università di Udine Scuola di Medicina", "Università di Foggia Scuola di Medicina", "Università di Teramo Scuola di Scienze", "Università di Urbino Scuola di Scienze", "Università di Camerino Scuola di Scienze",
)

CORSI = (
    "Ingegneria chimica", "Ingegneria civile", "Ingegneria elettrica", "Ingegneria meccanica", "Ingegneria informatica",
    "Ingegneria gestionale", "Ingegneria aerospaziale", "Ingegneria biomedica", "Ingegneria energetica", "Ingegneria edile",
    "Ingegneria dei materiali", "Ingegneria delle telecomunicazioni", "Ingegneria nucleare", "Ingegneria navale", "Ingegneria idraulica",
    "Ingegneria dei trasporti", "Ingegneria della sicurezza", "Ingegneria del territorio", "Ingegneria del software", "Ingegneria chimica industriale",
    "Architettura", "Design", "Urbanistica", "Pianificazione territoriale", "Medicina e chirurgia",
    "Farmacia", "Chimica e tecnologia farmaceutiche", "Chimica", "Chimica industriale", "Fisica",
    "Matematica", "Statistica", "Informatica", "Biologia", "Biotecnologie",
    "Scienze biologiche", "Scienze ambientali", "Scienze geologiche", "Scienze naturali", "Scienze della terra",
    "Astronomia", "Scienze dei materiali", "Scienze politiche", "Scienze della comunicazione", "Scienze della formazione primaria",
    "Giurisprudenza", "Diritto internazionale", "Diritto penale", "Economia aziendale", "Economia politica",
    "Economia e commercio", "Management", "Marketing", "Finanza", "Economia digitale",
    "Lettere classiche", "Lettere moderne", "Filosofia", "Storia", "Storia medievale",
    "Storia contemporanea", "Storia dell architettura", "Lingue e letterature straniere", "Lingue e culture moderne", "Mediazione linguistica",
    "Traduzione", "Psicologia", "Psicologia clinica", "Sociologia", "Antropologia culturale",
    "Geografia", "Archeologia", "Beni culturali", "Conservazione dei beni culturali", "Musicologia",
    "Scienze motorie", "Scienze infermieristiche", "Ostetricia", "Fisioterapia", "Odontoiatria",
    "Veterinaria", "Agraria", "Scienze agrarie", "Scienze forestali", "Scienze gastronomiche",
    "Scienze alimentari", "Tecnologie alimentari", "Enologia", "Scienze della nutrizione", "Dietistica",
    "Logopedia", "Tecniche di radiologia", "Scienze pedagogiche", "Pedagogia", "Relazioni internazionali",
    "Cooperazione internazionale", "Economia internazionale", "Commercio internazionale", "Data science", "Intelligenza artificiale",
    "Cybersicurezza", "Sistemi informativi", "Bioinformatica", "Neuroscienze", "Scienze cognitive",
    "Scienze della comunicazione digitale", "Giornalismo", "Editoria", "Cinema e audiovisivi", "Teatro e spettacolo",
    "Scienze dello sport", "Scienze turistiche", "Turismo e beni culturali", "Economia del turismo", "Hospitality management",
    "Automatica", "Ingegneria dell autoveicolo", "Ingegneria ferroviaria", "Scienze infermieristiche pediatriche", "Scienze della formazione",
)

PROFESSIONI = (
    "farmacista", "ingegnere", "medico", "avvocato", "architetto", "insegnante", "infermiere", "commercialista", "notaio", "geometra",
    "perito", "dentista", "veterinario", "psicologo", "psichiatra", "pediatra", "chirurgo", "cardiologo", "dermatologo", "oculista",
    "radiologo", "anestesista", "biologo", "chimico", "fisico", "matematico", "statistico", "informatico", "programmatore", "sistemista",
    "analista", "giornalista", "redattore", "traduttore", "interprete", "bibliotecario", "archivista", "archeologo", "storico", "filosofo",
    "economista", "consulente", "manager", "dirigente", "imprenditore", "commerciante", "agronomo", "enologo", "pasticciere", "cuoco",
    "sommelier", "falegname", "fabbro", "elettricista", "idraulico", "meccanico", "muratore", "carpentiere", "saldatore", "tornitore",
    "geologo", "sociologo", "linguista", "pedagogista", "logopedista", "fisioterapista", "ortopedico", "neurologo", "nutrizionista", "dietista",
    "farmacologo", "tecnologo", "ricercatore", "docente", "professore", "ragioniere", "bancario", "assicuratore", "agente immobiliare", "magistrato",
    "giudice", "poliziotto", "carabiniere", "pilota", "paesaggista", "scenografo", "designer", "fotografo", "regista", "attore",
    "musicista", "compositore", "scultore", "pittore", "illustratore", "grafico", "editore", "libraio", "postino", "agricoltore",
    "allevatore", "pescatore", "marinaio", "capitano", "giardiniere", "sarto", "parrucchiere", "estetista", "guida turistica", "catechista",
    "tipografo", "orafo", "gioielliere", "barbiere", "ottico", "tassista", "camionista", "promotore", "mediatore", "psicoterapeuta",
    "osteopata", "consigliere",
)

AZIENDE = (
    "Barilla", "Ferrero", "Luxottica", "Benetton", "Armani", "Prada", "Gucci", "Versace", "Bulgari", "Lamborghini",
    "Ducati", "Piaggio", "Pirelli", "Brembo", "Fincantieri", "Leonardo", "Eni", "Enel", "Terna", "Snam",
    "Italgas", "Telecom Italia", "Mediaset", "Generali", "Unipol", "Intesa Sanpaolo", "UniCredit", "Banca Mediolanum", "Mediobanca", "Banco BPM",
    "Poste Italiane", "Trenitalia", "ITA Airways", "Iveco", "Dallara", "Maserati", "Alfa Romeo", "Lancia", "Magneti Marelli", "Beretta",
    "Campari", "Peroni", "Lavazza", "Illy", "Segafredo", "Perugina", "Motta", "Bauli", "Loacker", "Galbani",
    "Parmalat", "Granarolo", "Mutti", "Cirio", "Bertolli", "Carapelli", "Riso Scotti", "Buitoni", "Oviesse", "Coin",
    "Rinascente", "Esselunga", "Conad", "Coop", "Despar", "Eurospin", "Calzedonia", "Intimissimi", "Tezenis", "Diadora",
    "Kappa", "Geox", "Tod's", "Bottega Veneta", "Valentino", "Missoni", "Max Mara", "Replay", "Diesel", "Stone Island",
    "Moncler", "Colmar", "Kiton", "Brioni", "Zegna", "Borsalino", "Safilo", "Marcolin", "Mapei", "Saipem",
    "Webuild", "Danieli", "Tenaris", "Fastweb", "Iren", "A2A", "Hera", "Acea", "Edison", "Erg",
    "Saras", "Cerved", "Dedagroup", "Engineering", "Reply", "Almaviva", "Exprivia", "Sogei", "Bending Spoons", "Dolomiti Energia",
    "Ansaldo Energia", "Oto Melara", "Avio", "Bonfiglioli", "Carraro", "Ariston", "Candy", "Indesit", "Smeg", "Zanussi",
    "Biesse", "Comau", "Olivetti", "Brunello Cucinelli", "Salvatore Ferragamo", "Alessi", "Kartell", "Natuzzi", "Ferretti Yachts", "Technogym",
    "Vittoria Assicurazioni", "Cattolica Assicurazioni", "Reale Mutua", "Allianz Italia", "Zurich Italia", "Italiaonline", "Cementir", "Buzzi", "Ceramica Caesar", "Marazzi Group",
    "Bialetti", "De Longhi", "Smeg Italia", "Ilva", "Acciaierie Venete", "Feltrinelli", "Mondadori", "Rizzoli", "Gruppo Sole", "Gruppo Campari",
    "Cucine Lube", "Scavolini", "Natuzzi Group", "Poltrona Frau", "Cassina", "Artemide", "Foscarini", "Flos", "Luceplan", "Vitra Italia",
)

MESI = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)

# Forme di genere per i participi e gli aggettivi
FORME_GENERE = {
    "F": {"nat": "nata", "orig": "originaria", "lau": "laureata", "imp": "impiegata"},
    "M": {"nat": "nato", "orig": "originario", "lau": "laureato", "imp": "impiegato"},
}
PRONOME = {"F": "Lei", "M": "Lui"}

# Template per formato e attributo; ogni template termina con {v}
PIENO = {
    "nascita_data": [
        "è {nat} il {v}", "ha visto la luce il {v}", "ha i natali il {v}", "ha come data di nascita il {v}",
    ],
    "nascita_luogo": [
        "è {nat} a {v}", "è {orig} di {v}", "ha visto la luce a {v}", "ha i natali a {v}",
    ],
    "universita": [
        "ha studiato presso {v}", "ha conseguito la laurea presso {v}", "ha frequentato {v}", "ha fatto gli studi presso {v}",
    ],
    "corso": [
        "si è {lau} in {v}", "ha conseguito il titolo in {v}", "ha seguito il corso di {v}", "ha scelto il percorso di {v}",
    ],
    "professione": [
        "lavora come {v}", "è {v}", "esercita la professione di {v}", "ha il ruolo di {v}",
    ],
    "azienda": [
        "ha lavorato presso {v}", "è {imp} presso {v}", "collabora con {v}", "lavora per {v}",
    ],
}

TELEGRAFICO = {
    "nascita_data": ["{nat} data {v}", "data nascita {v}"],
    "nascita_luogo": ["{nat} luogo {v}", "luogo nascita {v}"],
    "universita": ["studi {v}", "formazione {v}"],
    "corso": ["{lau} {v}", "corso {v}"],
    "professione": ["professione {v}", "lavoro {v}"],
    "azienda": ["azienda {v}", "impiego {v}"],
}

TEMPLATE = {"italiano_pieno": PIENO, "telegrafico": TELEGRAFICO}
MARCATORI = frozenset(INTERLINGUA_MARKERS.values())

_DATA_INIZIO = date(1950, 1, 1)
_GIORNI = (date(2005, 12, 31) - _DATA_INIZIO).days + 1


def _data_nascita(rng: random.Random) -> str:
    d = _DATA_INIZIO + timedelta(days=rng.randrange(_GIORNI))
    return f"{d.day} {MESI[d.month - 1]} {d.year}"


def _maiuscola(testo: str) -> str:
    return testo[0].upper() + testo[1:]


def _predicato(person: Person, attr: str, tpl: str) -> str:
    return tpl.format(v=person.attrs[attr], **FORME_GENERE[person.gender])


def _prefisso(fmt: str, attr: str, gender: str) -> str:
    """Prefisso del template 0 (o marcatore), senza nome e senza valore."""
    if fmt == "interlingua":
        return INTERLINGUA_MARKERS[attr]
    tpl = TEMPLATE[fmt][attr][0]
    return tpl.split("{v}")[0].strip().format(**FORME_GENERE[gender])


def generate_persons(n: int, seed: int) -> list[Person]:
    """N persone con nome+cognome unici; data nascita "12 marzo 1987" (1950-2005)."""
    n_cognomi = len(COGNOMI)
    n_f = len(NOMI_F) * n_cognomi
    capacita = n_f + len(NOMI_M) * n_cognomi
    if n > capacita:
        raise ValueError(f"n={n} supera la capacità di nomi unici ({capacita})")
    rng = random.Random(seed)
    persone = []
    # Indici distinti su tutte le combinazioni nome x cognome: nomi unici per costruzione
    for idx in rng.sample(range(capacita), n):
        if idx < n_f:
            genere, nomi, pos = "F", NOMI_F, idx
        else:
            genere, nomi, pos = "M", NOMI_M, idx - n_f
        nome = nomi[pos // n_cognomi]
        cognome = COGNOMI[pos % n_cognomi]
        attrs = {
            "nascita_data": _data_nascita(rng),
            "nascita_luogo": rng.choice(CITTA),
            "universita": rng.choice(UNIVERSITA),
            "corso": rng.choice(CORSI),
            "professione": rng.choice(PROFESSIONI),
            "azienda": rng.choice(AZIENDE),
        }
        persone.append(Person(f"{nome} {cognome}", genere, attrs))
    return persone


def _render_pieno(person: Person, rng: random.Random) -> str:
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    frasi = []
    for i, attr in enumerate(ordine):
        pred = _predicato(person, attr, rng.choice(PIENO[attr]))
        if i == 0:
            frase = f"{person.name} {pred}"
        elif rng.random() < 0.5:
            frase = f"{PRONOME[person.gender]} {pred}"
        else:
            frase = _maiuscola(pred)
        frasi.append(frase + ".")
    return " ".join(frasi)


def _render_telegrafico(person: Person, rng: random.Random) -> str:
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    frasi = []
    for i, attr in enumerate(ordine):
        pred = _predicato(person, attr, rng.choice(TELEGRAFICO[attr]))
        frase = f"{person.name} {pred}" if i == 0 else _maiuscola(pred)
        frasi.append(frase + ".")
    return " ".join(frasi)


def _render_interlingua(person: Person, rng: random.Random) -> str:
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    parti = [person.name]
    parti.extend(f"{INTERLINGUA_MARKERS[a]} {person.attrs[a]}" for a in ordine)
    return " ".join(parti)


def render(person: Person, fmt: str, rng: random.Random) -> str:
    """Una biografia nel formato fmt, con template e ordine dei campi pescati da rng."""
    if fmt == "italiano_pieno":
        return _render_pieno(person, rng)
    if fmt == "telegrafico":
        return _render_telegrafico(person, rng)
    if fmt == "interlingua":
        return _render_interlingua(person, rng)
    raise ValueError(f"formato sconosciuto: {fmt}")


def eval_prompt(person: Person, attr: str, fmt: str) -> str:
    """Prompt nativo che termina subito prima del valore (senza spazio finale)."""
    return f"{person.name} {_prefisso(fmt, attr, person.gender)}"


def extract_answer(generated: str, fmt: str) -> str:
    """Taglia la continuazione generata al terminatore del valore e fa strip."""
    testo = generated.split(EOS, 1)[0]
    if fmt == "interlingua":
        parole = []
        for parola in testo.split():
            if parola in MARCATORI:
                break
            parole.append(parola)
        return " ".join(parole)
    if fmt in TEMPLATE:
        return re.split(r"[.\n]", testo, maxsplit=1)[0].strip()
    raise ValueError(f"formato sconosciuto: {fmt}")


def epoch_texts(persons: list[Person], fmt: str, epoch: int, seed: int) -> list[str]:
    """Tutte le biografie per un'epoca, in ordine casuale; deterministica in (seed, fmt, epoch)."""
    rng = random.Random(f"{seed}|{fmt}|{epoch}")
    ordine = list(persons)
    rng.shuffle(ordine)
    return [render(p, fmt, rng) for p in ordine]
