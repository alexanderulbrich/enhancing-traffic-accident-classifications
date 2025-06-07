# Contains prompts to be used in scripts

ZERO_SHOT_PROMPT = """
Aufgabe:
Du bist ein Klassifikationsassistent. 
Basierend auf den gegebenen Definitionen der Kategorien sollst du der angegebenen Textpassage die passendste Nummer eines Typs zuordnen und das Ergebnis im folgenden JSON-Format ausgeben:

{"typ": <Nummer des Typs>}

Definitionen der Kategorien:
	1.	Typ 1: Fahrunfall = Der Unfall wurde ausgelöst durch den Verlust der Kontrolle über das Fahrzeug (wegen nicht angepasster Geschwindigkeit oder falscher Einschätzung des Straßenverlaufs, des Straßenzustandes o. Ä.), ohne dass andere Verkehrsteilnehmer*innen dazu beigetragen haben. Infolge unkontrollierter Fahrzeugbewegungen kann es dann aber zum Zusammenstoß mit anderen Verkehrsteilnehmern*innen gekommen sein.
	2.	Typ 2: Abbiege-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem, den Vorrang Anderer zu beachtenden Abbieger und einem aus gleicher oder entgegengesetzter Richtung kommenden Verkehrsteilnehmer*innen (auch Fußgänger*innen!) an Kreuzungen, Einmündungen, Grundstücks- oder Parkplatzzufahrten.
	3.	Typ 3: Einbiegen/Kreuzen-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einbiegenden oder kreuzenden Wartepflichtigen und einem vorfahrtberechtigten Fahrzeug an Kreuzungen, Einmündungen oder Ausfahrten von Grundstücken und Parkplätzen.
	4.	Typ 4: Überschreiten-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug und einer Fußgänger*in auf der Fahrbahn, sofern dieser nicht in Längsrichtung ging und sofern das Fahrzeug nicht abgebogen ist. Dies gilt auch, wenn die Fußgänger*in nicht angefahren wurde.
	5.	Typ 5: Unfall durch ruhenden Verkehr = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug des fließenden Verkehrs und einem Fahrzeug, das parkt/hält bzw. Fahrmanöver im Zusammenhang mit dem Parken/Halten durchführte.
	6.	Typ 6: Unfall im Längsverkehr = Der Unfall wurde ausgelöst durch einen Konflikt zwischen Verkehrsteilnehmer*innen, die sich in gleicher oder entgegengesetzter Richtung bewegten, sofern dieser Konflikt nicht einem anderen Unfalltyp entspricht.
	7.	Typ 7: Sonstiger Unfall = Unfälle, die sich nicht den anderen Typen zuordnen lassen. Beispiele: Wenden, Rückwärtsfahren, Parker untereinander, Hindernis oder Tier auf der Fahrbahn, plötzlicher Fahrzeugschaden (Bremsversagen, Reifenschäden o. Ä.).

Anweisungen:
	•	Vergleiche den Text mit den Definitionen der Typen.
	•	Gib nur die Nummer des Typs an, der am besten zum Text passt, und formatiere die Antwort im JSON-Format, z. B.: {"typ": 3}.

Text:
"""

FEW_SHOT_PROMPT = """
Aufgabe:
Du bist ein Klassifikationsassistent. 
Basierend auf den gegebenen Definitionen der Kategorien sollst du der angegebenen Textpassage die passendste Nummer eines Typs zuordnen und das Ergebnis im folgenden JSON-Format ausgeben:

{"typ": <Nummer des Typs>}

Definitionen der Kategorien:
	1.	Typ 1: Fahrunfall = Der Unfall wurde ausgelöst durch den Verlust der Kontrolle über das Fahrzeug (wegen nicht angepasster Geschwindigkeit oder falscher Einschätzung des Straßenverlaufs, des Straßenzustandes o. Ä.), ohne dass andere Verkehrsteilnehmer*innen dazu beigetragen haben. Infolge unkontrollierter Fahrzeugbewegungen kann es dann aber zum Zusammenstoß mit anderen Verkehrsteilnehmern*innen gekommen sein.
	2.	Typ 2: Abbiege-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem, den Vorrang Anderer zu beachtenden Abbieger und einem aus gleicher oder entgegengesetzter Richtung kommenden Verkehrsteilnehmer*innen (auch Fußgänger*innen!) an Kreuzungen, Einmündungen, Grundstücks- oder Parkplatzzufahrten.
	3.	Typ 3: Einbiegen/Kreuzen-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einbiegenden oder kreuzenden Wartepflichtigen und einem vorfahrtberechtigten Fahrzeug an Kreuzungen, Einmündungen oder Ausfahrten von Grundstücken und Parkplätzen.
	4.	Typ 4: Überschreiten-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug und einer Fußgänger*in auf der Fahrbahn, sofern dieser nicht in Längsrichtung ging und sofern das Fahrzeug nicht abgebogen ist. Dies gilt auch, wenn die Fußgänger*in nicht angefahren wurde.
	5.	Typ 5: Unfall durch ruhenden Verkehr = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug des fließenden Verkehrs und einem Fahrzeug, das parkt/hält bzw. Fahrmanöver im Zusammenhang mit dem Parken/Halten durchführte.
	6.	Typ 6: Unfall im Längsverkehr = Der Unfall wurde ausgelöst durch einen Konflikt zwischen Verkehrsteilnehmer*innen, die sich in gleicher oder entgegengesetzter Richtung bewegten, sofern dieser Konflikt nicht einem anderen Unfalltyp entspricht.
	7.	Typ 7: Sonstiger Unfall = Unfälle, die sich nicht den anderen Typen zuordnen lassen. Beispiele: Wenden, Rückwärtsfahren, Parker untereinander, Hindernis oder Tier auf der Fahrbahn, plötzlicher Fahrzeugschaden (Bremsversagen, Reifenschäden o. Ä.).

Anweisungen:
	•	Vergleiche den Text mit den Definitionen der Typen.
	•	Gib nur die Nummer des Typs an, der am besten zum Text passt, und formatiere die Antwort im JSON-Format, z. B.: {"typ": 3}.

Beispiel1: 01 mit seinem Fahrrad auf dem Radweg, Isarring/ Dietlindenstraße, östliche Fahrtrichtung, eigenverschuldet gestürzt und verletzt.|VU wurde von unbeteiligtem ZEG M. beobachtet.|Der 01 mit dem Retter in das KH Schwabing.|Fahrrad des 01 an dortiger Örtlichkeit mit seinem eigenen Fahrradschloss abgestellt und abgesperrt.|Lichtbilder gefertigt.
Antwort: {‘typ’: 1}

Beispiel2: 01 befuhr mit seinem Pkw die Schleißheimer Straße in südlicher Richtung. 01 wollte nach rechts auf einen Parkstreifen abbiegen und übersah dabei den rechts neben ihm fahrenden 02 auf dessen Fahrrad, welcher auf der Fahrradspur neben der Fahrbahn fuhr. ||01 unverletzt, Pkw 01 leicht beschädigt.|02 leicht verletzt, verweigerte Verbringung ins KH, Sachschaden am Fahrrad des 02 noch unbekannt, lässt dieser noch überprüfen.
Antwort: {‘typ’: 2}

Beispiel3: 01 fuhr auf der Esswurmstr. in FR: südlich. 02 fuhr zur gleichen Zeit auf der Gaißacher Str. in FR: östlich. An der Kreuzung Esswurmstr./Gaißacher Str. übersah 01 den von rechts kommenden vorfahrtsberechtigten 02. Im Kreuzungsbereich kam es zur Berührung beider Fahrzeuge. Lichtbilder gefertigt.
Antwort: {‘typ’: 3}

Beispiel4: 02 befuhr mit ihrem Fahrzeug das Tal in östliche Richtung. Auf Höhe der Hausnummer 23 lief 01 plötzlich hinter einem geparkten Fahrzeug hervor und wollte, ohne auf den Verkehr zu achten, die Fahrbahn überqueren. ||02 konnte nicht mehr rechtzeitig zum Stehen kommen und touchierte 01 leicht. 01 verdrehte sich beim Zurückschrecken seine Knie.||Nachdem die Eltern von 01 informiert wurden, wurde dieser durch 02 ins Rechts der Isar verbracht.
Antwort: {‘typ’: 4}

Beispiel5: 02 parkte seinen Pkw in der Kundengarage des Kaufland. Dort hat ein unbekannter 01 die Beifahrertüre des 02 beschädigt. Vermutlich durch aufschlagen der Türe. 01 unbekannt. Keine Videoaufzeichnung vorhanden.
Antwort: {‘typ’: 5}

Beispiel6: 01 befuhr die Humboldstraße in südöstlicher Richtung. Zur gleichen Zeit befuhr 02 diese ebenfalls in gleiche Fahrtrichtung. 02 musste verkehrsbedingt bremsen. 01 bemerkte dies zu spät, konnte nicht rechtzeitig bremsen und es kam zum Zusammenstoß. 02 stand unter Schock und verspürte Schmerzen im Nackenbereich. 02 wurde mit dem RTW Ludwigsvorstadt 71/1 in das Klinikum Rinecker verbracht. |01 unverletzt. PKW 01 und 02 jeweils leicht beschädigt.  Lichtbilder gefertigt und Unfallskizze erstellt.
Antwort: {‘typ’: 6}

Text:
"""

ADJUSTED_FEW_SHOT_PROMPT = """
Aufgabe:
Du bist ein Klassifikationsassistent. 
Basierend auf den gegebenen Definitionen der Kategorien sollst du der angegebenen Textpassage die passendste Nummer eines Typs zuordnen und das Ergebnis im folgenden JSON-Format ausgeben:

{"typ": <Nummer des Typs>}

Definitionen der Kategorien:
	1.	Typ 1: Fahrunfall = Der Unfall wurde ausgelöst durch den Verlust der Kontrolle über das Fahrzeug (wegen nicht angepasster Geschwindigkeit oder falscher Einschätzung des Straßenverlaufs, des Straßenzustandes o. Ä.), ohne dass andere Verkehrsteilnehmer*innen dazu beigetragen haben. Infolge unkontrollierter Fahrzeugbewegungen kann es dann aber zum Zusammenstoß mit anderen Verkehrsteilnehmern*innen gekommen sein.
	2.	Typ 2: Abbiege-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem, den Vorrang Anderer zu beachtenden Abbieger und einem aus gleicher oder entgegengesetzter Richtung kommenden Verkehrsteilnehmer*innen (auch Fußgänger*innen!) an Kreuzungen, Einmündungen, Grundstücks- oder Parkplatzzufahrten.
	3.	Typ 3: Einbiegen/Kreuzen-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einbiegenden oder kreuzenden Wartepflichtigen und einem vorfahrtberechtigten Fahrzeug an Kreuzungen, Einmündungen oder Ausfahrten von Grundstücken und Parkplätzen.
	4.	Typ 4: Überschreiten-Unfall = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug und einer Fußgänger*in auf der Fahrbahn, sofern dieser nicht in Längsrichtung ging und sofern das Fahrzeug nicht abgebogen ist. Dies gilt auch, wenn die Fußgänger*in nicht angefahren wurde.
	5.	Typ 5: Unfall durch ruhenden Verkehr = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug des fließenden Verkehrs und einem Fahrzeug, das parkt/hält bzw. Fahrmanöver im Zusammenhang mit dem Parken/Halten durchführte.
	6.	Typ 6: Unfall im Längsverkehr = Der Unfall wurde ausgelöst durch einen Konflikt zwischen Verkehrsteilnehmer*innen, die sich in gleicher oder entgegengesetzter Richtung bewegten, sofern dieser Konflikt nicht einem anderen Unfalltyp entspricht.
	7.	Typ 7: Sonstiger Unfall = Unfälle, die sich nicht den anderen Typen zuordnen lassen. Beispiele: Wenden, Rückwärtsfahren, Parker untereinander, Hindernis oder Tier auf der Fahrbahn, plötzlicher Fahrzeugschaden (Bremsversagen, Reifenschäden o. Ä.).

Anweisungen:
	•	Vergleiche den Text mit den Definitionen der Typen.
	•	Prüfe sorgfältig die Typen 1 bis 6, bevor du Typ 7 wählst. Typ 7 ist ausschließlich eine Fallback-Kategorie, die nur verwendet werden soll, wenn sich der Unfall eindeutig keinem der anderen Typen zuordnen lässt.
	•	Gib nur die Nummer des Typs an, der am besten zum Text passt, und formatiere die Antwort im JSON-Format, z. B.: {"typ": 3}.

Beispiel1: 01 mit seinem Fahrrad auf dem Radweg, Isarring/ Dietlindenstraße, östliche Fahrtrichtung, eigenverschuldet gestürzt und verletzt.|VU wurde von unbeteiligtem ZEG M. beobachtet.|Der 01 mit dem Retter in das KH Schwabing.|Fahrrad des 01 an dortiger Örtlichkeit mit seinem eigenen Fahrradschloss abgestellt und abgesperrt.|Lichtbilder gefertigt.
Antwort: {‘typ’: 1}

Beispiel2: 01 befuhr mit seinem Pkw die Schleißheimer Straße in südlicher Richtung. 01 wollte nach rechts auf einen Parkstreifen abbiegen und übersah dabei den rechts neben ihm fahrenden 02 auf dessen Fahrrad, welcher auf der Fahrradspur neben der Fahrbahn fuhr. ||01 unverletzt, Pkw 01 leicht beschädigt.|02 leicht verletzt, verweigerte Verbringung ins KH, Sachschaden am Fahrrad des 02 noch unbekannt, lässt dieser noch überprüfen.
Antwort: {‘typ’: 2}

Beispiel3: 01 fuhr auf der Esswurmstr. in FR: südlich. 02 fuhr zur gleichen Zeit auf der Gaißacher Str. in FR: östlich. An der Kreuzung Esswurmstr./Gaißacher Str. übersah 01 den von rechts kommenden vorfahrtsberechtigten 02. Im Kreuzungsbereich kam es zur Berührung beider Fahrzeuge. Lichtbilder gefertigt.
Antwort: {‘typ’: 3}

Beispiel4: 02 befuhr mit ihrem Fahrzeug das Tal in östliche Richtung. Auf Höhe der Hausnummer 23 lief 01 plötzlich hinter einem geparkten Fahrzeug hervor und wollte, ohne auf den Verkehr zu achten, die Fahrbahn überqueren. ||02 konnte nicht mehr rechtzeitig zum Stehen kommen und touchierte 01 leicht. 01 verdrehte sich beim Zurückschrecken seine Knie.||Nachdem die Eltern von 01 informiert wurden, wurde dieser durch 02 ins Rechts der Isar verbracht.
Antwort: {‘typ’: 4}

Beispiel5: 02 parkte seinen Pkw in der Kundengarage des Kaufland. Dort hat ein unbekannter 01 die Beifahrertüre des 02 beschädigt. Vermutlich durch aufschlagen der Türe. 01 unbekannt. Keine Videoaufzeichnung vorhanden.
Antwort: {‘typ’: 5}

Beispiel6: 01 befuhr die Humboldstraße in südöstlicher Richtung. Zur gleichen Zeit befuhr 02 diese ebenfalls in gleiche Fahrtrichtung. 02 musste verkehrsbedingt bremsen. 01 bemerkte dies zu spät, konnte nicht rechtzeitig bremsen und es kam zum Zusammenstoß. 02 stand unter Schock und verspürte Schmerzen im Nackenbereich. 02 wurde mit dem RTW Ludwigsvorstadt 71/1 in das Klinikum Rinecker verbracht. |01 unverletzt. PKW 01 und 02 jeweils leicht beschädigt.  Lichtbilder gefertigt und Unfallskizze erstellt.
Antwort: {‘typ’: 6}

Text:
"""

DAMAGED_PARKED_VEHICLE_ANALYSIS_PROMPT = """
Du bist ein Klassifikationsagent.
Basierend auf den gegebenen Definitionen der Kategorien sollst du die passendste Nummer eines Typs der gegebenen Textpassage zuweisen und das Ergebnis im folgenden JSON-Format ausgeben:

{"typ": <Nummer des Typs>}

Typ 1 = Der Unfall wurde ausgelöst durch einen Konflikt zwischen einem Fahrzeug des fließenden Verkehrs und einem Fahrzeug, das parkt/hält bzw. Fahrmanöver im Zusammenhang mit dem Parken/Halten durchführte
Typ 2 = Der Unfall wurde ausgelöst durch einen Konflikt zwischen zwei oder mehr Fahrzeugen, die parkten/hielten bzw. ein Fahrmanöver im Zusammenhang mit dem Parken/Halten durchführten.
Typ 3 = Ein geparktes Fahrzeug wurde durch eine unbekannte Quelle beschädigt.
Text:
"""