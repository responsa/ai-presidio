
import json
import logging      
from typing import Any, List, Tuple

import requests

from presidio_analyzer import RemoteRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

logger = logging.getLogger("presidio-analyzer")


class AzureAiRemoteRecognizer(RemoteRecognizer):
    """
    A reference implementation of a remote recognizer.

    Calls Presidio analyzer as if it was an external remote PII detector
    :param pii_identification_url: Service URL for detecting PII
    :param supported_entities_url: Service URL for getting the supported entities
    by this service
    """

    def __init__(
        self,
        pii_identification_url: str,
        api_key: str,
    ):
        self.pii_identification_url = pii_identification_url
        self.api_key = api_key

        super().__init__(
            supported_entities=[], name=None, supported_language="it", version="1.0"
        )

    def load(self) -> None:
        logger.info("Loading the remote recognizer")
        self.supported_entities = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "IBAN_CODE", "POLICY_NUMBER", "CREDIT_CARD", "PHONE_NUMBER", "IT_FISCAL_CODE", "IT_IDENTITY_CARD", "IT_PASSPORT", "IT_DRIVER_LICENSE"]

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
    ) -> any: #List[RecognizerResult]:
        """Call an external service for PII detection."""

        # Configuration
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Payload for the request
        payload = {
            "messages": [
                {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "sei un NeuralEntityRecognizer che ha il compito di riconoscere le entità presenti in un testo e assegnare una categoria a ciascuna. Verrai utilizzato in un processo per riconoscere dati importanti presenti nelle stringhe di input che dovranno successivamente venir anonimizzati.\n \nLe categorie di entità che sai riconoscere sono solo le seguenti:\nPERSON: nome e cognome proprio di una persona fisica, può anche essere in forma di iniziali\nLOCATION: indirizzo, posizione geografica, nome di un paese o città o provincia o regione\nEMAIL_ADDRESS: indirizzo email\nIBAN_CODE: codice IBAN\nPOLICY_NUMBER: codice identificativo di una polizza assicurativa\nCREDIT_CARD: numero della carta di credito o debito o bancomat\nPHONE_NUMBER: numero di telefono fisso o di cellulare\nIT_FISCAL_CODE: codice fiscale emesso dallo stato italiano, partita IVA emessa dallo stato italiano\nIT_IDENTITY_CARD: numero di carta d'identità emessa dallo stato italiano\nIT_PASSPORT: numero di passaporto emesso dallo stato italiano\nIT_DRIVER_LICENSE: numero della patente emessa dallo stato italiano\n \nL'analisi deve essere condotta in modo scrupoloso, con la massima attenzione; devi individuare anche le entità che possono essere state leggermente alterate, ad esempio per sottolineare porzioni del testo . Sii scrupoloso nel analizzare le entità lunghe che possono contenere più entità: in questi casi nel risultato JSON riporta tutte le possibilità, utilizzando un valore di score in base al grado di confidenza\nDi seguito alcuni esempi di entità alterate e la loro corretta interpretazione:\nR-S-S-M-R-A-8-5-M-0-1-H-5-0-1-Z é una rappresentazione del codice fiscale RSSMRA85M01H501Z\nRSS/MRA/85M01/H501/Z é una rappresentazione del codice fiscale RSSMRA85M01H501Z\n \nRitorna il risultato finale in formato JSON delle sole entità che hai riconosciuto, seguendo queste indicazioni:\n{\"result\" : [{\"recognition_metadata\": {\"recognizer_name\": \"ResposaRecognizer\"} \"analysis_explanation\": null, \"entity_type\": la categoria dell' entità individuata, \"entity\": l'entità individuata così come è scritta nel testo originale, \"score\": la tua fiducia nel riconoscere il tipo di entità espressa con un valore numerico compreso tra 0.00 e 1.00 - é importante che anche piccole incertezze siano rappresentate in forma numerica e nel caso di più entità sovrapposte, quella più affidabile abbia un valore maggiore delle altre }]}"
                    }
                ]
                },
                {"role": "user",
                 "content": [
                    {
                        "type": "text",
                        "text": text 
                    }
                 ],}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 4000,
            "response_format": {
                "type": "json_object"
            },
        }

        # Send request
        try:
            response = requests.post(self.pii_identification_url, headers=headers, json=payload)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            return self.map_recognizer_results_from_response(response, text)
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

    def get_supported_entities(self) -> List[str]:
        """Return the list of supported entities."""
        return self.supported_entities

    def map_recognizer_results_from_response(self, response: requests.Response, text: str) -> List[RecognizerResult]:
        response_content = response.json().get("choices")[0].get("message").get("content")
        results = json.loads(response_content)
        current_text = text
        recognizer_results = []
        for result in results.get("result"):
            position = self.find_first_word_positions(text, result.get("entity"))
            current_text = current_text[position[1]:]
            recognizer_results.append(RecognizerResult(result.get("entity_type"), position[0], position[1], result.get("score"), None, result.get("recognition_metadata")))
        return recognizer_results

    @staticmethod
    def _recognizer_results_from_response(
        response: Any,
    ) -> List[RecognizerResult]:
        """Translate the service's response to a list of RecognizerResult."""
        recognizer_results = [RecognizerResult(**result) for result in response]

        return recognizer_results

    def find_first_word_positions(self, text: str, word: str) -> Tuple[int, int]:
        """
        Trova tutte le posizioni in cui la parola inizia e finisce nel testo.

        :param text: Il testo in cui cercare la parola.
        :param word: La parola da cercare nel testo.
        :return: Una lista di tuple con le posizioni di inizio e fine della parola.
        """
        positions = []
        start = 0
        while start < len(text):
            start = text.find(word, start)
            if start == -1:
                break
            end = start + len(word)
            positions.append((start, end))
            start += len(word)  # Continua la ricerca dopo la parola trovata
        return positions[0]