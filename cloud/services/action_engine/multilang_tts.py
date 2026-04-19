"""Multi-language voice deterrence — TTS in any language based on site config."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LANGUAGE_TEMPLATES = {
    "en": {1: "Attention: this area is monitored.", 2: "Warning: leave this area immediately.",
           3: "Final warning: security has been notified.", 4: "Security dispatched. Remain where you are."},
    "es": {1: "Atención: esta área está vigilada.", 2: "Advertencia: abandone esta área inmediatamente.",
           3: "Última advertencia: se ha notificado a seguridad.", 4: "Seguridad en camino. Permanezca donde está."},
    "fr": {1: "Attention: cette zone est surveillée.", 2: "Avertissement: quittez cette zone immédiatement.",
           3: "Dernier avertissement: la sécurité a été prévenue.", 4: "Sécurité en route. Restez où vous êtes."},
    "de": {1: "Achtung: Dieser Bereich wird überwacht.", 2: "Warnung: Verlassen Sie diesen Bereich sofort.",
           3: "Letzte Warnung: Sicherheit wurde benachrichtigt.", 4: "Sicherheit unterwegs. Bleiben Sie wo Sie sind."},
    "hi": {1: "ध्यान दें: यह क्षेत्र निगरानी में है।", 2: "चेतावनी: इस क्षेत्र को तुरंत छोड़ दें।",
           3: "अंतिम चेतावनी: सुरक्षा को सूचित किया गया है।", 4: "सुरक्षा भेजी गई है। जहाँ हैं वहीं रहें।"},
    "ar": {1: "تنبيه: هذه المنطقة مراقبة.", 2: "تحذير: غادر هذه المنطقة فوراً.",
           3: "تحذير أخير: تم إبلاغ الأمن.", 4: "الأمن في الطريق. ابقَ مكانك."},
    "zh": {1: "注意：此区域受到监控。", 2: "警告：请立即离开此区域。",
           3: "最后警告：已通知安保人员。", 4: "安保已派出。请留在原地。"},
    "ja": {1: "注意：この区域は監視されています。", 2: "警告：直ちにこの区域から離れてください。",
           3: "最終警告：セキュリティに通知しました。", 4: "セキュリティが向かっています。その場にいてください。"},
    "pt": {1: "Atenção: esta área é monitorada.", 2: "Aviso: saia desta área imediatamente.",
           3: "Último aviso: a segurança foi notificada.", 4: "Segurança a caminho. Permaneça onde está."},
    "ko": {1: "주의: 이 구역은 감시 중입니다.", 2: "경고: 즉시 이 구역을 떠나십시오.",
           3: "최종 경고: 보안팀에 통보되었습니다.", 4: "보안팀이 출동했습니다. 그 자리에 계십시오."},
}


@dataclass
class TTSRequest:
    text: str
    language: str
    camera_id: str
    level: int


class MultiLanguageTTS:
    """Generates deterrence messages in configured language with optional cloud TTS."""

    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self._site_languages: dict[str, str] = {}  # camera_id -> language code

    def set_site_language(self, camera_id: str, language: str):
        self._site_languages[camera_id] = language

    def get_message(self, camera_id: str, level: int, zone_name: str = "") -> TTSRequest:
        lang = self._site_languages.get(camera_id, self.default_language)
        templates = LANGUAGE_TEMPLATES.get(lang, LANGUAGE_TEMPLATES["en"])
        text = templates.get(level, templates[1])
        if zone_name:
            text = text.replace("this area", zone_name).replace("esta área", zone_name)
        return TTSRequest(text=text, language=lang, camera_id=camera_id, level=level)

    def synthesize(self, request: TTSRequest) -> bytes | None:
        """Optional: call cloud TTS API. Returns audio bytes or None."""
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=request.text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=request.language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            return response.audio_content
        except Exception as e:
            logger.debug("Cloud TTS unavailable: %s", e)
            return None
