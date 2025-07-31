from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
import logging
import time
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, ValidationError, Field, validator
from openai import OpenAI
from rapidfuzz import fuzz
from dotenv import load_dotenv
import random
from enum import Enum
import re
import bleach
import hashlib
from functools import wraps

# Güvenlik konfigürasyonu
SECURITY_CONFIG = {
    'MAX_MESSAGE_LENGTH': 500,
    'MAX_REQUESTS_PER_MINUTE': 60,
    'MAX_REQUESTS_PER_HOUR': 300,
    'REQUEST_TIMEOUT': 30,
    'MAX_RESPONSE_LENGTH': 2000,
    'ALLOWED_CHARS': re.compile(r'^[a-zA-ZığüşöçİĞÜŞÖÇ0-9\s.,!?()-]+$'),
    'BLOCKED_PATTERNS': [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload=',
        r'onerror=',
        r'eval\(',
        r'exec\(',
        r'import\s+os',
        r'__import__',
        r'\bfile\b.*open',
        r'subprocess',
        r'system\(',
        r'shell=True'
    ]
}

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sanitize_input(text: str) -> str:
    """Güvenli input sanitization"""
    if not isinstance(text, str):
        return ""
    
    # Uzunluk kontrolü
    text = text[:SECURITY_CONFIG['MAX_MESSAGE_LENGTH']]
    
    # HTML/Script temizleme
    text = bleach.clean(text, tags=[], attributes={}, strip=True)
    
    # Zararlı pattern kontrolü
    for pattern in SECURITY_CONFIG['BLOCKED_PATTERNS']:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Blocked malicious pattern: {pattern}")
            return ""
    
    # İzin verilen karakterler kontrolü
    if not SECURITY_CONFIG['ALLOWED_CHARS'].match(text):
        # Sadece güvenli karakterleri tut
        text = re.sub(r'[^a-zA-ZığüşöçİĞÜŞÖÇ0-9\s.,!?()-]', '', text)
    
    return text.strip()

def validate_api_request(f):
    """API request validasyon decorator'ı"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Content-Type kontrolü
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        # Request boyutu kontrolü
        if request.content_length and request.content_length > 1024:  # 1KB limit
            return jsonify({"error": "Request too large"}), 413
        
        return f(*args, **kwargs)
    return decorated_function

# API Key güvenliği için düzeltme
load_dotenv()

# Enum'lar ile tip güvenliği
class UserIntent(Enum):
    FAST_SERVICE = "fast_service"
    RECOMMENDATION = "recommendation"
    FOOD_INFO = "food_info"
    GENERAL_HELP = "general_help"

class FoodType(Enum):
    FOOD = "food"
    DRINK = "drink"
    ALL = "all"

class MenuItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    category: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    price: Optional[float] = Field(None, ge=0, le=10000)
    currency: Optional[str] = Field("₺", max_length=5)
    calories: Optional[int] = Field(None, ge=0, le=5000)
    ingredients: Optional[List[str]] = Field(None, max_items=50)
    allergens: Optional[List[str]] = Field(None, max_items=20)
    preparation_time: Optional[int] = Field(None, ge=0, le=180)
    spice_level: Optional[int] = Field(0, ge=0, le=5)
    image_url: Optional[str] = Field(None, max_length=500)
    page_number: Optional[int] = Field(None, ge=1, le=1000)
    available: Optional[bool] = Field(True)

    @validator('ingredients', 'allergens', pre=True)
    def validate_lists(cls, v):
        if v and isinstance(v, list):
            return [sanitize_input(str(item)) for item in v if item]
        return v

    @validator('name', 'category', 'description', pre=True)
    def sanitize_strings(cls, v):
        if v:
            return sanitize_input(str(v))
        return v

class FoodInfoAssistant:
    def __init__(self, menu_data_path="MenuDataset.json"):
        self.menu_items = self.load_menu_data(menu_data_path)
        
        # OpenAI API key kontrolü
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or len(api_key) < 10:
            logger.warning("OpenAI API key not found or invalid. GPT features disabled.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                # API key test
                self.client.models.list()
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
                self.client = None

    def load_menu_data(self, path):
        """Menü verisini güvenli şekilde yükle"""
        try:
            # Dosya yolu güvenlik kontrolü
            if not os.path.exists(path) or not path.endswith('.json'):
                logger.error(f"Invalid or missing menu file: {path}")
                return []
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error("Menu data must be a list")
                return []

            validated_items = []
            for item in data[:1000]:  # Maksimum 1000 item
                try:
                    menu_item = MenuItem(**item)
                    if menu_item.available is not False:
                        validated_items.append(menu_item)
                except ValidationError as e:
                    logger.warning(f"Invalid menu item skipped: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing menu item: {e}")
                    continue

            logger.info(f"Successfully loaded {len(validated_items)} menu items.")
            return validated_items
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in menu file")
            return []
        except Exception as e:
            logger.error(f"Menu loading error: {e}")
            return []

    def detect_user_intent(self, message: str) -> Tuple[UserIntent, Dict]:
        """Kullanıcı niyetini güvenli şekilde tespit et"""
        message_lower = sanitize_input(message).lower()
        
        if not message_lower:
            return UserIntent.GENERAL_HELP, {}

        # Hızlı servis tespiti
        if any(word in message_lower for word in ["hızlı", "acele", "çabuk", "hemen", "acil"]):
            return UserIntent.FAST_SERVICE, {}

        # Öneri tespiti
        elif any(word in message_lower for word in ["öner", "öneri", "tavsiye", "ne yiyeyim", "ne içeyim"]):
            preferences = {
                "glutensiz": any(word in message_lower for word in ["gluten", "glutensiz"]),
                "vegan": any(word in message_lower for word in ["vegan", "vejetaryen"]),
                "alkollü": "alkollü" in message_lower,
                "alkolsüz": "alkolsüz" in message_lower,
                "tatlı": "tatlı" in message_lower,
                "et": "et" in message_lower and "vejetaryen" not in message_lower,
                "balık": "balık" in message_lower,
                "çorba": "çorba" in message_lower,
                "salata": "salata" in message_lower
            }
            return UserIntent.RECOMMENDATION, {"preferences": preferences}

        # Belirli yemek bilgisi
        food_name = self.extract_food_name(message_lower)
        if food_name:
            return UserIntent.FOOD_INFO, {"food_name": food_name}

        return UserIntent.GENERAL_HELP, {}

    def get_fast_foods_only(self) -> List[MenuItem]:
        """Hızlı yiyecekleri güvenli şekilde getir"""
        fast_items = []
        allowed_categories = ["Atıştırmalık", "Balık", "Tavuk", "Burger", "Vegan"]

        for item in self.menu_items:
            if item.category and item.category in allowed_categories:
                if item.preparation_time and item.preparation_time <= 15:
                    fast_items.append(item)

        fast_items.sort(key=lambda x: x.preparation_time or 999)
        return fast_items[:10]  # Maksimum 10 item

    def get_fast_drinks_only(self) -> List[MenuItem]:
        """Hızlı içecekleri güvenli şekilde getir"""
        fast_drinks = []

        for item in self.menu_items:
            if item.category and "içecek" in item.category.lower():
                if item.preparation_time and item.preparation_time <= 5:
                    fast_drinks.append(item)

        fast_drinks.sort(key=lambda x: x.preparation_time or 999)
        return fast_drinks[:10]  # Maksimum 10 item

    def get_smart_recommendations_v2(self, user_message: str, food_type: FoodType = FoodType.ALL) -> List[MenuItem]:
        """Güvenli akıllı öneri sistemi"""
        user_message = sanitize_input(user_message)
        
        if not user_message:
            return []

        # Önce direkt eşleşme kontrolü
        direct_matches = self.get_direct_match_recommendations(user_message)
        if direct_matches:
            if food_type == FoodType.FOOD:
                direct_matches = [item for item in direct_matches
                                if not (item.category and "içecek" in item.category.lower())]
            elif food_type == FoodType.DRINK:
                direct_matches = [item for item in direct_matches
                                if item.category and "içecek" in item.category.lower()]
            return direct_matches[:2]

        # GPT kullan (eğer mevcut ise)
        if self.client:
            return self.get_gpt_recommendations(user_message, food_type)
        else:
            # Fallback: basit kategori filtresi
            return self.get_fallback_recommendations(food_type)

    def get_gpt_recommendations(self, user_message: str, food_type: FoodType) -> List[MenuItem]:
        """GPT ile güvenli öneri alma"""
        if not self.client:
            return self.get_fallback_recommendations(food_type)

        try:
            menu_summary = self.prepare_menu_for_gpt(food_type)
            
            type_instruction = ""
            if food_type == FoodType.FOOD:
                type_instruction = "SADECE yiyecek öner, içecek önerme!"
            elif food_type == FoodType.DRINK:
                type_instruction = "SADECE içecek öner, yiyecek önerme!"

            prompt = f"""
Sen bir restoran asistanısın.

{type_instruction}

Müşteri mesajı: "{user_message[:200]}"  # Prompt injection'a karşı kısalt

Menüdeki ürünler:
{menu_summary[:1500]}  # GPT token limitine dikkat et

En uygun 2 ürün öner. Sadece ürün adlarını listele:
1. [Ürün Adı]
2. [Ürün Adı]
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Sen sadece menüdeki ürünleri öneren bir asistansın."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,  # Token limiti
                timeout=10  # Timeout
            )

            gpt_response = response.choices[0].message.content
            recommended_items = self.extract_recommended_items_from_gpt(gpt_response)

            # Güvenlik filtresi
            if food_type == FoodType.FOOD:
                recommended_items = [i for i in recommended_items if not self.is_drink_or_dessert(i)]
            elif food_type == FoodType.DRINK:
                recommended_items = [i for i in recommended_items if self.is_drink(i)]

            return recommended_items[:2]

        except Exception as e:
            logger.error(f"GPT error: {e}")
            return self.get_fallback_recommendations(food_type)

    def prepare_menu_for_gpt(self, food_type: FoodType) -> str:
        """GPT için güvenli menü hazırlama"""
        menu_text = []

        for item in self.menu_items[:100]:  # Maksimum 100 item
            # Tip filtrelemesi
            if food_type == FoodType.FOOD and item.category and "içecek" in item.category.lower():
                continue
            elif food_type == FoodType.DRINK and not (item.category and "içecek" in item.category.lower()):
                continue

            item_info = f"- {item.name[:50]}"  # İsim kısaltma
            
            if item.category:
                item_info += f" ({item.category[:30]})"
            if item.preparation_time:
                item_info += f" - {item.preparation_time} dk"

            menu_text.append(item_info)

        return "\n".join(menu_text)

    # Güvenlik kontrolleri için yardımcı metodlar
    def is_drink(self, item: MenuItem) -> bool:
        """Güvenli içecek tespiti"""
        if not item.category:
            return False
        return "içecek" in item.category.lower()

    def is_dessert(self, item: MenuItem) -> bool:
        """Güvenli tatlı tespiti"""
        if not item.name:
            return False
        text = item.name.lower()
        dessert_kw = ["tatlı", "dessert", "dondurma", "pasta", "kek"]
        return any(k in text for k in dessert_kw)

    def is_drink_or_dessert(self, item: MenuItem) -> bool:
        return self.is_drink(item) or self.is_dessert(item)

    def get_fallback_recommendations(self, food_type: FoodType, k: int = 2) -> List[MenuItem]:
        """Güvenli fallback önerileri"""
        if food_type == FoodType.FOOD:
            pool = [i for i in self.menu_items if not self.is_drink_or_dessert(i)]
        elif food_type == FoodType.DRINK:
            pool = [i for i in self.menu_items if self.is_drink(i)]
        else:
            pool = [i for i in self.menu_items if self.is_dessert(i)]
        
        pool.sort(key=lambda x: x.preparation_time or 999)
        return pool[:k]

    def get_direct_match_recommendations(self, user_message: str) -> List[MenuItem]:
        """Güvenli direkt eşleşme"""
        message_lower = sanitize_input(user_message).lower()
        direct_matches = []

        # Basit kategori eşleştirme
        direct_patterns = {
            "pizza": ["pizza"],
            "burger": ["burger", "hamburger"],
            "tatlı": ["tatlı", "dessert"],
            "salata": ["salata"],
            "çorba": ["çorba"],
            "içecek": ["içecek", "drink"]
        }

        for category, keywords in direct_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                for item in self.menu_items:
                    if not item.name:
                        continue
                    item_text = item.name.lower()
                    if any(keyword in item_text for keyword in keywords):
                        direct_matches.append(item)
                        if len(direct_matches) >= 2:
                            break
                if direct_matches:
                    break

        return direct_matches[:2]

    def extract_food_name(self, message: str) -> str:
        """Güvenli yemek adı çıkarma"""
        message = sanitize_input(message)
        cleanup_phrases = [
            'hakkında bilgi ver', 'hakkında bilgi', 'bilgi ver',
            'nedir', 'nasıl', 'anlat'
        ]

        food_name = message.lower()
        for phrase in cleanup_phrases:
            food_name = food_name.replace(phrase, '').strip()

        food_name = food_name.replace('?', '').strip()
        return ' '.join(food_name.split())[:100]  # Maksimum 100 karakter

    def find_food_item(self, food_name: str) -> Optional[MenuItem]:
        """Güvenli yemek bulma"""
        if not food_name:
            return None

        food_name_clean = sanitize_input(food_name).lower().strip()

        # Tam eşleşme
        for item in self.menu_items:
            if item.name and item.name.lower().strip() == food_name_clean:
                return item

        # Kısmi eşleşme
        for item in self.menu_items:
            if item.name and food_name_clean in item.name.lower():
                return item

        # Fuzzy matching (güvenli)
        best_match = None
        best_score = 0

        for item in self.menu_items:
            if not item.name:
                continue
            try:
                score = fuzz.partial_ratio(food_name_clean, item.name.lower())
                if score > best_score and score >= 75:
                    best_score = score
                    best_match = item
            except:
                continue

        return best_match

    def extract_recommended_items_from_gpt(self, gpt_response: str) -> List[MenuItem]:
        """GPT yanıtından güvenli ürün çıkarma"""
        if not gpt_response:
            return []

        gpt_response = sanitize_input(gpt_response)
        recommended_items = []

        lines = gpt_response.split('\n')
        for line in lines[:5]:  # Maksimum 5 satır kontrol et
            if any(char.isdigit() for char in line[:3]) and ':' in line:
                try:
                    parts = line.split(':', 1)
                    if len(parts) >= 2:
                        product_part = parts[0]
                        if '.' in product_part:
                            product_name = product_part.split('.', 1)[1].strip()
                        else:
                            product_name = parts[1].strip()

                        found_item = self.find_food_item(product_name)
                        if found_item:
                            recommended_items.append(found_item)
                except:
                    continue

        if not recommended_items and self.menu_items:
            # Güvenli random seçim
            sample_size = min(2, len(self.menu_items))
            recommended_items = random.sample(self.menu_items, sample_size)

        return recommended_items[:2]

    # API için güvenli metodlar
    def format_food_info_for_api(self, item: MenuItem) -> Dict:
        """API için güvenli food info formatı"""
        return {
            "type": "food_info",
            "found": True,
            "item": {
                "id": str(item.id)[:50],
                "name": str(item.name)[:200],
                "description": str(item.description)[:500] if item.description else None,
                "price": float(item.price) if item.price else None,
                "currency": str(item.currency)[:5] if item.currency else "₺",
                "category": str(item.category)[:100] if item.category else None,
                "calories": int(item.calories) if item.calories else None,
                "ingredients": item.ingredients[:20] if item.ingredients else None,
                "allergens": item.allergens[:10] if item.allergens else None,
                "preparation_time": int(item.preparation_time) if item.preparation_time else None
            }
        }

    def format_recommendations_for_api(self, items: List[MenuItem], title: str = "Önerilerim") -> Dict:
        """API için güvenli öneriler formatı"""
        title = sanitize_input(title)[:100]
        formatted_items = []
        
        for item in items[:5]:  # Maksimum 5 öneri
            formatted_items.append({
                "id": str(item.id)[:50],
                "name": str(item.name)[:200],
                "description": str(item.description)[:200] if item.description else None,
                "price": float(item.price) if item.price else None,
                "currency": str(item.currency)[:5] if item.currency else "₺",
                "category": str(item.category)[:100] if item.category else None,
                "preparation_time": int(item.preparation_time) if item.preparation_time else None
            })
        
        return {
            "type": "recommendations",
            "message": title,
            "items": formatted_items
        }

    def process_message_for_api(self, message: str) -> Dict:
        """API için güvenli mesaj işleme"""
        message = sanitize_input(message)
        
        if not message:
            return {
                "type": "error",
                "message": "Geçersiz mesaj"
            }

        # Kullanıcı niyetini tespit et
        intent, context = self.detect_user_intent(message)
        
        if intent == UserIntent.FAST_SERVICE:
            return {
                "type": "fast_service_menu",
                "message": "⚡ Hızlı Servis Menüsü\n\nNe tür bir şey arıyorsunuz?\n\n1️⃣ Yiyecek\n2️⃣ İçecek\n3️⃣ Tatlı\n\n💬 Lütfen rakam yazarak seçim yapın (1, 2 veya 3)",
                "options": [
                    {"id": "1", "text": "Yiyecek", "type": "food"},
                    {"id": "2", "text": "İçecek", "type": "drink"}, 
                    {"id": "3", "text": "Tatlı", "type": "dessert"}
                ]
            }

        elif intent == UserIntent.RECOMMENDATION:
            recommended_items = self.get_smart_recommendations_v2(message, FoodType.ALL)
            return self.format_recommendations_for_api(recommended_items, "Sizin İçin Önerilerim")

        elif intent == UserIntent.FOOD_INFO:
            food_name = context.get("food_name", "")
            found_item = self.find_food_item(food_name)
            if found_item:
                return self.format_food_info_for_api(found_item)
            else:
                return {
                    "type": "food_info",
                    "found": False,
                    "message": f"❌ '{food_name}' menümüzde bulunamadı.\n\n💡 Başka seçenekler:\n• Tatlı öner\n• Hızlı hazırlanan bir şeyler"
                }

        else:  # GENERAL_HELP
            return {
                "type": "help",
                "message": """🤖 Nasıl Yardımcı Olabilirim?

🔸 Belirli bir yemek hakkında bilgi:
   → 'Köfte Bun nedir?'

🔸 Öneri almak:
   → 'Tatlı öner', 'Ne yiyeyim?'

🔸 Hızlı servis:
   → 'Acelem var', 'Hızlı bir şeyler öner'"""
            }

# Flask uygulaması
app = Flask(__name__)

# CORS konfigürasyonu - güvenli
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Sadece belirli origin'ler
     methods=["GET", "POST"],
     allow_headers=["Content-Type"]
)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=[
        f"{SECURITY_CONFIG['MAX_REQUESTS_PER_MINUTE']} per minute",
        f"{SECURITY_CONFIG['MAX_REQUESTS_PER_HOUR']} per hour"
    ]
)

# Güvenli başlatma
try:
    assistant = FoodInfoAssistant("MenuDataset.json")
    if not assistant.menu_items:
        logger.error("No menu items loaded. Check MenuDataset.json file.")
except Exception as e:
    logger.error(f"Failed to initialize assistant: {e}")
    assistant = None

@app.route('/chat', methods=['POST'])
@limiter.limit("30 per minute")  # Ek rate limiting
@validate_api_request
def chat():
    if not assistant:
        return jsonify({"error": "Service temporarily unavailable"}), 503

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message field required"}), 400
        
        user_input = sanitize_input(data.get("message", ""))
        
        if not user_input:
            return jsonify({"error": "Geçersiz mesaj"}), 400
        
        # Request hash için güvenlik
        request_hash = hashlib.sha256(user_input.encode()).hexdigest()
        logger.info(f"Processing request: {request_hash[:8]}")
        
        # Hızlı servis seçimi kontrolü (1, 2, 3)
        if user_input.strip() in ["1", "2", "3"]:
            service_types = {"1": "food", "2": "drink", "3": "dessert"}
            service_type = service_types.get(user_input.strip())
            
            if service_type == "food":
                fast_foods = assistant.get_fast_foods_only()
                result = assistant.format_recommendations_for_api(fast_foods, "⚡ Hızlı Servis Yiyecekler")
            elif service_type == "drink":
                fast_drinks = assistant.get_fast_drinks_only()
                result = assistant.format_recommendations_for_api(fast_drinks, "⚡ Hızlı Servis İçecekler")
            else:  # dessert
                desserts = assistant.get_fallback_recommendations(FoodType.ALL)
                result = assistant.format_recommendations_for_api(desserts, "⚡ Hızlı Servis Tatlılar")
        else:
            # Normal mesaj işleme
            result = assistant.process_message_for_api(user_input)
        
        # Response formatı
        if result["type"] == "fast_service_menu":
            return jsonify({
                "response": result["message"][:SECURITY_CONFIG['MAX_RESPONSE_LENGTH']],
                "type": "menu",
                "options": result["options"]
            })
        
        elif result["type"] == "food_info":
            if result["found"]:
                item = result["item"]
                response_text = f"""🍽️ **{item['name']}**

📝 {item['description'] or 'Açıklama yok'}
💰 Fiyat: {item['price']} {item['currency']}
🏷️ Kategori: {item['category']}"""

                if item.get('calories'):
                    response_text += f"\n🔥 Kalori: {item['calories']} kcal"
                if item.get('preparation_time'):
                    response_text += f"\n⏱️ Hazırlık: {item['preparation_time']} dk"
                
                return jsonify({
                    "response": response_text[:SECURITY_CONFIG['MAX_RESPONSE_LENGTH']]
                })
            else:
                return jsonify({
                    "response": result["message"][:SECURITY_CONFIG['MAX_RESPONSE_LENGTH']]
                })
        
        elif result["type"] == "recommendations":
            response_text = f"🍽️ **{result['message']}**\n\n"
            
            for i, item in enumerate(result['items'][:3], 1):  # Maksimum 3 öneri
                response_text += f"{i}. **{item['name']}**\n"
                if item.get('description'):
                    response_text += f"   {item['description'][:60]}...\n"
                if item.get('price'):
                    response_text +=
                return jsonify({
                "response": response_text[:SECURITY_CONFIG['MAX_RESPONSE_LENGTH']]
            })
        
        else:
            return jsonify({
                "response": result.get("message", "Bir hata oluştu.")[:SECURITY_CONFIG['MAX_RESPONSE_LENGTH']]
            })
    
    except json.JSONDecodeError:
        logger.warning("Invalid JSON received")
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "İç sunucu hatası"}), 500

@app.route('/health', methods=['GET'])
@limiter.limit("10 per minute")
def health():
    """Güvenli health check endpoint"""
    try:
        menu_count = len(assistant.menu_items) if assistant and assistant.menu_items else 0
        openai_status = "available" if assistant and assistant.client else "unavailable"
        
        return jsonify({
            "status": "OK", 
            "menu_items": menu_count,
            "openai": openai_status,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "ERROR"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit error handler"""
    return jsonify({
        "error": "Çok fazla istek. Lütfen daha sonra tekrar deneyin.",
        "retry_after": getattr(e, 'retry_after', None)
    }), 429

@app.errorhandler(413)
def payload_too_large(e):
    """Payload too large error handler"""
    return jsonify({
        "error": "İstek çok büyük. Daha kısa mesaj gönderin."
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Internal server error handler"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "error": "İç sunucu hatası"
    }), 500

# Güvenlik headers
@app.after_request
def after_request(response):
    """Güvenlik headers ekle"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Ana çalıştırıcı
if __name__ == "__main__":
    print(f"🚀 Güvenli server başlatılıyor... Port: 8001")
    
    if assistant and assistant.menu_items:
        print(f"📊 Yüklenen menü öğesi sayısı: {len(assistant.menu_items)}")
        print(f"🤖 OpenAI durumu: {'Aktif' if assistant.client else 'Pasif'}")
    else:
        print("❌ Menü verisi yüklenemedi. Lütfen 'MenuDataset.json' dosyasını kontrol edin.")
    
    # Güvenli Flask çalıştırma
    app.run(
        host='127.0.0.1',  # Sadece localhost
        port=8001, 
        debug=False,  # Production'da debug kapalı
        threaded=True,
        use_reloader=False  # Güvenlik için
    )