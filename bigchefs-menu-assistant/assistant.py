from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, HttpUrl
import random
from enum import Enum

# NOT: OpenAI client kaldırıldı - güvenlik riski nedeniyle

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
    id: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "₺"
    calories: Optional[int] = None
    ingredients: Optional[List[str]] = None
    allergens: Optional[List[str]] = None
    preparation_time: Optional[int] = None
    spice_level: Optional[int] = 0
    image_url: Optional[HttpUrl] = None
    page_number: Optional[int] = None
    available: Optional[bool] = True

class FoodInfoAssistant:
    def __init__(self, menu_data_path="MenuDataset.json"):
        self.menu_items = self.load_menu_data(menu_data_path)
        # OpenAI client kaldırıldı - güvenlik riski

    def load_menu_data(self, path):
        """Menü verisini yükle"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            validated_items = []
            for item in data:
                try:
                    menu_item = MenuItem(**item)
                    if menu_item.available is not False:
                        validated_items.append(menu_item)
                except Exception as e:
                    print(f"⚠️  Ürün yüklenemedi: {e}")
                    continue

            print(f"✅ {len(validated_items)} ürün başarıyla yüklendi.")
            return validated_items
        except Exception as e:
            print(f"❌ Menü yükleme hatası: {e}")
            return []

    def detect_user_intent(self, message: str) -> Tuple[UserIntent, Dict]:
        """Kullanıcı niyetini tespit et"""
        message_lower = message.lower()

        # Hızlı servis tespiti
        if any(word in message_lower for word in ["hızlı", "acele", "çabuk", "hemen", "acil"]):
            return UserIntent.FAST_SERVICE, {}

        # Öneri tespiti
        elif any(word in message_lower for word in ["öner", "öneri", "tavsiye", "ne yiyeyim", "ne içeyim"]):
            # Özel durumları tespit et
            preferences = {
                "glutensiz": "gluten" in message_lower or "glutensiz" in message_lower,
                "vegan": "vegan" in message_lower or "vejetaryen" in message_lower,
                "alkollü": "alkollü" in message_lower or "alkol" in message_lower,
                "alkolsüz": "alkolsüz" in message_lower,
                "tatlı": "tatlı" in message_lower,
                "et": "et" in message_lower and "vejetaryen" not in message_lower,
                "balık": "balık" in message_lower,
                "çorba": "çorba" in message_lower,
                "salata": "salata" in message_lower
            }
            return UserIntent.RECOMMENDATION, {"preferences": preferences}

        # Belirli yemek bilgisi
        food_name = self.extract_food_name(message)
        if food_name:
            return UserIntent.FOOD_INFO, {"food_name": food_name}

        # Genel yardım
        return UserIntent.GENERAL_HELP, {}

    def get_fast_foods_only(self) -> List[MenuItem]:
        """Sadece belirtilen kategorilerdeki hızlı yiyecekleri getir"""
        fast_items = []

        # SADECE BU KATEGORİLERE BAK
        allowed_categories = ["Atıştırmalık", "Balık", "Tavuk", "Burger", "Vegan"]

        for item in self.menu_items:
            # Sadece belirtilen kategorilere bak
            if item.category and item.category in allowed_categories:
                # Hızlı hazırlanıyorsa ekle
                if item.preparation_time and item.preparation_time <= 15:
                    fast_items.append(item)

        # Hazırlık süresine göre sırala
        fast_items.sort(key=lambda x: x.preparation_time or 999)

        return fast_items

    def get_fast_drinks_only(self) -> List[MenuItem]:
        """Sadece hızlı hazırlanan İÇECEKLERİ getir"""
        fast_drinks = []

        for item in self.menu_items:
            # Sadece içecekleri al
            if item.category and "içecek" in item.category.lower():
                if item.preparation_time and item.preparation_time <= 5:  # İçecekler için 5 dk
                    fast_drinks.append(item)

        # Hazırlık süresine göre sırala
        fast_drinks.sort(key=lambda x: x.preparation_time or 999)

        return fast_drinks

    def get_smart_recommendations_by_category(self, message: str, food_type: FoodType = FoodType.ALL) -> List[MenuItem]:
        """Kural tabanlı akıllı öneri sistemi (GPT olmadan)"""
        
        # Önce direkt eşleşme kontrolü
        direct_matches = self.get_direct_match_recommendations(message)
        if direct_matches:
            # Food type'a göre filtrele
            if food_type == FoodType.FOOD:
                direct_matches = [item for item in direct_matches
                                if not (item.category and "içecek" in item.category.lower())]
            elif food_type == FoodType.DRINK:
                direct_matches = [item for item in direct_matches
                                if item.category and "içecek" in item.category.lower()]

            return direct_matches[:2]

        # Kategori bazlı öneriler
        category = "genel"
        if "içecek" in message.lower():
            category = "içecek"
        elif "tatlı" in message.lower():
            category = "tatlı"

        items = self.simple_category_filter(category)
        return items[:2]

    def simple_category_filter(self, category: str) -> List[MenuItem]:
        """Basit kategori filtresi"""
        items = []
        for item in self.menu_items:
            if category == "içecek" and item.category and "içecek" in item.category.lower():
                items.append(item)
            elif category == "tatlı" and ("tatlı" in item.name.lower() or "dessert" in item.name.lower()):
                items.append(item)
            elif category == "genel":
                items.append(item)
        
        # Hazırlık süresine göre sırala
        items.sort(key=lambda x: x.preparation_time or 999)
        return items

    # --- Classifiers ---
    def is_drink(self, item: MenuItem) -> bool:
        """İçecek tespiti: kategori, ad ve açıklamaya bakar."""
        text = " ".join([
            (item.category or ""), (item.name or ""), (item.description or "")
        ]).lower()
        drink_kw = [
            "içecek", "kahve", "espresso", "latte", "cappuccino", "çay",
            "su", "kola", "soda", "ayran", "smoothie", "milkshake",
            "mojito", "kokteyl", "juice", "meyve suyu", "spring water"
        ]
        return any(k in text for k in drink_kw)

    def is_dessert(self, item: MenuItem) -> bool:
        """Tatlı tespiti."""
        text = " ".join([
            (item.category or ""), (item.name or ""), (item.description or "")
        ]).lower()
        dessert_kw = [
            "tatlı", "dessert", "dondurma", "pasta", "kek",
            "brownie", "cheesecake", "tiramisu", "sufle", "baklava"
        ]
        return any(k in text for k in dessert_kw)

    def is_drink_or_dessert(self, item: MenuItem) -> bool:
        return self.is_drink(item) or self.is_dessert(item)

    # --- Fallbacks ---
    def get_fallback_recommendations(self, food_type: FoodType, k: int = 2) -> List[MenuItem]:
        """En hızlı uygun ürünleri döndür."""
        if food_type == FoodType.FOOD:
            pool = [i for i in self.menu_items if not self.is_drink_or_dessert(i)]
        elif food_type == FoodType.DRINK:
            pool = [i for i in self.menu_items if self.is_drink(i)]
        else:  # FoodType.ALL -> tatlı senaryosu için
            pool = [i for i in self.menu_items if self.is_dessert(i)]
        pool.sort(key=lambda x: x.preparation_time or 999)
        return pool[:k]

    def get_direct_match_recommendations(self, user_message: str) -> List[MenuItem]:
        """Direkt eşleşme önerileri"""
        message_lower = user_message.lower()
        direct_matches = []

        # Glutensiz kontrolü - öncelikli
        if any(word in message_lower for word in ["glutensiz", "gluten free", "gluten içermeyen", "gluten hassasiyeti", "çölyak"]):
            # Sadece glutensiz makarna ve gluten free sufle'yi bul
            for item in self.menu_items:
                item_name_lower = item.name.lower()

                # Tam eşleşme kontrolü
                if "glutensiz makarna" in item_name_lower or "gluten free sufle" in item_name_lower:
                    direct_matches.append(item)

                # Eğer tam eşleşme bulunamazsa, alternatif glutensiz ürünleri kontrol et
                elif not direct_matches and ("glutensiz" in item_name_lower or "gluten free" in item_name_lower):
                    direct_matches.append(item)

            # Sadece ilk 2 sonucu döndür
            return direct_matches[:2]

        # Diğer direkt eşleşme senaryoları
        direct_patterns = {
            "vegan": ["vegan", "vejetaryen"],
            "köfte": ["köfte"],
            "pizza": ["pizza"],
            "burger": ["burger", "hamburger"],
            "tatlı": ["tatlı", "dessert"],
            "salata": ["salata"],
            "çorba": ["çorba", "soup"],
            "makarna": ["makarna", "pasta"],
            "döner": ["döner"],
            "lahmacun": ["lahmacun"],
            "pide": ["pide"],
            "kebap": ["kebap", "şiş"],
            "balık": ["balık", "fish"],
            "tavuk": ["tavuk", "chicken"],
            "et": ["et", "biftek", "steak"]
        }

        # Hangi kategorinin arandığını bul
        for category, keywords in direct_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                # Bu kategorideki ürünleri bul
                for item in self.menu_items:
                    item_text = f"{item.name} {item.description or ''} {item.category or ''}".lower()

                    if category == "vegan":
                        if ("vegan" in item_text or "vejetaryen" in item_text or
                            (item.category and "vegan" in item.category.lower())):
                            direct_matches.append(item)
                    else:
                        if any(keyword in item_text for keyword in keywords):
                            direct_matches.append(item)

                if direct_matches:
                    break

        return direct_matches[:2]

    def extract_food_name(self, message: str) -> str:
        """Mesajdan yemek adını çıkar"""
        message = message.strip()

        cleanup_phrases = [
            'hakkında bilgi ver', 'hakkında bilgi', 'bilgi ver', 'anlat',
            'açıkla', 'nedir', 'nasıl', 'tarif et', 'özellikleri',
            'detayları', 'detay ver', 'söyle', 'göster', 'anlatsana'
        ]

        food_name = message.lower()

        for phrase in cleanup_phrases:
            food_name = food_name.replace(phrase, '').strip()

        food_name = food_name.replace('?', '').strip()
        food_name = ' '.join(food_name.split())

        return food_name

    def find_food_item(self, food_name: str) -> Optional[MenuItem]:
        """Yemek adına göre ürün bul"""
        if not food_name:
            return None

        food_name_clean = food_name.lower().strip()

        # 1. TAM EŞLEŞME
        for item in self.menu_items:
            if item.name.lower().strip() == food_name_clean:
                return item

        # 2. İÇERME KONTROLÜ
        for item in self.menu_items:
            if food_name_clean in item.name.lower():
                return item

        # 3. Basit benzerlik kontrolü (rapidfuzz kaldırıldı)
        best_match = None
        for item in self.menu_items:
            if any(word in item.name.lower() for word in food_name_clean.split() if len(word) > 2):
                if not best_match:
                    best_match = item

        return best_match

    # ==================== API İÇİN YENİ METODLAR ====================

    def format_food_info_for_api(self, item: MenuItem) -> Dict:
        """Yemek bilgilerini API için formatla"""
        return {
            "type": "food_info",
            "found": True,
            "item": {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "price": int(item.price) if item.price and float(item.price).is_integer() else item.price,
                "currency": item.currency or "₺",
                "category": item.category,
                "calories": item.calories,
                "ingredients": item.ingredients,
                "allergens": item.allergens,
                "preparation_time": item.preparation_time,
                "image_url": str(item.image_url) if item.image_url else None
            }
        }

    def format_recommendations_for_api(self, items: List[MenuItem], title: str = "Önerilerim") -> Dict:
        """Önerileri API için formatla"""
        formatted_items = []
        for item in items:
            formatted_items.append({
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "price": int(item.price) if item.price and float(item.price).is_integer() else item.price,
                "currency": item.currency or "₺",
                "category": item.category,
                "calories": item.calories,
                "ingredients": item.ingredients,
                "allergens": item.allergens,
                "preparation_time": item.preparation_time,
                "image_url": str(item.image_url) if item.image_url else None
            })
        
        return {
            "type": "recommendations",
            "message": title,
            "items": formatted_items
        }

    def get_fast_service_recommendations_for_api(self, service_type: str) -> Dict:
        """Hızlı servis önerilerini API için formatla"""
        if service_type == "food":
            fast_foods = self.get_fast_foods_only()
            if not fast_foods:
                return {
                    "type": "recommendations",
                    "message": "⚠️ Hızlı Servis Yiyecek",
                    "items": [],
                    "note": "Belirtilen kategorilerde hızlı hazırlanan yiyecek bulunamadı."
                }
            return self.format_recommendations_for_api(fast_foods[:2], "⚡ Hızlı Servis Yiyecekler")
        
        elif service_type == "drink":
            fast_drinks = self.get_fast_drinks_only()
            if not fast_drinks:
                # Fallback: tüm içecekleri al ve en hızlı olanları seç
                all_drinks = [item for item in self.menu_items
                            if item.category and "içecek" in item.category.lower()]
                all_drinks.sort(key=lambda x: x.preparation_time or 999)
                fast_drinks = all_drinks[:2]
            return self.format_recommendations_for_api(fast_drinks[:2], "⚡ Hızlı Servis İçecekler")
        
        elif service_type == "dessert":
            desserts = [item for item in self.menu_items if self.is_dessert(item)]
            desserts.sort(key=lambda x: x.preparation_time or 999)
            return self.format_recommendations_for_api(desserts[:2], "⚡ Hızlı Servis Tatlılar")
        
        return {
            "type": "error",
            "message": "Geçersiz servis tipi"
        }

    def process_message_for_api(self, message: str) -> Dict:
        """API için mesaj işleme - JSON response döndürür"""
        
        # Kullanıcı niyetini tespit et
        intent, context = self.detect_user_intent(message)
        
        if intent == UserIntent.FAST_SERVICE:
            # Hızlı servis - interaktif menü yerine akıllı tespit
            message_lower = message.lower()
            
            if any(word in message_lower for word in ["yiyecek", "yemek", "açım"]):
                return self.get_fast_service_recommendations_for_api("food")
            elif any(word in message_lower for word in ["içecek", "su", "kahve", "çay"]):
                return self.get_fast_service_recommendations_for_api("drink")
            elif any(word in message_lower for word in ["tatlı", "dessert"]):
                return self.get_fast_service_recommendations_for_api("dessert")
            else:
                # Genel hızlı servis - yiyecek odaklı
                return self.get_fast_service_recommendations_for_api("food")

        elif intent == UserIntent.RECOMMENDATION:
            # Tercihlere göre öneri yap (GPT olmadan)
            recommended_items = self.get_smart_recommendations_by_category(message, FoodType.ALL)
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
                    "message": f"❌ '{food_name}' menümüzde bulunamadı.\n\n💡 Başka seçenekler:\n• Tatlı öner\n• Gluten içermeyen yemekler\n• Hızlı hazırlanan bir şeyler"
                }

        else:  # GENERAL_HELP
            help_message = """🤖 Nasıl Yardımcı Olabilirim?

🔸 Belirli bir yemek hakkında bilgi:
   → 'Köfte Bun nedir?'

🔸 Öneri almak:
   → 'Tatlı öner', 'Ne yiyeyim?'

🔸 Hızlı servis:
   → 'Acelem var', 'Hızlı bir şeyler öner'

🔸 Özel durumlar:
   → 'Gluten hassasiyetim var', 'Vejetaryen menü'"""

            return {
                "type": "help",
                "message": help_message
            }


# ==================== FLASK UYGULAMASI ====================
app = Flask(__name__)
CORS(app)

# Menü datasetini dosyadan yükle
assistant = FoodInfoAssistant("MenuDataset.json")

@app.route('/chat', methods=['POST'])  
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        
        if not user_input:
            return jsonify({"error": "Mesaj boş olamaz"}), 400
        
        # API metodunu kullan
        result = assistant.process_message_for_api(user_input)
        
        # Response tipine göre formatla
        if result["type"] == "food_info":
            if result["found"]:
                item = result["item"]
                response_text = f"""🍽️ **{item['name']}**

📝 {item['description'] or 'Açıklama yok'}
💰 Fiyat: {item['price']} {item['currency']}
🏷️ Kategori: {item['category']}"""

                if item['calories']:
                    response_text += f"\n🔥 Kalori: {item['calories']} kcal"
                if item['ingredients']:
                    response_text += f"\n🥄 İçindekiler: {', '.join(item['ingredients'])}"
                if item['allergens']:
                    response_text += f"\n⚠️ Alerjenler: {', '.join(item['allergens'])}"
                
                return jsonify({
                    "response": response_text,
                    "image": item.get("image_url")
                })
            else:
                return jsonify({
                    "response": result["message"]
                })
        
        elif result["type"] == "recommendations":
            response_text = f"🍽️ **{result['message']}**\n\n"
            
            for i, item in enumerate(result['items'], 1):
                response_text += f"{i}. **{item['name']}**\n"
                if item['description']:
                    response_text += f"   {item['description'][:60]}...\n"
                if item['price']:
                    response_text += f"   💰 {item['price']} {item['currency']}\n"
                if item['preparation_time']:
                    response_text += f"   ⏱️ Hazırlık: {item['preparation_time']} dk\n"
                response_text += "\n"
            
            return jsonify({
                "response": response_text
            })
        
        else:
            return jsonify({
                "response": result.get("message", "Bir hata oluştu.")
            })
    
    except Exception as e:
        print(f"Hata detayı: {str(e)}")  # Terminal'de hata detayını göster
        return jsonify({"error": f"Hata oluştu: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "OK", 
        "menu_items": len(assistant.menu_items) if assistant.menu_items else 0
    })

# ANA ÇALIŞTIRICI
if __name__ == "__main__":
    print(f"🚀 Server başlatılıyor... Port: ")
    print(f"📊 Yüklenen menü öğesi sayısı: {len(assistant.menu_items) if assistant.menu_items else 0}")
    
    # Flask versiyonu
    app.run(host='0.0.0.0', port=, debug=)